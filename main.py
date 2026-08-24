"""
PgBrain - FastAPI Backend (Railway Deployment)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import groq
import sys

print("Python version:", sys.version)
print("Starting PgBrain...")

load_dotenv()

# Database URL (hardcoded for Railway)
DATABASE_URL = "postgresql://neondb_owner:npg_GR9WZ3XFpwOx@ep-floral-fog-auuc311p.c-10.us-east-1.aws.neon.tech/neondb?sslmode=require"

# Initialize FastAPI
app = FastAPI(title="PgBrain API")

# CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
print("📥 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Groq client
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.get("/")
def root():
    return {"message": "PgBrain API is running."}

@app.post("/query", response_model=QueryResponse)
def query_pgbrain(request: QueryRequest):
    try:
        query_embedding = model.encode(request.query).tolist()
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                dc.content,
                sd.title
            FROM document_chunks dc
            JOIN source_documents sd ON dc.document_id = sd.id
            ORDER BY dc.embedding <=> %s::vector
            LIMIT 3;
        """, (query_embedding,))
        results = cur.fetchall()
        conn.close()
        if not results:
            return QueryResponse(
                answer="I don't have enough information.",
                sources=[]
            )
        context = "\n\n".join([f"[{title}]: {content}" for content, title in results])
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": "You are PgBrain. Answer in English using ONLY the provided context. If the context doesn't contain the answer, say 'I don't have enough information.'"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"}
            ],
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        sources = [title for content, title in results]
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
