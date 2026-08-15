"""
PgBrain - FastAPI Backend (Groq Integration)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import groq

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="PgBrain API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database config (Neon cloud)
DATABASE_URL = os.getenv("DATABASE_URL")

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
            return QueryResponse(answer="I don't have enough information.", sources=[])
        context = "\n\n".join([f"[{title}]: {content}" for content, title in results])
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are PgBrain. Answer in English using ONLY the provided context."},
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
