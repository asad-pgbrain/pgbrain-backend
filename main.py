"""
PgBrain - FastAPI Backend with Multi-Provider Fallback, Connection Pooling & Database Connection Management
"""
from functools import lru_cache
import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncpg
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import groq
import requests
import google.generativeai as genai
import time
import json
from contextlib import asynccontextmanager
import sys
from cryptography.fernet import Fernet
import psycopg
@lru_cache(maxsize=50)
def cached_query(query_hash):
    """Cache results for repeated queries"""
    return None  # Implementation below

# In query function, add cache check:
query_hash = hashlib.md5(request.query.encode()).hexdigest()
# Check cache first, if found return cached result2

print("Python version:", sys.version)
print("Starting PgBrain with Connection Pooling & Streaming...")

load_dotenv()

# Database URL
DATABASE_URL = "postgresql://neondb_owner:npg_GR9WZ3XFpwOx@ep-floral-fog-auuc311p.c-10.us-east-1.aws.neon.tech/neondb?sslmode=require"

# Encryption key for database credentials
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print(f"⚠️ No ENCRYPTION_KEY found. Generated new key: {ENCRYPTION_KEY}")
cipher = Fernet(ENCRYPTION_KEY.encode())

# Global connection pool
db_pool = None

# Lifespan manager for connection pool
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    print("🔌 Creating database connection pool...")
    db_pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        timeout=30
    )
    print("✅ Connection pool created successfully!")
    yield
    await db_pool.close()
    print("🔌 Connection pool closed.")

# Initialize FastAPI with lifespan
app = FastAPI(title="PgBrain API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
# Lazy loading embedding model
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print("📥 Loading embedding model (first time)...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

# Groq client
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Provider configuration
PROVIDERS = [
    PROVIDERS = [
    PROVIDERS = [
    PROVIDERS = [
    {"name": "groq", "model": "openai/gpt-oss-20b", "priority": 1, "api_key": os.getenv("GROQ_API_KEY")},
    {"name": "groq_fallback", "model": "llama-3.1-8b-instant", "priority": 2, "api_key": os.getenv("GROQ_API_KEY")},
    {"name": "gemini", "model": "gemini-1.5-flash", "priority": 3},
    {"name": "openrouter", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "priority": 4, "api_key": os.getenv("OPENROUTER_API_KEY")}
]

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    provider: str = ""

class DatabaseConnection(BaseModel):
    host: str
    port: int
    database: str
    username: str
    password: str
    user_id: str

def call_groq(model, prompt, api_key):
    client = groq.Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are PgBrain. Answer in English using ONLY the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content

def call_gemini(prompt):
    response = gemini_model.generate_content(prompt)
    return response.text

def call_openrouter(model, prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are PgBrain. Answer in English using ONLY the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

def call_llm_with_fallback(prompt, max_retries=2):
    for provider in sorted(PROVIDERS, key=lambda x: x["priority"]):
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempting with: {provider['name']} (attempt {attempt+1})")
                if provider["name"] in ["groq", "groq_fallback"]:
                    answer = call_groq(provider["model"], prompt, provider["api_key"])
                elif provider["name"] == "gemini":
                    answer = call_gemini(prompt)
                elif provider["name"] == "openrouter":
                    answer = call_openrouter(provider["model"], prompt, provider["api_key"])
                else:
                    continue
                print(f"✅ Success with: {provider['name']}")
                return answer, provider["name"]
            except Exception as e:
                print(f"❌ Provider {provider['name']} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        print(f"⚠️ All attempts failed for provider: {provider['name']}")
    raise Exception("All providers failed after max retries")

async def get_similar_chunks(query_embedding):
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    async with db_pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT 
                dc.content,
                sd.title
            FROM document_chunks dc
            JOIN source_documents sd ON dc.document_id = sd.id
            ORDER BY dc.embedding <=> $1::vector
            LIMIT 3;
        """, embedding_str)
        return [(r["content"], r["title"]) for r in results]

def generate_stream(prompt):
    try:
        answer, provider = call_llm_with_fallback(prompt)
        yield f"data: {json.dumps({'answer': answer, 'provider': provider, 'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
@app.get("/")
def root():
    return {"message": "PgBrain API is running. Connection pooling + streaming enabled."}

@app.post("/query", response_model=QueryResponse)
async def query_pgbrain(request: QueryRequest):
    try:
        model = get_embedding_model()
# Preprocess query — remove trailing '?' for consistent speed
raw_query = request.query.strip()
if raw_query.endswith('?'):
    raw_query = raw_query[:-1]  # Remove trailing '?'
        query_embedding = model.encode(request.query).tolist()
        results = await get_similar_chunks(query_embedding)
        if not results:
            return QueryResponse(
                answer="I don't have enough information.",
                sources=[],
                provider=""
            )
        context = "\n\n".join([f"[{title}]: {content}" for content, title in results])
        prompt = f"Context:\n{context}\n\nQuestion: {raw_query}"
        answer, provider = call_llm_with_fallback(prompt)
        sources = [title for content, title in results]
        return QueryResponse(answer=answer, sources=sources, provider=provider)
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/connect")
async def connect_database(conn: DatabaseConnection):
    try:
        # Test connection
        test_conn = psycopg2.connect(
            host=conn.host,
            port=conn.port,
            database=conn.database,
            user=conn.username,
            password=conn.password
        )
        test_conn.close()
        
        # Encrypt credentials
        creds = {
            "host": conn.host,
            "port": conn.port,
            "database": conn.database,
            "username": conn.username,
            "password": conn.password
        }
        encrypted = cipher.encrypt(json.dumps(creds).encode())
        
        # Store in database
        async with db_pool.acquire() as pool_conn:
            await pool_conn.execute("""
                CREATE TABLE IF NOT EXISTS user_connections (
                    user_id TEXT PRIMARY KEY,
                    encrypted_creds TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            await pool_conn.execute("""
                INSERT INTO user_connections (user_id, encrypted_creds)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE SET
                    encrypted_creds = $2,
                    updated_at = NOW()
            """, conn.user_id, encrypted.decode())
        
        return {"status": "success", "message": "Database connected successfully!"}
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/connections/{user_id}")
async def get_user_connection(user_id: str):
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(
                "SELECT encrypted_creds FROM user_connections WHERE user_id = $1",
                user_id
            )
            if not result:
                return {"connected": False}
            decrypted = cipher.decrypt(result["encrypted_creds"].encode())
            creds = json.loads(decrypted.decode())
            return {"connected": True, "host": creds["host"], "database": creds["database"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(  
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    try:
        model = get_embedding_model()
        query_embedding = model.encode(request.query).tolist()
        results = await get_similar_chunks(query_embedding)
        
        if not results:
            return {"answer": "I don't have enough information.", "sources": []}
        
        context = "\n\n".join([f"[{title}]: {content}" for content, title in results])
        prompt = f"Context:\n{context}\n\nQuestion: {request.query}"
        
        return StreamingResponse(
            generate_stream(prompt),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    try:
        model = get_embedding_model()
        query_embedding = model.encode(request.query).tolist()
        results = await get_similar_chunks(query_embedding)
        
        if not results:
            return {"answer": "I don't have enough information.", "sources": []}
        
        context = "\n\n".join([f"[{title}]: {content}" for content, title in results])
        prompt = f"Context:\n{context}\n\nQuestion: {request.query}"
        
        return StreamingResponse(
            generate_stream(prompt),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
