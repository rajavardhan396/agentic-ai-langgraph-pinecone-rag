from fastapi import FastAPI, HTTPException
from .models import QueryRequest, QueryResponse
from .graph import ask

app = FastAPI(title="Agentic AI eBook RAG", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return ask(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
