# Agentic AI eBook — LangGraph + Pinecone RAG Chatbot

Custom Python RAG chatbot for the AI Engineer interview assignment.

## Architecture
```
PDF -> PyPDF -> RecursiveCharacterTextSplitter -> OpenAI Embeddings -> Pinecone
                                                     |
Query -> LangGraph START -> Retrieve -> Generate -> Ground/Confidence -> END
                                                     |
                                                     v
                                                  FastAPI
```

## Setup
```bash
git clone https://github.com/rajavardhan396/agentic-ai-langgraph-pinecone-rag.git
cd agentic-ai-langgraph-pinecone-rag
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Create .env from .env.example and set OPENAI_API_KEY and PINECONE_API_KEY.

## Ingestion
```bash
python -m scripts.ingest
```
The script downloads the Agentic AI eBook, extracts page text, creates 800-character chunks with 100-character overlap, embeds them, and stores chunk text/page/source metadata in Pinecone. The PDF is excluded from Git.

## API
```bash
uvicorn app.main:app --reload
```
Swagger: http://127.0.0.1:8000/docs

POST /query with:
```json
{"query":"What is Agentic AI?"}
```

Response:
```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["..."],
  "confidence_score": 0.92
}
```

## Grounding
Generation is restricted to retrieved eBook context. A separate LangGraph grading node assigns a 0–1 support score. Scores below 0.55 return: "I don't have enough information in the eBook to answer that."

The required out-of-scope question "What is the capital of France?" therefore should not be answered from outside knowledge.

## Validation
1. Core definition of Agentic AI
2. Architecture/components
3. Industry use cases
4. Agentic AI vs traditional generative AI chatbots
5. Challenges/limitations
6. Capital of France

## Structure
```
app/
  config.py
  graph.py
  ingestion.py
  main.py
  models.py
  vectorstore.py
scripts/ingest.py
tests/test_graph.py
requirements.txt
.env.example
.gitignore
```

## Engineering decisions
- Recursive chunking: 800/100
- OpenAI text-embedding-3-small
- Pinecone cosine similarity
- LangGraph StateGraph
- FastAPI + Pydantic
- API keys and PDF excluded from Git
