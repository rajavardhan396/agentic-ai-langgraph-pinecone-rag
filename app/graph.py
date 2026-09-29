from typing import TypedDict
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from .config import settings
from .vectorstore import retrieve

class RAGState(TypedDict, total=False):
    query: str
    matches: list
    context: list[str]
    answer: str
    confidence: float

llm = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You answer questions strictly from the supplied Agentic AI eBook context.
Use ONLY the context. Do not use outside knowledge. If the context is insufficient, say:
"I don't have enough information in the eBook to answer that."
Never invent facts, page numbers, citations, examples, or definitions."""

def retrieve_node(state: RAGState):
    matches = retrieve(state["query"])
    context = [m.metadata.get("text", "") for m in matches if m.metadata and m.metadata.get("text")]
    return {"matches": matches, "context": context}

def generate_node(state: RAGState):
    joined = "\n\n--- CHUNK ---\n\n".join(state.get("context", []))
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{joined}\n\nQUESTION:\n{state['query']}"
    return {"answer": llm.invoke(prompt).content.strip()}

def grade_node(state: RAGState):
    if not state.get("context"):
        return {"confidence": 0.0}
    prompt = f"""Grade whether the answer is fully supported by the context.
Return JSON only: {{"supported": true/false, "score": 0.0-1.0}}
CONTEXT:
{chr(10).join(state['context'])}
ANSWER:
{state['answer']}"""
    try:
        data = json.loads(llm.invoke(prompt).content)
        score = float(data.get("score", 0.0))
        if not data.get("supported", False):
            score = min(score, 0.49)
        return {"confidence": max(0.0, min(1.0, score))}
    except Exception:
        return {"confidence": 0.0}

def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("grade", grade_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "grade")
    builder.add_edge("grade", END)
    return builder.compile()

rag_graph = build_graph()

def ask(query: str) -> dict:
    result = rag_graph.invoke({"query": query})
    confidence = float(result.get("confidence", 0.0))
    answer = result.get("answer", "")
    if confidence < 0.55:
        answer = "I don't have enough information in the eBook to answer that."
    return {
        "query": query,
        "final_answer": answer,
        "retrieved_context_chunks": result.get("context", []),
        "confidence_score": round(confidence, 2),
    }
