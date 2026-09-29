from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from .config import settings

def get_embeddings():
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

def get_pinecone_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    names = pc.list_indexes().names()
    if settings.pinecone_index_name not in names:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc.Index(settings.pinecone_index_name)

def upsert_documents(documents):
    index = get_pinecone_index()
    embeddings = get_embeddings()
    vectors = []
    for doc in documents:
        vectors.append({
            "id": f"chunk-{doc.metadata['chunk_id']}",
            "values": embeddings.embed_query(doc.page_content),
            "metadata": {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
                "chunk_id": doc.metadata.get("chunk_id", 0),
            },
        })
    for start in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[start:start + 100])
    return len(vectors)

def retrieve(query: str, top_k: int | None = None):
    index = get_pinecone_index()
    qvec = get_embeddings().embed_query(query)
    result = index.query(vector=qvec, top_k=top_k or settings.top_k, include_metadata=True)
    return list(result.matches)
