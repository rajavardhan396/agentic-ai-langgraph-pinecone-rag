from app.ingestion import download_pdf, load_and_chunk
from app.vectorstore import upsert_documents

if __name__ == "__main__":
    path = download_pdf()
    docs = load_and_chunk(str(path))
    count = upsert_documents(docs)
    print(f"Indexed {count} chunks from {path}")
