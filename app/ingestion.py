from pathlib import Path
import httpx
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import settings

def download_pdf(force: bool = False) -> Path:
    path = Path(settings.pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(settings.pdf_url)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path

def load_and_chunk(pdf_path: str | None = None) -> list[Document]:
    path = Path(pdf_path or settings.pdf_path)
    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": path.name, "page": page_no}))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
    return chunks
