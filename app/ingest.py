from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.vectorstore import get_vectorstore, delete_document


def ingest_pdf(path: Path) -> int:
    """Parse -> chunk -> embed -> store. Returns number of chunks stored."""
    pages = PyMuPDFLoader(str(path)).load()
    if not any(p.page_content.strip() for p in pages):
        raise ValueError(
            "No extractable text found. This looks like a scanned PDF - run OCR first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]

    for i, c in enumerate(chunks):
        c.metadata = {
            "source": path.name,
            "page": int(c.metadata.get("page", 0)) + 1,  # 1-based for humans
            "chunk_id": i,
        }

    delete_document(path.name)  # re-uploading the same file replaces it
    get_vectorstore().add_documents(
        chunks, ids=[f"{path.name}::{i}" for i in range(len(chunks))]
    )
    return len(chunks)
