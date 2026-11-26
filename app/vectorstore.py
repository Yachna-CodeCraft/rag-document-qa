from functools import lru_cache
from collections import Counter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import config


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="documents",
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def list_documents() -> list[dict]:
    data = get_vectorstore().get(include=["metadatas"])
    counts = Counter(m.get("source", "unknown") for m in data["metadatas"])
    return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]


def delete_document(name: str) -> None:
    get_vectorstore().delete(where={"source": name})
