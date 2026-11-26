from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.graph import ask
from app.ingest import ingest_pdf
from app.vectorstore import list_documents, delete_document

app = FastAPI(title="Document Q&A (RAG)", version="1.0.0")
STATIC = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    source: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": config.LLM_PROVIDER}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = file.file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB.")
    path = config.UPLOAD_DIR / name
    path.write_bytes(data)
    try:
        chunks = ingest_pdf(path)
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    return {"name": name, "chunks": chunks}


@app.get("/api/documents")
def documents():
    return list_documents()


@app.delete("/api/documents/{name}")
def remove(name: str):
    delete_document(name)
    (config.UPLOAD_DIR / Path(name).name).unlink(missing_ok=True)
    return {"deleted": name}


@app.post("/api/ask")
def ask_endpoint(req: AskRequest):
    if not list_documents():
        raise HTTPException(400, "Upload a document first.")
    try:
        return ask(req.question, req.source)
    except Exception as e:  # e.g. missing API key
        raise HTTPException(500, f"Generation failed: {e}")
