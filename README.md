# Document Q&A: Retrieval-Augmented Generation (RAG)

Upload PDFs and ask questions in natural language. Answers are generated **only from your documents**, with **page-level citations**. If the answer isn't in the documents, the system says so instead of guessing.

##LiveDemo: https://rag-document-qa-jadl.onrender.com

## Architecture

```
INGESTION   PDF -> PyMuPDF parse -> chunking (800 chars, 150 overlap) -> embeddings -> ChromaDB (+ metadata)
QUERY       question -> embedding -> cosine similarity search (top-k)
GENERATION  LangGraph:  retrieve -> [relevant?] -> generate (LLM + prompt) -> answer + sources
                                         \-> fallback ("couldn't find this")
```

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Explicit, testable state machine with a conditional "no relevant context" branch |
| Framework | **LangChain** | Loaders, splitters, prompt templates, model wrappers |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) | Free, local, fast |
| Vector DB | **ChromaDB** (cosine) | Local and persistent, zero setup |
| LLM | Gemini (default), Claude or OpenAI, switchable in `.env` | Provider-agnostic design |
| API | **FastAPI** | Upload / ask / list / delete endpoints |
| Prompts | `app/prompts.py` | Small versioned prompt library |

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# edit .env and add your API key
uvicorn app.main:app --reload
# open http://localhost:8000
```

Minimal `.env` for Gemini:

```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-from-aistudio.google.com
GEMINI_MODEL=<a current Gemini Flash model name>
```

Docker: `docker build -t docqa . && docker run -p 8000:8000 --env-file .env docqa`

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/upload` | Upload a PDF (parse, chunk, embed, store) |
| POST | `/api/ask` | `{"question": "...", "source": "optional.pdf"}` -> answer + sources |
| GET | `/api/documents` | List indexed documents |
| DELETE | `/api/documents/{name}` | Remove a document |

## Design decisions

- **Chunk overlap** prevents answers from being cut at chunk boundaries.
- **Relevance threshold** (`MIN_RELEVANCE`): if nothing retrieved is similar enough, the LLM is *not called* and the system answers "not found". This reduces hallucination and saves cost.
- **Prompt rules** force context-only answers with `[file, p.N]` citations.
- **Metadata** (source, page, chunk_id) stored with every vector enables citations and per-document filtering.
- **Re-uploading** a file replaces its old chunks (no duplicates).
- **Scanned PDFs** are detected and rejected with a clear message (OCR not included).

## Evaluate retrieval

Put questions about your own PDF in `eval/questions.json`, then run:

```bash
python -m eval.eval      # prints PASS/FAIL per question and Hit@k
```

Result on my test document: **Hit@4 = __ / __**

## Limitations and next steps

- Add OCR (Tesseract) for scanned PDFs
- Hybrid search (BM25 + vectors) and a reranker
- Conversation memory (follow-up questions)
- Streaming responses
- Support DOCX / TXT
- Authentication and per-user collections
