## RAG System (FastAPI + Streamlit)
**Developed by Leticia Bossatto Marchezi**

Document Q&A using Retrieval-Augmented Generation. Upload one or more PDFs, we chunk and embed them, store FAISS per-file indexes, and answer questions using your selected LLM provider.

### Project layout
- `api/`
  - `main.py`: FastAPI app wiring and CORS
  - `routes/main.py`: Endpoints (`/documents`, `/question`, `/models`, `/health`)
  - `services/embeddings.py`: PDF parsing (pypdf), chunking, embeddings (OpenAI), FAISS storage (per-file)
  - `services/llm.py`: LLM abstraction (OpenAI, Gemini, Hugging Face)
- `frontend/`
  - `main/frontend.py`: Streamlit UI
  - `main/routers.py`: HTTP client to call the API
- `docker-compose.yml`: Runs API and frontend; persists FAISS to `./vector_store`

### Requirements
- Docker Desktop
- API keys for the providers you will use
  - OpenAI: `OPENAI_API_KEY` (required for embeddings)
  - Google AI (Gemini): `GOOGLE_API_KEY`

### Environment variables (compose)
Compose already passes provider keys via `environment:`. Set them in your Windows environment or a `.env` file in the project root so compose can substitute:

```
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_google_key
```

### Run
```
docker-compose up --build
```

- API: `http://localhost:8000` (Docs at `/docs`, health at `/health`)
- Frontend: `http://localhost:8501`
- Vector store: `./vector_store` (mounted into the API container)

### macOS/Linux quick commands
- Start services (attached logs):
```
docker-compose up --build
```
- Upload two PDFs via curl (paths for macOS):
```
curl -F "files=@/Users/<you>/Documents/a.pdf" -F "files=@/Users/<you>/Documents/b.pdf" http://localhost:8000/documents | cat
```
- Ask a question via curl:
```
curl -H "Content-Type: application/json" \
  -d '{"question":"What is X?","llm_provider":"openai"}' \
  http://localhost:8000/question | cat
```


### Windows:
```
curl -F "files=@C:\path\to\a.pdf" -F "files=@C:\path\to\b.pdf" http://localhost:8000/documents | cat
```

Example question:
```
curl -H "Content-Type: application/json" -d "{\"question\":\"What is X?\",\"llm_provider\":\"openai\"}" http://localhost:8000/question | cat
```

### Using the system
1) Open the frontend at `http://localhost:8501`
2) Upload one or more PDFs and click “Process Documents”
   - Each file is indexed under `vector_store/<filename-stem>/index.faiss|index.pkl`
   - Re-uploading the same filename is skipped (no reprocessing)
3) Choose provider (OpenAI/Gemini) and model (from `/models`)
4) Ask a question; the system retrieves similar chunks across all indexed files and generates an answer with references

### API endpoints
- `GET /health` → `{ "status": "ok" }`
- `GET /models` → `{ "openai": [...], "gemini": [...] }`
- `POST /documents` (multipart)
  - Field name: `files` (repeatable)
  - Returns summary: processed, skipped, total_chunks, per-file results
- `POST /question` (JSON)
  - `{ "question": "...", "llm_provider": "openai|gemini", "model": "optional" }`
  - Returns `{ "answer": str, "references": [str] }`

### Implementation details
- Chunking: RecursiveCharacterTextSplitter with `chunk_size=1000`, `chunk_overlap=200`
- PDF parsing: pypdf (in-memory); scanned PDFs may yield no text (OCR not included)
- Embeddings: OpenAI `text-embedding-3-small` via official SDK
- Vector store: FAISS per file; aggregated retrieval across all stored indexes

### Troubleshooting
- Non-JSON errors in frontend: check API logs with `docker-compose logs -f api`
- Zero chunks: PDF likely has no extractable text (e.g., scanned). Consider adding OCR if needed
- Keys missing: ensure environment variables are set before `docker-compose up -d`


