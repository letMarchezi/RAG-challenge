## RAG System (FastAPI + Streamlit)
**Developed by Leticia Bossatto Marchezi**

Document Q&A using Retrieval-Augmented Generation. Upload one or more PDFs, we chunk and embed them, store FAISS per-file indexes, and answer questions using your selected LLM provider.


Default embedder is `text-embedding-3-small` from OpenAI. It can be changed to others models from OpenAI through env variables, or to another model provider applying minimal changes on OpenAIEmbeddingsDirect class to the adequate endpoint usage in addition of api keys in .env.
Available models are set on `routes/main.py` in `Model_Options`.  

### Project layout
- `api/`
  - `main.py`: FastAPI app wiring and CORS
  - `routes/main.py`: Endpoints (`/documents`, `/question`, `/models`, `/health`)
  - `services/embeddings.py`: PDF parsing (pypdf), chunking, embeddings (OpenAI), FAISS storage (per-file)
  - `services/llm.py`: LLM abstraction (OpenAI, Gemini)
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
Compose passes provider keys via `environment:`. Set them in a `.env` file in the project root so compose can substitute:

```
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_google_key
OPENAI_EMBEDDING_MODEL=selected_model
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

### Using the system
1) Open the frontend at `http://localhost:8501`
2) Upload one or more PDFs and click “Process Documents”
   - Each file is indexed under `vector_store/<filename-stem>/index.faiss|index.pkl`
   - Re-uploading the same filename is skipped (no reprocessing)
3) Choose provider (OpenAI/Gemini) and model (from `/models`)
4) Ask a question; the system retrieves similar chunks across all indexed files and generates an answer with references
   - The frontend shows the API response time next to the answer

### API endpoints
- `GET /health` → `{ "status": "ok" }`
- `GET /models` → `{ "openai": [...], "gemini": [...] }`
- `POST /documents` (multipart)
  - Field name: `files` (repeatable)
  - Returns summary: processed, skipped, total_chunks, per-file results
- `POST /question` (JSON)
  - `{ "question": "...", "llm_provider": "openai|gemini", "model": "optional", "document_ids": ["<file-stem>", ...] }`
  - When `document_ids` is provided, retrieval searches only those per-file FAISS indexes. The frontend automatically passes the IDs of files uploaded in the current session so questions are restricted to those uploads.
  - Returns `{ "answer": str, "references": [str] }`

### Implementation details
- Chunking: RecursiveCharacterTextSplitter with chunk_size=1400 and chunk_overlap=300 (length counted via Python's len)
- PDF parsing: pypdf (in-memory); scanned PDFs may yield no text (OCR not included)
- Embeddings: OpenAI text-embedding-3-small via official SDK, batched requests for efficiency
- Vector store: FAISS per file; aggregated retrieval across all stored indexes

### Design decisions and good practices
- Separation of concerns: Endpoints live under `api/routes`, while the main logic is in `api/services` (embeddings, LLM). This keeps routes thin and services testable and reusable.
- Provider/model abstraction: The LLM service cleanly switches between providers (OpenAI, Gemini) and models with minimal changes. Provider/model changes are detected per-request and the service is refreshed only when needed.
- Per-file indexes and re-upload skipping: Each uploaded PDF is stored in its own FAISS index directory and re-uploads of the same filename are skipped, avoiding redundant compute and cost of resources.
- Session-scoped retrieval: The frontend records the document IDs uploaded in the current session and passes them to the API so retrieval can be constrained to those documents, improving relevance and performance.
- Top-k retrieval: Retrieval collects candidates across per-file indexes, sorts by similarity score, and returns the top-k results (default k=5) to balance relevance, token usage, and latency.
- Input hygiene and batching: Text is sanitized before embedding; empty chunks are filtered out; embedding requests are sent in batches to reduce API overhead.
- Structured prompting: Answers are requested with a consistent structure (answer plus references), improving reliability of downstream parsing and display.
- Persistence: FAISS data is mounted to `./vector_store` and persists across container restarts for reproducibility and faster startup.

### Next steps
- Expose k (top-k retrieval size) as a request/UI option for advanced users.
- Add an endpoint to list available document IDs for manual selection beyond the current session.
- Include more LLM providers and models for answer's response
- Flexibilize the choice of embedder model similarly to answer generation 
- Add OCR (e.g., Tesseract) for scanned PDFs that have no extractable text.
- Consider structured LLM outputs (e.g., JSON) to further harden parsing.

### Troubleshooting
- Non-JSON errors in frontend: check API logs with `docker-compose logs -f api`
- Zero chunks: PDF likely has no extractable text (e.g., scanned). Consider adding OCR if needed
- Keys missing: ensure environment variables are set before `docker-compose up -d`


