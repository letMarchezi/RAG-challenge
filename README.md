## RAG System (FastAPI + Streamlit)
**Developed by Leticia Bossatto Marchezi**

Document Q&A using Retrieval-Augmented Generation. Upload one or more PDFs, we chunk and embed them, store FAISS per-file indexes, and answer questions using your selected LLM provider.


Default embedder is `text-embedding-3-small` from OpenAI. It can be changed to others models from OpenAI through env variables, or to another model provider applying minimal changes on OpenAIEmbeddingsDirect class to the adequate endpoint usage in addition of api keys in .env.
Available models are set on `routes/main.py` in `Model_Options`.  

### Project layout
- `api/`
  - `main.py`: FastAPI app wiring
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
<img width="1700" height="2200" alt="ex_2-1" src="https://github.com/user-attachments/assets/9ee28d7a-3b6e-4f38-9687-2b8c74b16a96" />
<img width="1700" height="2200" alt="ex_2-2" src="https://github.com/user-attachments/assets/94aa1d14-6125-4148-8f6f-8c4ec2c803eb" />


Sample response for /documents (truncated):
```json
{
  "message": "Documents processed",
  "processed": 2,
  "skipped": 0,
  "total": 2,
  "total_chunks": 412,
  "results": [
    {
      "filename": "a.pdf",
      "message": "Document processed successfully",
      "skipped": false,
      "document_id": "a",
      "documents_indexed": 10,
      "total_chunks": 230
    },
    {
      "filename": "b.pdf",
      "message": "Document processed successfully",
      "skipped": false,
      "document_id": "b",
      "documents_indexed": 8,
      "total_chunks": 182
    }
  ]
}
```
- Ask a question via curl:
```
curl -H "Content-Type: application/json" -d '{
  "question":"What maintenance is recommended?",
  "llm_provider":"openai",
  "document_ids":["lb5001","weg-cestari-manual-iom-guia-consulta-rapida-50111652-pt-en-es-web"]
}' http://localhost:8000/question | cat
```

Sample response (truncated):
```json
{
  "answer": "A manutenção preventiva periódica deve ser realizada por pessoas qualificadas...",
  "references": "A manutenção preventiva periódica visa principalmente verificar as condições...",
  "citations": [
    {
      "document_id": "weg-cestari-manual-iom-guia-consulta-rapida-50111652-pt-en-es-web",
      "page": 21,
      "score": 1.1511561870574951,
      "snippet": "A manutenção preventiva periódica visa principalmente verificar as condições..."
    }
  ]
}
```

### Using the system
1) Open the frontend at `http://localhost:8501`
2) Upload one or more PDFs and click “Process Documents”
   - Each file is indexed under `vector_store/<filename-stem>/index.faiss|index.pkl`
   - Re-uploading the same filename is skipped (no reprocessing)
3) Choose provider (OpenAI/Gemini) and model (from `/models`)
4) Ask a question; the system retrieves similar chunks across all indexed files and generates an answer with references and citations
   - The frontend shows the API response time next to the answer and a Citations panel with file (document_id), page, score and snippet preview

### API endpoints
- `GET /health` → `{ "status": "ok" }`
- `GET /models` → `{ "openai": [...], "gemini": [...] }`
- `POST /documents` (multipart)
  - Field name: `files` (repeatable)
  - Returns summary: processed, skipped, total_chunks, per-file results
- `POST /question` (JSON)
  - `{ "question": "...", "llm_provider": "openai|gemini", "model": "optional", "document_ids": ["<file-stem>", ...] }`
  - `document_ids` is required. The frontend always sends the IDs of files uploaded in the current session (may be an empty array if none).
  - Returns a structured JSON object:
    - `answer` (string)
    - `references` (string with supporting excerpt text)
    - `citations` (array of objects): `{ document_id: str, page: int|null, score: number, snippet: str }`

### Implementation details
- Chunking: RecursiveCharacterTextSplitter with chunk_size=1400 and chunk_overlap=300 (length counted via Python's len)
- PDF parsing: pypdf (in-memory); scanned PDFs may yield no text (OCR not included)
- Embeddings: OpenAI `text-embedding-3-small` via official SDK, batched requests for efficiency
- Vector store: FAISS per file; aggregated retrieval across all stored indexes

### Design decisions and good practices
- Separation of concerns: Endpoints live under `api/routes`, while the main logic is in `api/services` (embeddings, LLM). This keeps routes thin and services testable and reusable.
- Provider/model abstraction: The LLM service cleanly switches between providers (OpenAI, Gemini) and models with minimal changes. Provider/model changes are detected per-request and the service is refreshed only when needed.
- Per-file indexes and re-upload skipping: Each uploaded PDF is stored in its own FAISS index directory and re-uploads of the same filename are skipped, avoiding redundant compute and cost of resources.
- Session-scoped retrieval: The frontend records the document IDs uploaded in the current session and passes them to the API so retrieval can be constrained to those documents, improving relevance and performance.
- Top-k retrieval: Retrieval collects candidates across per-file indexes, sorts by similarity score, and returns the top-k results (default k=5) to balance relevance, token usage, and latency.
- Input hygiene and batching: Text is sanitized before embedding; empty chunks are filtered out; embedding requests are sent in batches to reduce API overhead.
- Structured outputs: The LLM is instructed to return strict JSON with `answer`, `references`, and `citations`. The backend safely parses the JSON and falls back gracefully if needed.
- Propagated retrieval metadata: Each retrieved chunk carries `document_id`, `page`, and `score`. Minimal metadata is embedded into the prompt so the model can ground citations; the same metadata is returned in `citations`.
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


