from typing import List, Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from services.embeddings import EmbeddingsService
from services.llm import LLMService

api = APIRouter()

# Initialize services
embeddings_service = EmbeddingsService()
llm_service = None  # Will be initialized based on user choice


class QuestionRequest(BaseModel):
    question: str
    llm_provider: str = "openai"
    model: Optional[str] = None
    document_ids: List[str]


class Model_Options:
    OPENAI = ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano", "gpt-4o"]
    GEMINI = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]


@api.get("/models")
def get_models() -> dict:
    return {"openai": Model_Options.OPENAI, "gemini": Model_Options.GEMINI}


@api.post("/documents")
def generate_embeddings(files: List[UploadFile] = File(...)) -> dict:
    """Upload one or more PDF documents and generate embeddings for each.

    Each file is processed into its own FAISS index folder under `vector_store/<filename-stem>`.
    If an index for a given filename already exists, it is skipped.
    """
    results: List[dict] = []
    processed_count = 0
    skipped_count = 0
    total_chunks = 0

    for file in files:
        content = file.file.read()
        res = embeddings_service.process_pdf(content, file.filename)
        results.append(
            {
                "filename": file.filename,
                **res,
            }
        )
        if res.get("skipped"):
            skipped_count += 1
        else:
            processed_count += 1
            total_chunks += res.get("total_chunks", 0)

    return {
        "message": "Documents processed",
        "processed": processed_count,
        "skipped": skipped_count,
        "total": len(results),
        "total_chunks": total_chunks,
        "results": results,
    }


@api.post("/question")
def prompt_llm_rag(request: QuestionRequest) -> dict:
    """Generates the answer for the question using RAG"""
    global llm_service

    # Initialize or update LLM service if provider or model changed
    if (
        llm_service is None
        or getattr(llm_service, "provider", None) != request.llm_provider
        or getattr(llm_service, "model", None) != request.model
    ):
        llm_service = LLMService(request.llm_provider, request.model)

    # Get relevant documents using embeddings, constrained to uploaded docs
    relevant_docs = embeddings_service.similarity_search(
        request.question,
        document_ids=request.document_ids,
    )

    # Generate answer using LLM
    result = llm_service.generate_answer(request.question, relevant_docs)

    return result
