import time

import requests

UPLOAD_FILE_ENDPOINT = "http://api:8000/documents"
QUESTION_ENDPOINT = "http://api:8000/question"
MODELS_ENDPOINT = "http://api:8000/models"


def upload_documents(files_bytes_and_names: list[tuple[str, bytes]]) -> dict:
    """Upload one or more documents to the API.

    Args:
        files_bytes_and_names: List of tuples (filename, bytes)

    Returns:
        Dict containing processing results
    """
    files = []
    for idx, (filename, content) in enumerate(files_bytes_and_names):
        files.append(("files", (filename, content, "application/pdf")))
    response = requests.post(UPLOAD_FILE_ENDPOINT, files=files)
    return response.json()


def ask_question(
    question: str,
    llm_provider: str = "openai",
    model: str | None = None,
    document_ids: list[str] = [],
) -> dict:
    """Send question to the API

    Args:
        question: User's question
        llm_provider: The LLM provider to use
        model: Optional model name for the selected provider
        document_ids: Optional list of document directory stems to restrict
            retrieval to (e.g., those uploaded in the current session)

    Returns:
        Dict containing answer and references
    """
    payload = {"question": question, "llm_provider": llm_provider}
    if model:
        payload["model"] = model
    # Always send document_ids (may be empty list if none uploaded)
    payload["document_ids"] = document_ids
    response = requests.post(QUESTION_ENDPOINT, json=payload)
    return response.json()


def get_available_models(max_retries: int = 8, backoff_seconds: float = 0.5) -> dict:
    """Get model list with simple retry to tolerate API startup delays."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{MODELS_ENDPOINT}", timeout=3)
            return response.json()
        except Exception as exc:
            last_err = exc
            time.sleep(backoff_seconds * (2**attempt))
    return {
        "error": f"API not reachable after {max_retries} attempts.",
        "details": str(last_err) if last_err else "unknown",
        "openai": ["gpt-4.1-mini"],
        "gemini": ["gemini-2.0-flash-lite"],
    }
