import requests
import os

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

def ask_question(question: str, llm_provider: str = "openai", model: str | None = None) -> dict:
    """Send question to the API
    
    Args:
        question: User's question
        llm_provider: The LLM provider to use
        
    Returns:
        Dict containing answer and references
    """
    payload = {"question": question, "llm_provider": llm_provider}
    if model:
        payload["model"] = model
    response = requests.post(QUESTION_ENDPOINT, json=payload)
    return response.json()


def get_available_models():
    response = requests.get(f"{MODELS_ENDPOINT}")
    return response.json()