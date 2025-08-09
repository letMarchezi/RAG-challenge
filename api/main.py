import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routes.main import api as api_router

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


app = FastAPI(
    title="RAG System",
    version="0.0.1",
    description="Developed by Leticia",
)


app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Main"])
async def health_check():
    return {"status": "ok"}
