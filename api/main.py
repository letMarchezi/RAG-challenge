import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
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

# add the CORS config for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow for all (do not use in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Main"])
async def health_check():
    return {"status": "ok"}

