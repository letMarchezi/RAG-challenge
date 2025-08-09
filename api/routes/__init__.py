from fastapi import APIRouter
from routes.main import api as api_router

router = APIRouter()
router.include_router(api_router)
