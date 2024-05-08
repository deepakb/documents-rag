from fastapi import APIRouter

from . import chat, documents, github

base_router = APIRouter()

base_router.include_router(
    documents.router, tags=["documents"], prefix="/v1"
)
base_router.include_router(github.router, tags=["github"], prefix="/v1")
base_router.include_router(chat.router, tags=["chat"], prefix="/v1")
