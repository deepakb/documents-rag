from fastapi import APIRouter, Depends

from core.model import ChatRequest
from services.chat_handler import ChatHandler
from vendor.chat import get_chat_handler

router = APIRouter()


@router.post("/chat/", tags=["chat"], summary="Chat with ai and get response")
async def chat(
    chatRequest: ChatRequest,
    chat_handler: ChatHandler = Depends(get_chat_handler)
):
    response = await chat_handler.chat(chatRequest)
    return response
