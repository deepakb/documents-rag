import json
from typing import Sequence
from fastapi import APIRouter, Depends
from loguru import logger
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException

from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.vector_retriver import VectorRetriever
from vendor.openai import get_openai_client
from vendor.mongo_atlas import get_mongodb_client
from core.model import ChatRequest
from config.settings import api, mongo
from services.api_response import Response

router = APIRouter()


@router.post("/chat/", tags=["chat"], summary="Chat with AI assistant")
async def chat(
    chatRequest: ChatRequest,
    openai: OpenAIClient = Depends(get_openai_client),
    mongo_client: MongoDBAtlasClient = Depends(get_mongodb_client)
):
    try:
        retriver = VectorRetriever(openai, mongo_client)
        context = await retriver.invoke(chatRequest, [mongo.embedded_collection])
        context_data = json.loads(context)[0]
        api_response = await openai.fetch_chat_response(chatRequest.question, context_data['text'])
        response = Response.success(message=api_response)
        return response.to_dict()
    except Exception as e:
        response, status_code = Response.failure(str(e), status_code=500)
        raise HTTPException(
            status_code=status_code,
            detail=response.to_dict()
        )
