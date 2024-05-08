import json
from fastapi import HTTPException

from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.api_response import Response
from services.vector_retriever import VectorRetriever
from core.model import ChatRequest
from config.settings import mongo


class ChatHandler:
    """
    Handles chat operations by processing user queries and retrieving responses.
    """

    def __init__(self, openai: OpenAIClient, mongo_client: MongoDBAtlasClient, retriever: VectorRetriever):
        """
        Initializes the ChatHandler.

        Args:
            openai (OpenAIClient): OpenAI client for generating chat responses.
            mongo_client (MongoDBAtlasClient): MongoDB client for database operations.
            retriever (VectorRetriever): VectorRetriever for retrieving context vectors.
        """
        self.openai = openai
        self.mongo_client = mongo_client
        self.retriever = retriever

    async def chat(self, chatRequest: ChatRequest):
        """
        Processes user chat requests and generates chat responses.

        Args:
            chatRequest (ChatRequest): Chat request object containing user query.

        Returns:
            dict: Response dictionary containing the chat response.
        """
        try:
            # Retrieve context vectors based on the chat request
            context = await self.retriever.invoke(chatRequest, [mongo.embedded_collection])
            context_data = json.loads(context)[0]

            # Fetch chat response using OpenAI
            api_response = await self.openai.fetch_chat_response(chatRequest.question, context_data['text'])

            # Return success response
            return Response.success(message=api_response)
        except Exception as e:
            # Handle exceptions and return failure response
            response, status_code = Response.failure(str(e), status_code=500)
            raise HTTPException(
                status_code=status_code,
                detail=response.to_dict()
            )
