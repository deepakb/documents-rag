# Dependency Resolver for Chat Handling
from fastapi import Depends
from vendor.openai import get_openai_client
from vendor.mongodb import get_mongodb_client
from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.chat_handler import ChatHandler
from services.vector_retriever import VectorRetriever
from vendor.retriever import get_vector_retriever


def get_chat_handler(
    openai: OpenAIClient = Depends(get_openai_client),
    mongo_client: MongoDBAtlasClient = Depends(get_mongodb_client),
    retriever: VectorRetriever = Depends(get_vector_retriever)
) -> ChatHandler:
    """
    Dependency resolver function to provide an instance of ChatHandler.

    Parameters:
    - openai (OpenAIClient): Instance of OpenAIClient for processing documents with OpenAI.
    - mongo_client (MongoDBAtlasClient): Instance of MongoDBAtlasClient for database operations.
    - retriever (VectorRetriever): Instance of VectorRetriever for retrieving vectors from MongoDB using OpenAI services.

    Returns:
    - ChatHandler: Instance of ChatHandler initialized with the provided dependencies.
    """
    return ChatHandler(openai, mongo_client, retriever)
