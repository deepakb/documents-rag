from fastapi import Depends

from vendor.openai import get_openai_client
from vendor.mongodb import get_mongodb_client
from services.vector_retriever import VectorRetriever
from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient


def get_vector_retriever(
    openai: OpenAIClient = Depends(get_openai_client),
    mongo_client: MongoDBAtlasClient = Depends(get_mongodb_client)
) -> VectorRetriever:
    """
    Dependency resolver function to provide an instance of VectorRetriever.

    Parameters:
    - openai (OpenAIClient): Instance of OpenAIClient for fetching alternate questions and creating embeddings.
    - mongo_client (MongoDBAtlasClient): Instance of MongoDBAtlasClient for database operations.

    Returns:
    - VectorRetriever: Instance of VectorRetriever initialized with the provided dependencies.
    """
    return VectorRetriever(openai, mongo_client)
