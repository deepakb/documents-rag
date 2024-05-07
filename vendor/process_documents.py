# Dependency Resolver for Document Processing
from fastapi import Depends
from vendor.openai import get_openai_client
from vendor.mongo_atlas import get_mongodb_client
from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.document_handler import DocumentHandler


def get_document_handler(
    # Dependency for OpenAI client
    openai: OpenAIClient = Depends(get_openai_client),
    mongo_client: MongoDBAtlasClient = Depends(
        get_mongodb_client)  # Dependency for MongoDB client
) -> DocumentHandler:
    """
    Dependency resolver function to provide an instance of DocumentHandler.

    Parameters:
    - openai (OpenAIClient): Instance of OpenAIClient for processing documents with OpenAI.
    - mongo_client (MongoDBAtlasClient): Instance of MongoDBAtlasClient for database operations.

    Returns:
    - DocumentHandler: Instance of DocumentHandler initialized with the provided dependencies.
    """
    return DocumentHandler(openai, mongo_client)
