# Dependency Resolver for Document Processing
from fastapi import Depends
from vendor.openai import get_openai_client
from vendor.mongodb import get_mongodb_client
from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.github_handler import GithubHandler


def get_github_handler(
    # Dependency for OpenAI client
    openai: OpenAIClient = Depends(get_openai_client),
    mongo_client: MongoDBAtlasClient = Depends(
        get_mongodb_client)  # Dependency for MongoDB client
) -> GithubHandler:
    """
    Dependency resolver function to provide an instance of GithubHandler.

    Parameters:
    - openai (OpenAIClient): Instance of OpenAIClient for processing documents with OpenAI.
    - mongo_client (MongoDBAtlasClient): Instance of MongoDBAtlasClient for database operations.

    Returns:
    - GithubHandler: Instance of GithubHandler initialized with the provided dependencies.
    """
    return GithubHandler(openai, mongo_client)
