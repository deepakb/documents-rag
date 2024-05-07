from services.openai_client import OpenAIClient
from config.settings import api


def get_openai_client():
    """
    Dependency provider function to initialize OpenAIClient.

    Returns:
        OpenAIClient: An instance of OpenAIClient.
    """
    return OpenAIClient(api.openai_key)
