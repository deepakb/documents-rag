from datetime import datetime
from typing import Dict
from pydantic import BaseModel

from document import Document, DocumentRepository
from embedded_document import EmbeddedDocument, EmbeddedDocumentRepository
from exceptions.exceptions import EntityDoesNotExistError


class Chat(BaseModel):
    """
    Represents a chat model.

    Attributes:
        document_id (str): The ID of the document associated with the chat.
        user_prompt (str): The prompt provided by the user for the chat.
        filters (Dict[str, str]): Filters associated with the chat.
    """
    document_id: str
    user_prompt: str
    filters: Dict[str, str]


class ChatService:
    """
    Represents a service for managing chats.
    """

    @staticmethod
    def create_context(chat: Chat):
        """
        Creates a context for the chat.

        Args:
            chat (Chat): The chat for which context is to be created.

        Returns:
            dict: The context created for the chat.
        """
        # Your implementation here
        pass

    @staticmethod
    def get_chat_response(chat: Chat):
        """
        Retrieves a response for the chat.

        Args:
            chat (Chat): The chat for which a response is to be retrieved.

        Returns:
            str: The response for the chat.
        """
        # Your implementation here
        pass
