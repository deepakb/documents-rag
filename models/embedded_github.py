from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_mongo import AbstractRepository, ObjectIdField
from typing import Optional, List, Dict, Mapping
from pymongo.database import Database

from config.settings import mongo
from exceptions.exceptions import EntityDoesNotExistError


class EmbeddedGithubDocument(BaseModel):
    """
    Represents an embedded GitHub document.

    Attributes:
        id (ObjectIdField): The unique identifier of the document.
        chunk_id (str): The ID of the chunk.
        document_id (str): The ID of the document.
        raw_chunk (str): The raw chunk data.
        vector_chunk (List[float]): The vector chunk data.
        token_count (int): The count of tokens.
        created_at (datetime): The timestamp indicating when the document was created. Defaults to the current datetime when not provided.
        expires_at (Optional[datetime]): The timestamp indicating when the document expires (if applicable).
        metadata (Dict[str, str]): Additional metadata for the document, such as its source.
    """
    id: ObjectIdField = Field(
        default_factory=ObjectIdField, primary_key=True, alias="_id")
    chunk_id: str
    document_id: str
    raw_chunk: str
    vector_chunk: List[float]
    token_count: int
    created_at: datetime = datetime.now()
    expires_at: Optional[datetime]
    metadata: Dict[str, str]


class EmbeddedGithubDocumentRepository(AbstractRepository[EmbeddedGithubDocument]):
    """
    Repository class for interacting with the 'embedded_documents' collection.
    """

    class Meta:
        collection_name = mongo.embedded_github

    async def delete_embedded_documents(self, filter: Mapping[str, str]) -> int:
        collection = self.get_collection()
        response = collection.delete_many(filter)
        if response.deleted_count == 0:
            raise EntityDoesNotExistError(message="Document not found")
        return response.deleted_count
