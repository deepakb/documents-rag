from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_mongo import AbstractRepository, ObjectIdField
from typing import Optional, List
from pymongo.database import Database

from settings import mongo


class EmbeddedDocuments(BaseModel):
    """
    Represents an embedded document.

    Attributes:
        id (ObjectIdField): The unique identifier of the document.
        chunk_id (str): The ID of the chunk.
        documents_id (str): The ID of the documents.
        raw_chunk (str): The raw chunk data.
        vector_chunk (List[float]): The vector chunk data.
        token_count (int): The count of tokens.
        created_at (datetime): The timestamp indicating when the document was created. Defaults to the current datetime when not provided.
        expires_at (Optional[datetime]): The timestamp indicating when the document expires (if applicable).
    """
    id: ObjectIdField = Field(
        default_factory=ObjectIdField, primary_key=True, alias="_id")
    chunk_id: str
    documents_id: str
    raw_chunk: str
    vector_chunk: List[float]
    token_count: int
    created_at: datetime = datetime.now()
    expires_at: Optional[datetime]


class EmbeddedDocumentsRepository(AbstractRepository[EmbeddedDocuments]):
    """
    Repository class for interacting with the 'embedded_documents' collection.
    """

    class Meta:
        collection_name = mongo.documents_collection

    def delete_by_field(self, document_ids: List[str], field: str) -> int:
        """
        Delete documents from the MongoDB collection based on provided document IDs.

        Args:
            document_ids (List[str]): The list of document IDs to delete.
            field (str): The field to match document IDs against.

        Returns:
            int: The number of documents deleted.
        """
        collection = self.get_collection()
        result = collection.delete_many(
            {field: {"$in": document_ids}})
        return result.deleted_count
