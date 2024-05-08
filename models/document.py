from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Literal, Dict
from pydantic_mongo import AbstractRepository, ObjectIdField
from bson import ObjectId

from exceptions.exceptions import EntityDoesNotExistError, TypeError as TError


class Document(BaseModel):
    """
    Represents a document.

    Attributes:
        id (ObjectIdField): The unique identifier of the document.
        name (str): The name of the document.
        type (str): The type of the document.
        url (str, optional): The URL of the document (required if type is 'github').
        status (Literal["pending", "completed"]): The status of the document, which can be either "pending" or "completed".
        created_at (datetime): The timestamp indicating when the document was created. Defaults to the current datetime when not provided.
        updated_at (datetime): The timestamp indicating when the document was last updated. Defaults to the current datetime when not provided.
    """
    id: ObjectIdField = Field(
        default_factory=ObjectIdField, primary_key=True, alias="_id")
    name: str
    type: Literal['txt', 'docx', 'doc', 'pdf', 'ppt', 'github']
    url: str = None
    status: Literal["pending", "completed"]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    @field_validator('url')
    @classmethod
    def valid_url(cls, v: str, info: ValidationInfo) -> str:
        if info.data['type'] == 'github':
            if v is None:
                raise ValueError("URL is required when type is 'github'")
        return v


class DocumentRepository(AbstractRepository[Document]):
    """
    Repository class for interacting with the 'documents' collection.

    Attributes:
        Meta (class): Inner class containing metadata for the repository.
            collection_name (str): The name of the MongoDB collection.
    """
    class Meta:
        collection_name = 'documents'

    async def get_document(self, document_id: str) -> Document:
        collection = self.get_collection()
        document = collection.find_one({"_id": document_id})
        if not document:
            raise EntityDoesNotExistError(message="Document not found")
        return document

    async def delete_document(self, document_id: str) -> int:
        collection = self.get_collection()
        response = collection.delete_one({"_id": ObjectId(document_id)})
        if response.deleted_count == 0:
            raise EntityDoesNotExistError(message="Document not found")
        return response.deleted_count

    def update_document(self, filters: Dict[str, str], update_data: Dict[str, str]) -> int:
        """
        Update documents in the MongoDB collection based on partial matching filters and partial update data.

        Args:
            filters (Dict[str, str]): The filters to partially match against the Document model.
            update_data (Dict[str, str]): The data to partially update in the documents.

        Returns:
            int: The number of documents updated.
        """
        # Construct the update query
        update_query = {"$set": update_data}

        # Execute the update operation
        collection = self.get_collection()
        response = collection.update_one(filters, update_query)
        if response.modified_count == 0:
            raise EntityDoesNotExistError(message="Document not found")
        return response.modified_count
