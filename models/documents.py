from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal
from pydantic_mongo import AbstractRepository, ObjectIdField


class Document(BaseModel):
    """
    Represents a document.

    Attributes:
        id (ObjectIdField): The unique identifier of the document.
        name (str): The name of the document.
        type (str): The type of the document.
        status (Literal["pending", "completed"]): The status of the document, which can be either "pending" or "completed".
        created_at (datetime): The timestamp indicating when the document was created. Defaults to the current datetime when not provided.
        updated_at (datetime): The timestamp indicating when the document was last updated. Defaults to the current datetime when not provided.
    """
    id: ObjectIdField = Field(
        default_factory=ObjectIdField, primary_key=True, alias="_id")
    name: str
    type: str
    status: Literal["pending", "completed"]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class DocumentRepository(AbstractRepository[Document]):
    """
    Repository class for interacting with the 'documents' collection.

    Attributes:
        Meta (class): Inner class containing metadata for the repository.
            collection_name (str): The name of the MongoDB collection.
    """
    class Meta:
        collection_name = 'documents'

    def update_by_field(self, field: str, value: str, update_data: dict) -> int:
        """
        Update documents in the MongoDB collection based on a specific field and value.

        Args:
            field (str): The field to match against.
            value (Any): The value to match against the field.
            update_data (dict): The data to update in the documents.

        Returns:
            int: The number of documents updated.
        """
        collection = self.get_collection()
        result = collection.update_many({field: value}, {"$set": update_data})
        return result.modified_count
