from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal
from pydantic_mongo import AbstractRepository, ObjectIdField


class Documents(BaseModel):
    """
    Represents a document.

    Attributes:
        id (ObjectIdField): The unique identifier of the document.
        name (str): The name of the document.
        type (str): The type of the document.
        status (Literal["pending", "completed"]): The status of the document, which can be either "pending" or "completed".
        created_at (datetime): The timestamp indicating when the document was created. Defaults to the current datetime when not provided.
    """
    id: ObjectIdField = Field(
        default_factory=ObjectIdField, primary_key=True, alias="_id")
    name: str
    type: str
    status: Literal["pending", "completed"]
    created_at: datetime = datetime.now()


class DocumentsRepository(AbstractRepository[Documents]):
    """
    Repository class for interacting with the 'documents' collection.

    Attributes:
        Meta (class): Inner class containing metadata for the repository.
            collection_name (str): The name of the MongoDB collection.
    """
    class Meta:
        collection_name = 'documents'
