from pydantic import BaseModel


class ChatRequest(BaseModel):
    document_id: str
    question: str
    filters: dict
