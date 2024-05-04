from pydantic import BaseModel

class ChatRequest(BaseModel):
    file_id: str
    question: str
    filters: dict
