from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    file_id: str
