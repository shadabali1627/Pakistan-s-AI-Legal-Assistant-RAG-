# backend/schemas.py
from pydantic import BaseModel
from typing import Literal

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str