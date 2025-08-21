from pydantic import BaseModel, Field
from typing import List


class MainAgentResponse(BaseModel):
    """Structured response format for the conversation agent."""
    message: str = Field(description="The AI's response message")
