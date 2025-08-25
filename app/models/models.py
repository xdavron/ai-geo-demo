from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)


class QueryResponse(BaseModel):
    answer: str
    session_id: str


class StreamQueryResponse(BaseModel):
    """Response model for streaming chat endpoint.

    Attributes:
        content: The content of the current chunk.
        done: Whether the stream is complete.
    """

    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int