from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    chunks_count: int
    message: str


class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=None, description="Number of results to return")


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    content: str
    similarity_score: float


class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[SearchResult]
    total: int


class ChatQuery(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to use")
    conversation_history: Optional[List[dict]] = Field(default=None, description="Conversation history")


class ChatResponse(BaseModel):
    success: bool
    query: str
    answer: str
    sources: List[SearchResult]
    model: str


class ErrorResponse(BaseModel):
    success: bool
    error: str
    message: str
