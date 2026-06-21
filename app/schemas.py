"""Pydantic request/response schemas for the REST API.

Every endpoint returns a JSON object with a ``success`` boolean at the
top level.  On failure, ``success`` is ``false`` and the ``error`` /
``message`` fields describe what went wrong.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentUploadResponse(BaseModel):
    """Response for ``POST /api/documents/upload``."""
    success: bool
    document_id: str
    filename: str
    chunks_count: int
    message: str


class SearchQuery(BaseModel):
    """Request body for ``POST /api/search``."""
    query: str
    top_k: Optional[int] = Field(default=None, description="Number of results to return")


class SearchResult(BaseModel):
    """A single retrieved chunk with its similarity score."""
    chunk_id: str
    document_id: str
    filename: str
    content: str
    similarity_score: float


class SearchResponse(BaseModel):
    """Response for ``POST /api/search``."""
    success: bool
    query: str
    results: List[SearchResult]
    total: int


class ChatQuery(BaseModel):
    """Request body for ``POST /api/chat``."""
    query: str
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to use")
    conversation_history: Optional[List[dict]] = Field(
        default=None,
        description="OpenAI-style message list for multi-turn dialogue",
    )


class ChatResponse(BaseModel):
    """Response for ``POST /api/chat``."""
    success: bool
    query: str
    answer: str
    sources: List[SearchResult]
    model: str


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx."""
    success: bool
    error: str
    message: str
