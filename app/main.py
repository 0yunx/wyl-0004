"""FastAPI application — RAG Knowledge Base Q&A API.

Provides three core endpoints:

- ``POST /api/documents/upload`` — ingest a PDF/TXT document
- ``POST /api/search``         — vector similarity search
- ``POST /api/chat``           — retrieval-augmented generation

Plus CRUD helpers for document management and a health-check endpoint.

All responses follow a uniform JSON envelope with a ``success`` boolean.
On error, ``success`` is ``false`` and ``error`` / ``message`` fields
describe the failure.
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import (
    DocumentUploadResponse,
    SearchQuery,
    SearchResponse,
    SearchResult,
    ChatQuery,
    ChatResponse,
    ErrorResponse,
)
from .document_parser import DocumentParser
from .chunker import TextChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .llm_client import LLMClient

app = FastAPI(
    title="RAG Knowledge Base API",
    description="基于 RAG 的个人知识库问答 API（手写分块 + HNSW 向量检索）",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_parser = DocumentParser()
chunker = TextChunker()
embedder = Embedder()
vector_store = VectorStore()
llm_client = LLMClient()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — confirm the API is running."""
    return {"message": "RAG Knowledge Base API is running", "status": "ok"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check — returns status and number of indexed documents."""
    return {"status": "healthy", "documents_count": len(vector_store.documents)}


@app.post(
    "/api/documents/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
    },
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF or TXT document.

    Pipeline: validate → parse → chunk → embed → store → index (HNSW).

    The file is validated for extension and size *before* any parsing
    work begins so that non-text files (e.g. images) are rejected early.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file is not allowed",
        )

    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size: {settings.MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    try:
        text, file_type = document_parser.parse(file.filename, content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse document: {str(e)}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extractable text content found in the document",
        )

    chunks_text = chunker.chunk(text)

    if not chunks_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create chunks from document content",
        )

    try:
        chunk_vectors = await embedder.embed_batch(chunks_text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}",
        )

    doc_id = vector_store.add_document(
        filename=file.filename,
        file_type=file_type,
        chunks_text=chunks_text,
        chunk_vectors=chunk_vectors,
    )

    return DocumentUploadResponse(
        success=True,
        document_id=doc_id,
        filename=file.filename,
        chunks_count=len(chunks_text),
        message=f"Document uploaded successfully. Created {len(chunks_text)} chunks.",
    )


@app.get("/api/documents", tags=["Documents"])
async def list_documents():
    """List all indexed documents with metadata."""
    return {
        "success": True,
        "documents": vector_store.list_documents(),
        "total": len(vector_store.documents),
    }


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the index.

    Vectors are soft-deleted; physical compaction occurs automatically
    when garbage exceeds 20 % of total rows.
    """
    if not vector_store.get_document(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id {doc_id} not found",
        )

    vector_store.delete_document(doc_id)
    return {"success": True, "message": f"Document {doc_id} deleted successfully"}


@app.post(
    "/api/search",
    response_model=SearchResponse,
    tags=["Search"],
)
async def search(query: SearchQuery):
    """Semantic vector search over indexed document chunks.

    Uses the HNSW index for O(log N) retrieval.  Returns the top-k
    most similar chunks with their cosine-similarity scores.
    """
    if not query.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    try:
        query_vector = await embedder.embed(query.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {str(e)}",
        )

    top_k = query.top_k or settings.TOP_K
    results = vector_store.search(query_vector, top_k)

    search_results = [
        SearchResult(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            filename=chunk["filename"],
            content=chunk["content"],
            similarity_score=score,
        )
        for chunk, score in results
    ]

    return SearchResponse(
        success=True,
        query=query.query,
        results=search_results,
        total=len(search_results),
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
async def chat(query: ChatQuery):
    """Retrieval-augmented generation: search → LLM answer.

    1. Embed the query and retrieve the top-k chunks via HNSW.
    2. Pass the chunks as context to the LLM.
    3. Return the LLM's answer along with source citations.
    """
    if not query.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    try:
        query_vector = await embedder.embed(query.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {str(e)}",
        )

    top_k = query.top_k or settings.TOP_K
    results = vector_store.search(query_vector, top_k)

    context_chunks = [
        SearchResult(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            filename=chunk["filename"],
            content=chunk["content"],
            similarity_score=score,
        )
        for chunk, score in results
    ]

    try:
        answer = await llm_client.generate_answer(
            query=query.query,
            context_chunks=context_chunks,
            conversation_history=query.conversation_history,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}",
        )

    return ChatResponse(
        success=True,
        query=query.query,
        answer=answer,
        sources=context_chunks,
        model=settings.CHAT_MODEL,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Convert FastAPI HTTPExceptions into the uniform error envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.status_code,
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all handler — prevents stack traces from leaking in responses."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": 500,
            "message": f"Internal server error: {str(exc)}",
        },
    )
