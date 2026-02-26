"""
FastAPI interface for the Simple RAG system.

Run with:
    pixi run uvicorn simple_rag.api:app --reload

Endpoints:
    POST   /documents          Upload a PDF and ingest it (chunk + embed)
    GET    /documents          List all loaded documents
    GET    /documents/{doc_id} Get details for a specific document
    DELETE /documents/{doc_id} Remove a document from memory
    POST   /query              Ask a question against a loaded document
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from simple_rag.rag import (
    delete_document,
    documents,
    ingest_pdf_bytes,
    list_documents,
    query_document,
)


class PrettyJSONResponse(JSONResponse):
    """JSONResponse subclass that pretty-prints with 2-space indentation."""

    def render(self, content) -> bytes:
        import json

        return json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8")


app = FastAPI(
    title="Simple RAG API",
    description="A minimal Retrieval-Augmented Generation API powered by Gemini.",
    version="0.1.0",
    default_response_class=PrettyJSONResponse,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    doc_id: str
    question: str


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    context_chunks: list[str]


class DocumentInfo(BaseModel):
    doc_id: str
    name: str
    num_chunks: int


class IngestResponse(BaseModel):
    doc_id: str
    name: str
    num_chunks: int
    message: str


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/documents", response_model=IngestResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF file. It will be chunked, embedded, and stored in memory."""

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    pdf_bytes = await file.read()
    filename = file.filename or "upload.pdf"

    try:
        doc_id = ingest_pdf_bytes(pdf_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {e}")

    doc = documents[doc_id]
    return IngestResponse(
        doc_id=doc_id,
        name=doc["name"],
        num_chunks=doc["num_chunks"],
        message=f"Successfully ingested '{filename}' into {doc['num_chunks']} chunks.",
    )


@app.get("/documents", response_model=list[DocumentInfo])
async def get_documents():
    """List all currently loaded documents."""
    return list_documents()


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str):
    """Get metadata for a specific document."""
    doc = documents.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentInfo(
        doc_id=doc_id,
        name=doc["name"],
        num_chunks=doc["num_chunks"],
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def remove_document(doc_id: str):
    """Remove a document from the in-memory store."""
    try:
        delete_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DeleteResponse(doc_id=doc_id, message="Document deleted.")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question against a previously uploaded document."""
    try:
        result = query_document(req.doc_id, req.question)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Document '{req.doc_id}' not found."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        doc_id=req.doc_id,
        question=req.question,
        answer=result["answer"],
        context_chunks=result["context_chunks"],
    )
