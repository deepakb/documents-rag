from fastapi import APIRouter, Depends
from typing import List
from fastapi import File, UploadFile

from services.document_handler import DocumentHandler
from vendor.document import get_document_handler

router = APIRouter()


@router.post("/documents/", tags=["documents"], summary="Upload and process documents")
async def add_documents(
    files: List[UploadFile] = File(...),
    doc_handler: DocumentHandler = Depends(get_document_handler)
):
    response = await doc_handler.process(files)
    return response


@router.delete("/documents/{document_id}", tags=["documents"], summary="Delete document by id")
async def delete_documents(
    document_id: str,
    doc_handler: DocumentHandler = Depends(get_document_handler)
):
    response = await doc_handler.delete(document_id)
    return response
