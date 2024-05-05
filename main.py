import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException

from database import MongoDBAtlasClient
from openai_client import OpenAIClient
from retriver import VectorRetriever
from model import ChatRequest
from api_response import Response
from documents import ProcessDocuments
from settings import api, mongo
from models.embedded_documents import EmbeddedDocumentsRepository

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAIClient
openai = OpenAIClient(api.openai_key)

# Initialize MongoDB Atlas client
client = MongoDBAtlasClient(mongo.uri, mongo.database)

# Initialize document processor
doc_processor = ProcessDocuments(openai, client)

# Define tags
tags_metadata = [
    {"name": "documents", "description": "Operations related to documents"},
    {"name": "chat", "description": "Operations related to chat"},
]


@app.post("/add-documents/", tags=["documents"], summary="Upload and process documents")
async def add_documents(files: List[UploadFile] = File(...)):
    response = await doc_processor.process(files, mongo.embedded_collection)
    return response


@app.delete("/delete-documents/", tags=["documents"], summary="Delete documents by IDs")
async def delete_documents(document_ids: List[str], summary="Delete documents by IDs"):
    try:
        object_ids = [str(doc_id) for doc_id in document_ids]
        repo = EmbeddedDocumentsRepository(database=client.db)
        deleted_count = repo.delete_by_field(object_ids, 'documents_id')
        response = Response.success(
            message=f"{deleted_count} documents deleted successfully"
        )
        return response.to_dict()
    except Exception as e:
        response, status_code = Response.failure(str(e), status_code=500)
        raise HTTPException(
            status_code=status_code,
            detail=response.to_dict()
        )


@app.post("/chat/", tags=["chat"], summary="Chat with AI assistant")
async def chat(chatRequest: ChatRequest):
    try:
        retriver = VectorRetriever(openai, client)
        context = await retriver.invoke(chatRequest, [mongo.embedded_collection])
        context_data = json.loads(context)[0]
        api_response = await openai.fetch_chat_response(chatRequest.question, context_data['text'])
        response = Response.success(message=api_response)
        return response.to_dict()
    except Exception as e:
        response, status_code = Response.failure(str(e), status_code=500)
        raise HTTPException(
            status_code=status_code,
            detail=response.to_dict()
        )
