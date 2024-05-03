import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException

from database import MongoDBAtlasClient
from utils import get_env_variable
from openai_client import OpenAIClient
from retriver import VectorRetriever
from models import ChatRequest
from api_response import Response
from documents import ProcessDocuments

DB_NAME = 'documents_rag'
COLLECTION_NAME = 'documents'

# Initialize FastAPI app
app = FastAPI()

# Intialize OpenAIClient
openai = OpenAIClient(get_env_variable('OPENAI_API_KEY'))

# Initialize MongoDB Atlas client
client = MongoDBAtlasClient(get_env_variable('MONGODB_URI'), DB_NAME)


@app.post("/add-documents/")
async def add_documents(files: List[UploadFile] = File(...)):
    process_document = ProcessDocuments(openai, client)
    response = await process_document.process(files, COLLECTION_NAME)
    return response


@app.delete("/delete-documents/")
async def delete_documents(document_ids: List[str]):
    try:
        # Convert string document IDs to ObjectId
        object_ids = [str(doc_id) for doc_id in document_ids]

        # Delete documents from MongoDB Atlas
        deleted_count = client.delete(
            COLLECTION_NAME, object_ids, 'file_id')

        response = Response.success(
            message=f"{deleted_count} documents deleted successfully")
        return response.to_dict()
    except Exception as e:
        response, status_code = Response.failure(str(e), status_code=500)
        raise HTTPException(
            status_code=status_code,
            detail=response.to_dict()
        )


@app.post("/chat/")
async def chat(chatRequest: ChatRequest):
    try:
        retriver = VectorRetriever(openai, client)
        context = await retriver.invoke(chatRequest, [COLLECTION_NAME])
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
