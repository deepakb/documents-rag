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

# Initialize FastAPI app
app = FastAPI()

# Intialize OpenAIClient
openai = OpenAIClient(get_env_variable('OPENAI_API_KEY'))

# Initialize MongoDB Atlas client
client = MongoDBAtlasClient(get_env_variable(
    'MONGODB_URI'), get_env_variable('DB_NAME'))

# Intialize document processor
doc_processor = ProcessDocuments(openai, client)


@app.post("/add-documents/")
async def add_documents(files: List[UploadFile] = File(...)):
    response = await doc_processor.process(files, get_env_variable('EMBEDDED_COLLECTION'))
    return response


@app.delete("/delete-documents/")
async def delete_documents(document_ids: List[str]):
    try:
        object_ids = [str(doc_id) for doc_id in document_ids]
        deleted_count = client.delete(
            get_env_variable('EMBEDDED_COLLECTION'), object_ids, 'file_id')

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
        context = await retriver.invoke(chatRequest, [get_env_variable('EMBEDDED_COLLECTION')])
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
