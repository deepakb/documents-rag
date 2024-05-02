import os
import uuid
import datetime
import shutil
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_core.documents import Document

from database import MongoDBAtlasClient
from utils import check_file_type, get_token_counts, get_env_variable, load_document, split_text_into_chunks
from openai_client import OpenAIClient
from retriver import VectorRetriever
from models import ChatRequest

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
    results = []
    for file in files:
        try:
            # Check if the file type is valid
            if not check_file_type(file.filename):
                raise ValueError(f"Invalid file type for {file.filename}")

            request_id = str(uuid.uuid4())
            file_id = str(uuid.uuid4())
            temp_folder_path = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            mounted_path = '/tmp'
            folder_path = os.path.join(mounted_path, temp_folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            full_file_path = os.path.join(folder_path, file.filename)

            # Save the uploaded file to a temporary location
            with open(full_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            file_extension = file.filename.rsplit('.', 1)[1].lower()

            # Parse text from the uploaded document and split into chunks
            documents = await load_document(full_file_path, file_extension)

            # Create embedding for the documents
            all_texts = [d.page_content for d in documents]
            all_tokens = sum([get_token_counts(d) for d in all_texts])
            # model limit is 8192
            chunk_size = all_tokens if all_tokens < 8100 else 8100
            chunks = await split_text_into_chunks(all_texts, chunk_size)

            created_at = datetime.datetime.now()
            expires_at = created_at + datetime.timedelta(days=1)

            vectors = []
            doc_id = 1
            for chunk in chunks:
                uniqie_id = request_id + '-' + str(doc_id)
                token_count = get_token_counts(chunk)
                vector_text = await openai.create_embedding(chunk)
                vectors.append({"chunk_id": uniqie_id, "file_id": file_id, "file_name": file.filename.replace(
                    ' ', '_'), "raw_chunk": chunk, "vector_chunk": vector_text[0], "token_count": token_count, "created_at": created_at, "expires_at": expires_at})
                doc_id += 1

            # Store the embedded vectors in MongoDB Atlas
            client.save(COLLECTION_NAME, vectors)

            # Delete temp folder after it's usage
            try:
                shutil.rmtree(folder_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            results.append({"filename": file.filename,
                           "message": "Document uploaded successfully"})
        except ValueError as e:
            results.append({"filename": file.filename, "error": str(e)})

    return results


@app.delete("/delete-documents/")
async def delete_documents(document_ids: List[str]):
    try:
        # Convert string document IDs to ObjectId
        object_ids = [str(doc_id) for doc_id in document_ids]

        # Delete documents from MongoDB Atlas
        deleted_count = client.delete(
            COLLECTION_NAME, object_ids, 'file_id')

        return {"message": f"{deleted_count} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/")
async def chat(chatRequest: ChatRequest):
    try:
        api_res = []
        retriver = VectorRetriever(openai, client)
        context = await retriver.invoke(chatRequest, [COLLECTION_NAME])
        context_data = json.loads(context)[0]
        response = await openai.fetch_chat_response(chatRequest.question, context_data['text'])
        api_res.append({"question": chatRequest.question,
                        "message": response})
        return api_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
