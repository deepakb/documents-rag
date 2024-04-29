import os
import uuid
import datetime
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from database import MongoDBAtlasClient
from utils import check_file_type, get_token_counts, get_env_variable
from openai_client import OpenAIClient

DB_NAME = 'documents_rag'
COLLECTION_NAME = 'files'

# Initialize FastAPI app
app = FastAPI()

# Intialize OpenAIClient
openai = OpenAIClient(get_env_variable('OPENAI_API_KEY'))

# Initialize MongoDB Atlas client
client = MongoDBAtlasClient(get_env_variable('MONGODB_URI'), DB_NAME)


async def parse_document(file: str, file_extension: str) -> List[Document]:
    if file_extension == "pdf":
        loader = PyPDFLoader(file)
    elif file_extension in ["docx", "doc"]:
        loader = UnstructuredWordDocumentLoader(file)
    elif file_extension == "pptx":
        loader = UnstructuredPowerPointLoader(file)
    elif file_extension == "txt":
        loader = TextLoader(file)
    else:
        # Handle unsupported file formats
        raise ValueError("Unsupported file format")

    documents = loader.load()
    return documents


async def get_embedding(documents: List[Document], request_id: str, file_id: str, file_name: str):
    all_texts = [d.page_content for d in documents]
    all_tokens = sum([get_token_counts(d) for d in all_texts])
    # model limit is 8192
    chunk_size = all_tokens if all_tokens < 8100 else 8100
    chunks = await openai.split_text_into_chunks(all_texts, chunk_size)

    created_at = datetime.datetime.now()
    expires_at = created_at + datetime.timedelta(days=1)

    text_chunks = []
    doc_id = 1
    for chunk in chunks:
        uniqie_id = request_id + '-' + str(doc_id)
        token_count = get_token_counts(chunk)
        vector_text = openai.create_embedding(chunk)
        text_chunks.append({"chunk_id": uniqie_id, "file_id": file_id, "file_name": file_name.replace(
            ' ', '_'), "raw_chunk": chunk, "vector_chunk": vector_text, "token_count": token_count, "created_at": created_at, "expires_at": expires_at})
        doc_id += 1

    return text_chunks


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
            documents = await parse_document(full_file_path, file_extension)

            # Create embedding for the documents
            vectors = await get_embedding(documents, request_id, file_id, file.filename)
            # await client.create_vector_store(openai.embeddings, documents, COLLECTION_NAME)

            # Store the embedded vectors in MongoDB Atlas
            client.insert_documents(COLLECTION_NAME, vectors)

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
        deleted_count = client.delete_documents(
            COLLECTION_NAME, object_ids, 'file_id')

        return {"message": f"{deleted_count} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(prompt: str, file_id: str):
    try:
        response = await openai.chat(prompt, file_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
