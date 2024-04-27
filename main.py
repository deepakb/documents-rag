import os
import uuid
import datetime
import shutil
import tiktoken
from dotenv import load_dotenv, find_dotenv
from typing import List
from fastapi import FastAPI, File, UploadFile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient

from database import MongoDBAtlasClient

# Load .env variable
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI()

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
    openai_api_key=get_env_variable('OPENAI_API_KEY')
)

# Define encoding
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

# MongoDB Atlas integration (Replace placeholders with your MongoDB Atlas connection details)
client = MongoDBAtlasClient(get_env_variable('MONGODB_URI'), 'documents_rag')

# Define a function to check if the file type is valid
def check_file_type(filename: str) -> bool:
    valid_extensions = ['txt', 'docx', 'doc', 'pdf', 'ppt']
    file_extension = filename.split(".")[-1]
    return file_extension in valid_extensions

# Return the number of tokens in a text string
def get_token_counts(string: str) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Define a function to parse text from various document formats
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

async def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=0
    )
    chunks = splitter.split_text("\n".join(text))
    return chunks

# Define a function to embed text into vectors using Langchain
async def embed_text(documents: List[Document], request_id: str, file_id: str, file_name: str):
    all_texts = [d.page_content for d in documents]
    all_tokens = sum([get_token_counts(d) for d in all_texts])
    # We are keeping higher than normal chunk sizes since we want to load this quickly.
    # The 8100 is due to the limit of the embedding model which is 8192
    chunk_size = all_tokens if all_tokens < 8100 else 8100
    chunks = await split_text_into_chunks(all_texts, chunk_size);

    created_at = datetime.datetime.now()
    expires_at = created_at + datetime.timedelta(days=1)

    text_chunks = []
    doc_id = 1
    for chunk in chunks:
        uniqie_id = request_id + '-' +  str(doc_id)
        token_count = get_token_counts(chunk)
        vector_text = embeddings.embed_documents([chunk])
        text_chunks.append({ "chunk_id": uniqie_id, "file_id": file_id, "file_name": file_name.replace(' ', '_'), "raw_chunk": chunk, "vector_chunk": vector_text, "token_count": token_count, "created_at": created_at, "expires_at": expires_at })
        doc_id += 1

    return text_chunks

# Define the ping endpoint to make sure API is working
@app.get("/ping/")
async def ping():
    return { 'status': 'success', 'message': 'pong' }

# Define the document upload endpoint
@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
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

            # Embed the parsed text into vectors
            vectors = await embed_text(documents, request_id, file_id, file.filename)

            # Store the embedded vectors in MongoDB Atlas
            client.insert_documents('files', vectors)

            # Delete temp folder after it's usage
            try:
                shutil.rmtree(folder_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

            results.append({"filename": file.filename, "message": "Document uploaded successfully"})
        except ValueError as e:
            results.append({"filename": file.filename, "error": str(e)})

    return results
