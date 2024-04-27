import os
import uuid
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, File, UploadFile
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

# Load .env variable
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# MongoDB Atlas integration (Replace placeholders with your MongoDB Atlas connection details)
client = MongoClient(os.getenv('MONGODB_URI'))

# Define a function to parse text from various document formats
def parse_document(file: UploadFile) -> list:
    file_extension = file.filename.split(".")[-1]

    if file_extension == "pdf":
        loader = PyPDFLoader(file)
    elif file_extension == "docx":
        loader = Docx2txtLoader(file)
    elif file_extension == "pptx":
        loader = UnstructuredPowerPointLoader(file)
    elif file_extension == "txt":
        loader = TextLoader(file)
    else:
        # Handle unsupported file formats
        raise ValueError("Unsupported file format")
    
    documents = loader.load()
    texts = []
    for doc in documents:
        texts.append(doc.text)

    return texts

def split_text_into_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding="cl100k_base", chunk_size=2000, chunk_overlap=20
    )
    chunks = splitter.split(text)
    return chunks

# Define a function to embed text into vectors using Langchain
def embed_text(text: str) -> list:
    # Use Langchain's OpenAI loaders to embed text into vectors
    vectors = embeddings.embed_documents([text])
    return vectors

# Define a function to check if the file type is valid
def check_file_type(filename: str) -> bool:
    valid_extensions = ['txt', 'docx', 'doc', 'pdf', 'ppt']
    file_extension = filename.split(".")[-1]
    return file_extension in valid_extensions

# Define the document upload endpoint
@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            # Check if the file type is valid
            if not check_file_type(file.filename):
                raise ValueError(f"Invalid file type for {file.filename}")
            
            # Generate a unique filename
            unique_filename = str(uuid.uuid4()) + "_" + file.filename
            # Define the path to save the file
            temp_file_path = os.path.join("/tmp", unique_filename)

            # Save the uploaded file to a temporary location
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(await file.read())

            # Parse text from the uploaded document and split into chunks
            texts = parse_document(temp_file_path)
            # Embed the parsed text into vectors
            for text in texts:
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    vectors = embed_text(chunk)

            # Store the embedded vectors in MongoDB Atlas
            # Replace "your_database" and "your_collection" with your database and collection names
            # db["your_collection"].insert_one({"text": text, "vectors": vectors})
            
            results.append({"filename": file.filename, "message": "Document uploaded successfully"})
        except ValueError as e:
            results.append({"filename": file.filename, "error": str(e)})

    return results
