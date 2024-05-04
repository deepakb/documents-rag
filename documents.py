import os
import uuid
import datetime
import shutil
from bson import ObjectId
from typing import List, Tuple
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader

from utils import get_token_counts
from openai_client import OpenAIClient
from database import MongoDBAtlasClient
from api_response import Response
from models.embedded_documents import EmbeddedDocumentsRepository, EmbeddedDocuments


class ProcessDocuments:
    def __init__(self, openai: OpenAIClient, mongo_client: MongoDBAtlasClient):
        self.openai = openai
        self.mongo_client = mongo_client

    async def process(self, files: List[UploadFile], collection: str):
        results = []
        for file in files:
            try:
                # Check file type is valid or throw value error
                if not self._check_file_type(file.filename):
                    raise ValueError(f"Invalid file type for {file.filename}")

                # Save the file to a temp location
                file_id = str(uuid.uuid4())
                folder_path, full_file_path, file_extension, file_name = await self._save_file_temp_loc(file)

                # Parse text from the uploaded document and loads as documents
                documents = await self._load_document(full_file_path, file_extension)

                # Create embedding records for all documents given
                vectors = await self._create_vectors(documents, file_id, file_name)

                # Store the embedded vectors in MongoDB Atlas
                repo = EmbeddedDocumentsRepository(
                    database=self.mongo_client.db)
                repo.save_many(vectors)

                # self.mongo_client.save(collection, vectors)

                # Delete temp folder after it's usage
                try:
                    shutil.rmtree(folder_path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                results.append(
                    {"file_name": file_name, "message": "Document uploaded successfully"})
            except ValueError as e:
                results.append({"file_name": file_name, "error": str(e)})

        return Response.success(data=results)

    def _check_file_type(self, filename: str) -> bool:
        """
        Checks if the given filename has a valid extension.

        Args:
            filename (str): The name of the file to check.

        Returns:
            bool: True if the file extension is valid, False otherwise.
        """
        valid_extensions = ['txt', 'docx', 'doc', 'pdf', 'ppt']
        file_extension = filename.split(".")[-1]
        return file_extension in valid_extensions

    async def _load_document(self, file: str, file_extension: str) -> List[Document]:
        """
        Loads a document from the specified file based on its extension.

        Args:
            file (str): The path to the file to be loaded.
            file_extension (str): The extension of the file.

        Returns:
            List[Document]: A list of Document objects containing the loaded content.

        Raises:
            ValueError: If the file format is not supported.
        """
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

    async def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
            Splits the input text into chunks of specified size.

            Args:
                text (str): The input text to be split into chunks.
                chunk_size (int): The size of each chunk.

            Returns:
                List[str]: A list of text chunks.
            """
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=0
        )
        chunks = splitter.split_text("\n".join(text))
        return chunks

    async def _save_file_temp_loc(self, file: UploadFile) -> Tuple[str, str, str, str]:
        """
        Saves the uploaded file to a temporary location and returns the path and file extension.

        Args:
            file (UploadFile): The uploaded file.

        Returns:
            Tuple[str, str]: The path to the saved file and the file extension.
        """
        file_name = file.filename
        temp_folder_path = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        mounted_path = '/tmp'
        folder_path = os.path.join(mounted_path, temp_folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_file_path = os.path.join(folder_path, file_name)

        # Save the uploaded file to a temporary location
        with open(full_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_extension = file_name.rsplit('.', 1)[1].lower()

        return folder_path, full_file_path, file_extension, file_name

    async def _create_vectors(self, documents: List[Document], file_id: str, file_name: str) -> List[EmbeddedDocuments]:
        texts = [d.page_content for d in documents]
        tokens = sum([get_token_counts(d) for d in texts])
        # model limit is 8192
        chunk_size = tokens if tokens < 8100 else 8100
        chunks = await self._split_text_into_chunks(texts, chunk_size)

        created_at = datetime.datetime.now()
        expires_at = created_at + datetime.timedelta(days=1)
        vectors = []
        doc_id = 1

        for chunk in chunks:
            unique_id = file_id + '-' + str(doc_id)
            token_count = get_token_counts(chunk)
            vector_text = await self.openai.create_embedding(chunk)
            vectors.append(
                EmbeddedDocuments(
                    id=ObjectId(),
                    chunk_id=unique_id,
                    documents_id=file_id,
                    file_id=file_id,
                    file_name=file_name.replace(' ', '_'),
                    raw_chunk=chunk,
                    vector_chunk=vector_text[0],
                    token_count=token_count,
                    created_at=created_at,
                    expires_at=expires_at
                )
            )
            doc_id += 1

        return vectors
