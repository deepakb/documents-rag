import os
import datetime
import shutil
import requests
import re
import subprocess
import uuid
from fastapi import HTTPException
from loguru import logger
from bson import ObjectId
from typing import List, Tuple
from fastapi import UploadFile
from pymongo.results import InsertOneResult, UpdateResult
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NotebookLoader, DirectoryLoader

from utils.utils import get_token_counts
from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.api_response import Response
from models.embedded_document import EmbeddedDocumentRepository, EmbeddedDocument as EmbeddedDocumentModel
from models.document import DocumentRepository, Document as DocumentModel


class GithubHandler:
    """
    Handles document processing, including upload, processing, and deletion.
    """

    def __init__(self, openai: OpenAIClient, mongo_client: MongoDBAtlasClient):
        """
        Initializes the GithubHandler with OpenAI client and MongoDB client.

        Args:
            openai (OpenAIClient): OpenAI client for text embedding.
            mongo_client (MongoDBAtlasClient): MongoDB client for database operations.
        """
        self.openai = openai
        self.mongo_client = mongo_client

    async def process(self, repo_url: str):
        """
        Processes uploaded files, saves them, extracts text, creates embeddings, and stores them.

        Args:
            files (List[UploadFile]): List of files to process.

        Returns:
            Response: Response object indicating success or failure of the operation.
        """
        try:
            # Check url is valid or not
            if not self.is_valid_github_repo_url(repo_url):
                raise ValueError(f"Invalid url {repo_url}")

            repo_name = repo_url.split("/")[-1]
            temp_folder_path = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            mounted_path = '/tmp'
            folder_path = os.path.join(mounted_path, temp_folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                os.chmod(folder_path, 0o755)
            cloned_repo = await self._clone_github_repo(repo_url, folder_path)

            if cloned_repo:
                documents = await self._load_document(folder_path)
                print(documents)

                # # Save document for processing
                # document_id = self._create_document(
                #     'github', repo_name, repo_url)
                # if document_id is None:
                #     raise ValueError("Not able to process github repo")

                # # Update the documet status to completed
                # document_repo = DocumentRepository(
                #     database=self.mongo_client.db)
                # document_repo.update_document(
                #     {"_id": ObjectId(document_id)}, {"status": "completed"})

            # # Save the file to a temp location
            # folder_path, full_file_path, file_extension, file_name = await self._save_file_temp_loc(file)

            # # Parse text from the uploaded document and loads as documents
            # documents = await self._load_document(full_file_path, file_extension)

            # # Create embedding records for all documents given
            # vectors = await self._create_vectors(documents, document_id, file_name)

            # # Store the embedded vectors in MongoDB Atlas
            # embedded_doc_repo = EmbeddedDocumentRepository(
            #     database=self.mongo_client.db)
            # embedded_doc_repo.save_many(vectors)

            # document_repo = DocumentRepository(
            #     database=self.mongo_client.db)
            # document_repo.update_document(
            #     {"_id": ObjectId(document_id)}, {"status": "completed"})

            # Delete temp folder after it's usage
            # try:
            #     shutil.rmtree(folder_path)
            # except OSError as e:
            #     print("Error: %s - %s." % (e.filename, e.strerror))

            # results.append(
            #     {"file_name": file_name, "message": "Document uploaded successfully"})
        except ValueError as e:
            # results.append({"file_name": file_name, "error": str(e)})
            pass

        return Response.success(data={})

    def is_valid_github_repo_url(self, url):
        # Regular expression pattern for GitHub repository URL
        pattern = r'^https?://github\.com/[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+$'

        # Check if the URL matches the pattern
        if re.match(pattern, url):
            # Make an HTTP GET request to the URL
            response = requests.get(url)
            # Check if the response status is OK (200)
            if response.status_code == 200:
                return True
        return False

    async def _clone_github_repo(self, repo_url: str, folder_path: str):
        try:
            subprocess.run(
                ['git', 'clone', repo_url, folder_path], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return False

    def _create_document(self, type: str, name: str, url: str):
        """
        Saves the document in the DocumentRepository and returns its ID.

        Args:
            ext (str): Extension of the document.
            file_name (str): Name of the file.

        Returns:
            str: ID of the saved document.
        """
        document = DocumentModel(
            id=ObjectId(),
            name=name,
            type=type,
            url=url,
            status="pending"
        )
        document_repo = DocumentRepository(
            database=self.mongo_client.db)
        response = document_repo.save(document)

        return self._get_document_id(response)

    def _get_document_id(self, response):
        """
        Extracts the document ID from the database response.

        Args:
            response: Response from the database.

        Returns:
            str: ID of the document.
        """
        if isinstance(response, InsertOneResult):
            return str(response.inserted_id)
        elif isinstance(response, UpdateResult):
            if response.upserted_id:
                return str(response.upserted_id)
            elif response.inserted_id:
                return str(response.inserted_id)
            else:
                return None
        else:
            return None

    async def _load_document(self, repo_path: str) -> List[Document]:
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
        extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'ts', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json',
                      'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

        for ext in extensions:
            glob_pattern = f'**/*.{ext}'
            try:
                loader = None
                if ext == 'ipynb':
                    loader = NotebookLoader(
                        str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
                else:
                    loader = DirectoryLoader(repo_path, glob=glob_pattern)

                documents = loader.load() if callable(loader.load) else []
                print(documents)
            except Exception as e:
                print(
                    f"Error loading files with pattern '{glob_pattern}': {e}")
                continue

        return documents

    async def _load_and_index_files(self, repo_path):
        extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'ts', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json',
                      'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

        file_type_counts = {}
        documents_dict = {}

        for ext in extensions:
            glob_pattern = f'**/*.{ext}'
            try:
                loader = None
                if ext == 'ipynb':
                    loader = NotebookLoader(
                        str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
                else:
                    loader = DirectoryLoader(repo_path, glob=glob_pattern)

                loaded_documents = loader.load() if callable(loader.load) else []
                if loaded_documents:
                    file_type_counts[ext] = len(loaded_documents)
                    for doc in loaded_documents:
                        file_path = doc.metadata['source']
                        relative_path = os.path.relpath(file_path, repo_path)
                        file_id = str(uuid.uuid4())
                        doc.metadata['source'] = relative_path
                        doc.metadata['file_id'] = file_id

                        documents_dict[file_id] = doc
            except Exception as e:
                print(
                    f"Error loading files with pattern '{glob_pattern}': {e}")
                continue

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=3000, chunk_overlap=200)

        # split_documents = []
        # for file_id, original_doc in documents_dict.items():
        #     split_docs = text_splitter.split_documents([original_doc])
        #     for split_doc in split_docs:
        #         split_doc.metadata['file_id'] = original_doc.metadata['file_id']
        #         split_doc.metadata['source'] = original_doc.metadata['source']

        #     split_documents.extend(split_docs)

        # index = None
        # if split_documents:
        #     tokenized_documents = [clean_and_tokenize(
        #         doc.page_content) for doc in split_documents]
        #     index = BM25Okapi(tokenized_documents)
        # return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]
