import os
import datetime
import shutil
import requests
import re
import subprocess
import uuid
import glob
from fastapi import HTTPException
from loguru import logger
from bson import ObjectId
from typing import List, Tuple
from fastapi import UploadFile
from pymongo.results import InsertOneResult, UpdateResult
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NotebookLoader, DirectoryLoader
from rank_bm25 import BM25Okapi

from utils.utils import get_token_counts, clean_and_tokenize
from services.openai_client import OpenAIClient
from services.database import MongoDBAtlasClient
from services.api_response import Response
from models.embedded_document import EmbeddedDocumentRepository, EmbeddedDocument as EmbeddedDocumentModel
from models.document import DocumentRepository, Document as DocumentModel
from models.embedded_github import EmbeddedGithubDocument as EmbeddedGithubModel, EmbeddedGithubDocumentRepository


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
                # Save document for processing
                document_id = self._create_document(
                    'github', repo_name, repo_url)
                if document_id is None:
                    raise ValueError("Not able to process github repo")

                documents = await self._load_document(folder_path, document_id)
                vectors = await self._create_vectors(documents)

                # Store the embedded vectors in MongoDB Atlas
                embedded_gh_doc_repo = EmbeddedGithubDocumentRepository(
                    database=self.mongo_client.db)
                embedded_gh_doc_repo.save_many(vectors)

                # Delete temp folder after it's usage
                try:
                    shutil.rmtree(folder_path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

                # Update the documet status to completed
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

    async def _load_document(self, repo_path: str, document_id: str):
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
        notebook_files = glob.glob(f'{repo_path}/**/*.ipynb', recursive=True)
        other_files = glob.glob(f'{repo_path}/**/*.*', recursive=True)
        loaded_documents = []
        documents_dict = {}

        try:
            if notebook_files:
                notebook_loader = NotebookLoader(repo_path)
                notebook_documents = notebook_loader.load() if callable(notebook_loader.load) else []
                loaded_documents.extend(notebook_documents)

            if other_files:
                directory_loader = DirectoryLoader(repo_path)
                other_documents = directory_loader.load() if callable(directory_loader.load) else []
                loaded_documents.extend(other_documents)

            if loaded_documents:
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id
                    doc.metadata['document_id'] = document_id

                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files: {e}")


        return documents_dict

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

    async def _create_vectors(self, documents: dict[str, Document]) -> List[EmbeddedGithubModel]:
        """
        Create embedding vectors for the given documents.

        Args:
            documents (List[Document]): List of documents.
            document_id (str): ID of the document.

        Returns:
            List[EmbeddedGithubModel]: List of embedded github document models.
        """
        split_documents = []
        for _file_id, doc in documents.items():
            texts = doc.page_content
            tokens = get_token_counts(texts)
            # model limit is 8192
            chunk_size = tokens if tokens < 8100 else 8100
            split_docs = await self._split_text_into_chunks(texts, chunk_size)
            for split_doc in split_docs:
                split_doc.metadata['file_id'] = doc.metadata['file_id']
                split_doc.metadata['source'] = doc.metadata['source']
                split_doc.metadata['document_id'] = doc.metadata['document_id']

            split_documents.extend(split_docs)

        index = None
        if split_documents:
            tokenized_documents = [clean_and_tokenize(
                doc.page_content) for doc in split_documents]
            index = BM25Okapi(tokenized_documents)
        return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]

        # vectors = []
        # doc_id = 1

        # for d in documents:
        #     texts = d.page_content
        #     tokens = get_token_counts(texts)
        #     # model limit is 8192
        #     chunk_size = tokens if tokens < 8100 else 8100
        #     chunks = await self._split_text_into_chunks([texts], chunk_size)

        #     created_at = datetime.datetime.now()
        #     expires_at = created_at + datetime.timedelta(days=1)

        #     for chunk in chunks:
        #         unique_id = document_id + '-' + str(doc_id)
        #         token_count = get_token_counts(chunk)
        #         vector_text = await self.openai.create_embedding(chunk)
                
        #         # Access metadata from the document
        #         metadata = d.metadata
                
        #         vectors.append(
        #             EmbeddedGithubModel(
        #                 id=ObjectId(),
        #                 chunk_id=unique_id,
        #                 document_id=document_id,
        #                 raw_chunk=chunk,
        #                 vector_chunk=vector_text[0],
        #                 token_count=token_count,
        #                 created_at=created_at,
        #                 expires_at=expires_at,
        #                 metadata=metadata
        #             )
        #         )
        #         doc_id += 1

        # return vectors
    
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
            encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=100
        )
        chunks = splitter.split_text(text)
        return chunks
