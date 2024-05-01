import os
import tiktoken
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader

# Define encoding for the specified model
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

# Load environment variables from the .env file
load_dotenv(find_dotenv())


def get_env_variable(var_name: str) -> str:
    """
    Retrieves the value of the specified environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the specified environment variable.

    Raises:
        ValueError: If the specified environment variable is not found.
    """
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value


def check_file_type(filename: str) -> bool:
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


def get_token_counts(string: str) -> int:
    """
    Counts the number of tokens in the provided string.

    Args:
        string (str): The input string to count tokens from.

    Returns:
        int: The number of tokens in the string.
    """
    num_tokens = len(encoding.encode(string))
    return num_tokens


async def load_document(file: str, file_extension: str) -> List[Document]:
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


async def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
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
