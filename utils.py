import os
import tiktoken
from dotenv import load_dotenv, find_dotenv

# Define encoding
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

# Load .env variable
load_dotenv(find_dotenv())


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value


def check_file_type(filename: str) -> bool:
    valid_extensions = ['txt', 'docx', 'doc', 'pdf', 'ppt']
    file_extension = filename.split(".")[-1]
    return file_extension in valid_extensions


def get_token_counts(string: str) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens
