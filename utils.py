import os
import tiktoken
from typing import List

from dotenv import load_dotenv, find_dotenv

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
