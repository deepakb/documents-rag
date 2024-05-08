import tiktoken
from typing import List

# Define encoding for the specified model
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')


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
