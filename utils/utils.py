import tiktoken
from typing import List
import re
import nltk
import os

nltk.download("punkt")

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

def clean_and_tokenize(text):
    """
    Cleans and tokenizes the input text.

    Args:
        text (str): The input text to clean and tokenize.

    Returns:
        List[str]: List of tokens after cleaning and tokenization.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\b(?:http|ftp)s?://\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return nltk.word_tokenize(text)

def format_documents(documents):
    """
    Formats a list of documents into a numbered list with source information.

    Args:
        documents (List[Document]): List of documents.

    Returns:
        str: Numbered list of documents with source information and page content.
    """
    numbered_docs = "\n".join([f"{i+1}. {os.path.basename(doc.metadata['source'])}: {doc.page_content}" for i, doc in enumerate(documents)])
    return numbered_docs

def format_user_question(question):
    """
    Formats the user question by removing extra spaces.

    Args:
        question (str): The user question to be formatted.

    Returns:
        str: The formatted user question.
    """
    question = re.sub(r'\s+', ' ', question).strip()
    return question
