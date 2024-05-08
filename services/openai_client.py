from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List
import html
from langchain_core.output_parsers import StrOutputParser

from core.prompts import ALTERNATE_QUESTION_PROMPT, DOCUMENT_CHAT_PROMPT


class OpenAIClient:
    """
    A client class for interacting with OpenAI services.

    Attributes:
        embeddings (OpenAIEmbeddings): An instance of OpenAIEmbeddings for creating embeddings.
        llm (ChatOpenAI): An instance of ChatOpenAI for chat interactions.

    Methods:
        create_embedding(text: str) -> List[List[float]]: 
            Creates embeddings for the input text.
        chat(prompt_template: ChatPromptTemplate, payload) -> str: 
            Initiates a chat using the provided prompt template and payload.
        fetch_alternate_questions(que: str, no_of_questions: int) -> str: 
            Fetches alternate questions based on the input query.
        fetch_chat_response(que: str, context: str) -> str: 
            Fetches chat response as per document context
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAIClient with the provided API key.

        Args:
            api_key (str): The API key for accessing OpenAI services.
        """
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            openai_api_key=api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
            api_key=api_key
        )
        self.output_parser = StrOutputParser()

    async def create_embedding(self, text: str) -> List[List[float]]:
        """
        Creates embeddings for the input text.

        Args:
            text (str): The input text to create embeddings for.

        Returns:
            List[List[float]]: A list of embeddings for the input text.
        """
        vector_text = self.embeddings.embed_documents([text])
        return vector_text

    async def _chat(self, prompt_template: ChatPromptTemplate, payload):
        """
        Initiates a chat using the provided prompt template and payload.

        Args:
            prompt_template (ChatPromptTemplate): The prompt template for initiating the chat.
            payload: The payload to be used in the chat.

        Returns:
            str: The response from the chat.
        """
        chain = prompt_template | self.llm | self.output_parser
        response = chain.invoke(payload)
        return response

    async def fetch_chat_response(self, que: str, context: str):
        """
        Fetches alternate questions based on the input query and context.

        Args:
            que (str): The input query for which alternate questions are to be fetched.
            context (str): The context string to provide additional information.

        Returns:
            str: The response containing alternate questions.
        """
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        DOCUMENT_CHAT_PROMPT,
                    ),
                    ("human", "{que}"),
                ]
            )
            response = await self._chat(prompt, {
                "context": context,
                "que": que
            })

            if response:
                return [html.escape(response) for response in response.split('\n')]
            else:
                return [html.escape(response)]
        except Exception as e:
            print(e)

    async def fetch_alternate_questions(self, que: str, no_of_questions: int) -> str:
        """
        Fetches alternate questions based on the input query.

        Args:
            que (str): The input query for which alternate questions are to be fetched.
            no_of_questions (int): The number of alternate questions to fetch.

        Returns:
            str: The response containing alternate questions.
        """
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        ALTERNATE_QUESTION_PROMPT,
                    ),
                    ("human", "{que}"),
                ]
            )
            response = await self._chat(prompt, {
                "no_of_questions": no_of_questions,
                "que": que
            })

            if (response):
                return [html.escape(response) for response in response.split('\n')]
            else:
                return [html.escape(response)]
        except Exception as e:
            print(e)
