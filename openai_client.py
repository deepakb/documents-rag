from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List


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
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAIClient with the provided API key.

        Args:
            api_key (str): The API key for accessing OpenAI services.
        """
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024,
            openai_api_key=api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0,
            api_key=api_key
        )

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

    def chat(self, prompt_template: ChatPromptTemplate, payload) -> str:
        """
        Initiates a chat using the provided prompt template and payload.

        Args:
            prompt_template (ChatPromptTemplate): The prompt template for initiating the chat.
            payload: The payload to be used in the chat.

        Returns:
            str: The response from the chat.
        """
        chain = prompt_template | self.llm
        response = chain.invoke(payload)
        return response

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
                        "Your role is to assist by creating {no_of_questions} diverse renditions of the user query, aimed at retrieving pertinent documents from a vector database. By offering varied perspectives on the query, the aim is to mitigate the constraints of distance-based similarity search. Deliver these alternative inquiries in a list format, each separated by a newline.",
                    ),
                    ("human", "{que}"),
                ]
            )
            response = self.chat(prompt, {
                "no_of_questions": no_of_questions,
                "que": que
            })
            return response
        except Exception as e:
            print(e)
