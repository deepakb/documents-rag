from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List


class OpenAIClient:
    def __init__(self, api_key: str):
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
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that provides information from documents. {context}\n"
             "If there is no relevant information in the provided context, then say I don't know"),
            ("human", "{input}"),
        ])

    async def split_text_into_chunks(self, text: str, chunk_size: int):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=0
        )
        chunks = splitter.split_text("\n".join(text))
        return chunks

    async def create_embedding(self, text: str) -> List[List[float]]:
        vector_text = self.embeddings.embed_documents([text])
        return vector_text

    async def chat(self, prompt: str, file_id: str, context, max_tokens: int = 50):
        f_prompt = self.prompt_template.format(context=context, input=prompt)
        respone = self.llm.invoke(f_prompt)
        return respone
