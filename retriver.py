import json
from typing import List
from model import ChatRequest
from openai_client import OpenAIClient
from database import MongoDBAtlasClient


class VectorRetriever:
    """
    A class for retrieving vectors from a MongoDB database using OpenAI's services.

    Attributes:
        openai (OpenAIClient): An instance of OpenAIClient for fetching alternate questions and creating embeddings.
        mongo_client (MongoDBAtlasClient): An instance of MongoDBAtlasClient for database operations.

    Methods:
        invoke(chatRequest: ChatRequest, collections: List[str], filters: dict) -> dict: 
            Invokes OpenAI to fetch alternate questions based on the input chat request.
        do_vector_search(collections: List[str], source: List[str], pre_filters: dict) -> str: 
            Performs vector search on the specified collections and returns results.
    """

    def __init__(self, openai: OpenAIClient, mongo_client: MongoDBAtlasClient):
        """
        Initializes VectorRetriever with OpenAIClient and MongoDBAtlasClient instances.

        Args:
            openai (OpenAIClient): An instance of OpenAIClient.
            mongo_client (MongoDBAtlasClient): An instance of MongoDBAtlasClient.
        """
        self.openai = openai
        self.mongo_client = mongo_client

    async def invoke(self, chatRequest: ChatRequest, collections=List[str]):
        """
        Invokes OpenAI to fetch alternate questions based on the input chat request.

        Args:
            chatRequest (ChatRequest): An instance of ChatRequest containing the user's question, file_id and filters.
            collections (List[str]): A list of MongoDB collections to search.

        Returns:
            str: A dictionary containing alternate questions fetched from OpenAI.
        """
        (file_id, question, filters) = chatRequest
        varients = await self.openai.fetch_alternate_questions(question, 5)
        response = await self._do_vector_search(collections, varients, filters)
        return response

    async def _do_vector_search(self, collections: List[str], source: List[str], pre_filters: dict):
        """
        Performs vector search on the specified collections and returns results.

        Args:
            collections (List[str]): A list of MongoDB collections to search.
            source (List[str]): A list of strings representing vectors to search for.
            pre_filters (dict): Filters to apply before performing the search.

        Returns:
            str: A JSON string containing the search results.
        """
        query_vectors = []
        for query in source:
            try:
                embedding = await self.openai.create_embedding(query)
                query_vectors.append(embedding)
            except Exception as e:
                print(f"Error creating embedding for query '{query}': {e}")

        for col in collections:
            results = []
            try:
                collection = self.mongo_client.db[col]
                for query_vector in query_vectors:
                    try:
                        params = {
                            "queryVector": query_vector[0],
                            "path": "vector_chunk",
                            "numCandidates": 100,
                            "limit": 1,
                            "index": "rag_doc_index",
                        }

                        # if pre_filters:
                        #     params["filter"] = pre_filters

                        query = {"$vectorSearch": params}

                        pipeline = [
                            query,
                            {"$set": {"score": {"$meta": "vectorSearchScore"}}}
                        ]

                        response = collection.aggregate(pipeline=pipeline)
                        for res in response:
                            try:
                                chunk_res = self.mongo_client.get(col, {"chunk_id": res['chunk_id']}, {
                                    "raw_chunk": 1, "file_name": 1, "_id": 0}, 'single')
                                results.append({'score': res['score'], 'text': chunk_res['raw_chunk'],
                                                'source':  chunk_res['file_name']})
                            except Exception as e:
                                print(f"Error processing result: {e}")
                    except Exception as e:
                        print(f"Error querying collection '{col}': {e}")
            except Exception as e:
                print(f"Error accessing collection '{col}': {e}")

        return json.dumps(results)
