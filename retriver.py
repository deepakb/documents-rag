import json
from typing import List
from models import ChatRequest
from openai_client import OpenAIClient
from database import MongoDBAtlasClient


class VectorRetriever:
    """
    A class for retrieving vectors from a MongoDB database using OpenAI's services.

    Attributes:
        openai (OpenAIClient): An instance of OpenAIClient for fetching alternate questions and creating embeddings.
        mongo_client (MongoDBAtlasClient): An instance of MongoDBAtlasClient for database operations.
        messages (list): A list of system messages.

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
        self.messages = [
            {
                "role": "system",
                "content": "You are an intelligent assistant tasked with aiding users in obtaining contextually relevant answers."
            }]
        self.openai = openai
        self.mongo_client = mongo_client

    async def invoke(self, chatRequest: ChatRequest, collections=List[str], filters=dict) -> dict:
        """
        Invokes OpenAI to fetch alternate questions based on the input chat request.

        Args:
            chatRequest (ChatRequest): An instance of ChatRequest containing the user's question.
            collections (List[str]): A list of MongoDB collections to search.
            filters (dict): Optional filters for refining the search.

        Returns:
            dict: A dictionary containing alternate questions fetched from OpenAI.
        """
        response = await self.openai.fetch_alternate_questions(chatRequest.question, 5)
        return response

    def do_vector_search(self, collections: List[str], source: List[str], pre_filters: dict) -> str:
        """
        Performs vector search on the specified collections and returns results.

        Args:
            collections (List[str]): A list of MongoDB collections to search.
            source (List[str]): A list of strings representing vectors to search for.
            pre_filters (dict): Filters to apply before performing the search.

        Returns:
            str: A JSON string containing the search results.
        """
        allResults = []
        queryVectors = [self.openai.create_embedding(
            query) for query in source]

        for col in collections:
            results = []
            collection = self.db[col]
            for queryVector in queryVectors:
                params = {
                    "queryVector": queryVector,
                    "path": "documents",
                    "numCandidates": 100,
                    "limit": 1,
                    "index": "raw",
                }

                if pre_filters:
                    params["filter"] = pre_filters

                query = {"$vectorSearch": params}

                pipeline = [
                    query,
                    {"$set": {"score": {"$meta": "vectorSearchScore"}}}
                ]

                response = collection.aggregate(pipeline=pipeline)
                for res in response:
                    chunkTextData = json.loads(self.mongo_client.get(col, {"chunkId": res['chunkId']}, {
                                               "chunkValueRaw": 1, "fileName": 1, "_id": 0}, 'single'))
                    results.append({'score': res['score'], 'text': chunkTextData['chunkValueRaw'],
                                   'source':  chunkTextData['fileName'], "kbStore": res['kbStore']})

            if len(results) > 0:
                sorted_results = sorted(results, key=lambda x: x['score'])
                allResults.append(sorted_results[0])

        return json.dumps(allResults)
