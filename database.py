import json
from typing import List
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class MongoDBAtlasClient:
    """
    A class for interacting with MongoDB Atlas.

    Attributes:
        uri (str): The URI for connecting to MongoDB Atlas.
        db_name (str): The name of the MongoDB database.
    """

    def __init__(self, uri: str, db_name: str):
        """
        Initialize the MongoDBAtlasClient with URI, database name, and OpenAIEmbeddings instance.

        Args:
            uri (str): The URI for connecting to MongoDB Atlas.
            db_name (str): The name of the MongoDB database.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def insert_document(self, collection_name: str, document: dict) -> str:
        """
        Insert a single document into a MongoDB collection.

        Args:
            collection_name (str): The name of the collection to insert the document into.
            document (dict): The document to insert into the collection.

        Returns:
            str: The ObjectId of the inserted document.
        """
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id

    def insert_documents(self, collection_name: str, documents: List[dict]) -> List[str]:
        """
        Insert multiple documents into a MongoDB collection.

        Args:
            collection_name (str): The name of the collection to insert the documents into.
            documents (List[dict]): The list of documents to insert into the collection.

        Returns:
            List[str]: The list of ObjectIds of the inserted documents.
        """
        collection = self.db[collection_name]
        result = collection.insert_many(documents)
        return result.inserted_ids

    def delete_documents(self, collection_name: str, document_ids: List[str], field: str) -> int:
        """
        Delete documents from a MongoDB collection based on provided document IDs.

        Args:
            collection_name (str): The name of the collection from which to delete documents.
            document_ids (List[str]): The list of document IDs to delete.
            field (str): The field to match document IDs against.

        Returns:
            int: The number of documents deleted.
        """
        collection = self.db[collection_name]
        result = collection.delete_many({field: {"$in": document_ids}})
        return result.deleted_count

    @staticmethod
    def get(self, collection: str, query: dict, projection: dict, return_type: str = 'single'):
        """
        Retrieve data from a MongoDB collection based on provided parameters.

        Args:
            collection (str): The name of the MongoDB collection from which data is retrieved.
            query (dict): A dictionary specifying filter conditions for the query.
            projection (dict): A dictionary specifying which fields to include or exclude in the returned documents.
            return_type (str, optional): Specifies whether to return a single document ('single') or multiple documents ('multiple'). Defaults to 'single'.

        Returns:
            result: The result of the query. Either a single document or a cursor object depending on return_type.
        """

        # Select the specified collection
        collection = self.db[collection]

        # Perform the query
        if return_type == 'single':
            result = collection.find_one(query, projection)
        else:
            result = collection.find(query, projection)

        return result

    async def create_vector_store(self, embeddings: OpenAIEmbeddings, data: List[Document], collection_name):
        try:
            # print("Creating vector store...")
            # texts = [d.page_content for d in data]
            # metadatas = [d.metadata for d in data]
            # print(texts)
            # print(metadatas)
            collection = self.db[collection_name]
            vectorStore = MongoDBAtlasVectorSearch.from_documents(
                data, embeddings, collection=collection
            )
            print("Vector store created successfully!")
        except Exception as e:
            print("An error occurred:", e)

    async def vector_search(self, collection_name, user_prompt, file_id):
        collection = self.db[collection_name]
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'rag',
                    'path': 'vector_chunk',
                    'queryVector': user_prompt[0],
                    'numCandidates': 200,
                    'limit': 10
                }
            }, {
                '$project': {
                    '_id': 0,
                    'raw_chunk': 1,
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            }
        ]
        results = collection.aggregate(pipeline)
        response = [{'raw_chunk': document['raw_chunk'],
                     'score': document['score']} for document in results]

        return response[0]['raw_chunk']
