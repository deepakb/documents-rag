from typing import List, Union
from pymongo import MongoClient


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
