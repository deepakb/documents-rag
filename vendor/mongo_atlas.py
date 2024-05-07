from services.database import MongoDBAtlasClient
from config.settings import mongo


def get_mongodb_client():
    """
    Dependency provider function to initialize MongoDBAtlasClient.

    Returns:
        MongoDBAtlasClient: An instance of MongoDBAtlasClient.
    """
    return MongoDBAtlasClient(mongo.uri, mongo.database)
