from pydantic_settings import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """
    Base settings class for loading configuration from environment variables or a .env file.

    Attributes:
        Config (class): Configuration options for the settings class.
            env_file (str): Path to the .env file for loading environment variables (defaults to ".env").
            extra (str): Handling of extra input variables (defaults to "allow" to allow extra variables).
    """
    class Config:
        env_file = ".env"
        extra = "allow"


class APISettings(BaseSettings):
    """
   Settings class for API configuration.

    Attributes:
        prefix (str): Prefix for API endpoints (defaults to "/api").
        version (str): API version (defaults to "0.1.0").
        openai_key (str): API key for OpenAI.
        debug (bool): Flag indicating whether debugging mode is enabled.
    """
    project_name: str = "documentsrag"
    prefix: str = "/api"
    version: str = "0.1.0"
    openai_key: str
    debug: bool

    class Config:
        env_prefix = "API_"


class MongoDBSettings(BaseSettings):
    """
    Settings class for MongoDB configuration.

    Attributes:
        uri (str): URI for connecting to MongoDB.
        database (str): Name of the MongoDB database.
        documents_collection (str): Name of the collection for documents.
        embedded_collection (str): Name of the collection for embedded documents.
    """
    uri: str
    database: str
    documents_collection: str = "documents"
    embedded_collection: str = "embedded_documents"
    embedded_github: str = "embedded_github"

    class Config:
        env_prefix = "MONGO_"


# Instances of settings classes
mongo = MongoDBSettings()
api = APISettings()
