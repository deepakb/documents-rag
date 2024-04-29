from pymongo import MongoClient


class MongoDBAtlasClient:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def insert_document(self, collection_name, document):
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id

    def insert_documents(self, collection_name, documents):
        collection = self.db[collection_name]
        result = collection.insert_many(documents)
        return result.inserted_ids

    def delete_documents(self, collection_name, document_ids, field):
        collection = self.db[collection_name]
        result = collection.delete_many({field: {"$in": document_ids}})
        return result.deleted_count
