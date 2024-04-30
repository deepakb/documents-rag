from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document


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
