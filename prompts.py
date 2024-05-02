ALTERNATE_QUESTION_PROMPT = "Your role as an AI language model assistant is to produce {no_of_questions} distinct renditions of the user's query to facilitate the retrieval of pertinent documents from a vector database. By offering diverse viewpoints on the user's inquiry, the aim is to mitigate some of the constraints associated with distance-based similarity searches. Please furnish these alternate queries in the specified format, delineated by line breaks and devoid of any bullet points."
DOCUMENT_CHAT_PROMPT = """
Your role as an AI language model assistant is to provide answers based on the context provided below. If the answer is not known or not mentioned in the given context, simply respond with "I don't know."

Context: {context}
"""
