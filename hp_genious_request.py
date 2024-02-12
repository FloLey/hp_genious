from qdrant_client import QdrantClient
from typing import List
from llama_index.embeddings import HuggingFaceEmbedding
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

collection_name = "hp_genious"
qdrant_client = QdrantClient(host='localhost', port=6333)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

def get_data(query: str, top_k: int) -> List[dict]:
    try:
        encoded_query = embed_model.get_text_embedding(query)
        result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=encoded_query,
            limit=top_k,
        )
        data = [{"text": x.payload['text'], "book": x.payload['book']} for x in result]
        return data
    except Exception as e:
        print(f"Failed to get context: {e}")
        return []

def create_chat_messages(question: str) -> List[dict]:
    data = get_data(question, top_k=5)
    context_prompt = "Context:\n"
    for item in data:
        context_prompt += f"""

{item['text']}

"""
        
    context_prompt += """
you are provided with excerpts from books relevant to the context. 
Use these excerpts to address the user's question. 
You may engage in hypothesizing to form an educated guess, but ensure to clearly indicate when you are doing so. 
If the necessary information is not contained within the provided context, respond with 'I don't know'. 
When you have sufficient information to answer, please adhere to the following format:

<answer to the question>
source: <the used chapter from the context>
"""
    
    messages = [
        {"role": "system", "content": context_prompt},
        {"role": "user", "content": question}
    ]
    
    return messages

def ask_with_rag(messages: List[dict], model="gpt-3.5-turbo"):
    try:        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Failed to ask OpenAI: {e}")
        return "An error occurred while processing your request."


