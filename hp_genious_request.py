from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from typing import List
from llama_index.embeddings import HuggingFaceEmbedding
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from functools import reduce

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'ls__e9a08eb25b094353a880897253a4ffd7'

top_k =3

llm = Ollama(model="mistral")

collection_name = "hp_genious"
qdrant_client = QdrantClient(host='localhost', port=6333)

embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")
output_parser = StrOutputParser()

def retrieve_data(query: dict) -> List[dict]:
    try:
        encoded_query = embed_model.get_text_embedding(query["question"])
        result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=encoded_query,
            limit=top_k,
        )
        data = [{"id":x.id, "score": x.score, "text": x.payload['text'], "book": x.payload['book'], "chapter": x.payload['chapter']} for x in result]
    except Exception as e:
        print(f"Failed to get context: {e}")
        return []

    return data

def setup_context(data: dict):
    context_prompt = "Context:\n"
    for item in data["data"]:
        context_prompt += f"""

    {item['text']}

    """
    return context_prompt

def get_data_multi(questions):
    mapped = map(retrieve_data, questions)
    result_list = reduce(lambda x, y: x+y, mapped)
    return result_list

def aggragate_data_multi(data: list) -> List[dict]:
    unique_data = {d['id']: d for d in data}.values()

    sorted_data = sorted(unique_data, key=lambda x: x['score'], reverse=True)

    top_data = sorted_data[:top_k]

    context_prompt = "Context:\n"
    for item in top_data:
        context_prompt += f"""

    {item['text']}

    """


    return top_data

def ask_multi_question(question: str):
    template= """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    questions = (
        prompt_perspectives 
        | llm 
        | StrOutputParser() 
        | (lambda x: [{"question": line} for line in x.split("\n") if line.strip()])
    )

    retrieval_chain = questions | get_data_multi | aggragate_data_multi
    
    template = ChatPromptTemplate.from_messages([
        ("system", "you are a helpfull AI bot. You are provided with excerpts from books. Use these excerpts to address the user's question in max 2 short sentences.\n{context_prompt}\n"
        "If the necessary information is not contained within the provided context, respond with 'I don't know'.\n" 
        "When you have sufficient information to answer, please adhere to the following format:\n"

        "Question: Slughorn teaches his students that Amortentia smells different to each person. What food does Harry smell?\n"
        "Answer: Harry smells the scent of treacle tart when he inhales Amortentia.\n"
        "source: book6_half_blood_prince_Chapter_9_THE_HALF_BLOOD_PRINCE\n"

        ),
        ("human", "Question: {question}\n"),
    ])
    
    rag_chain = (
        {"question": itemgetter("question"), "context_prompt": retrieval_chain}
        | template
        | llm
        | output_parser
    )
    try:
        result = rag_chain.invoke({"question": question})
        return result
    except Exception as e:
        print(f"Failed to ask: {e}")
        return "An error occurred while processing your rwequest."
    

def ask_simple(question: str):
    template = ChatPromptTemplate.from_messages([
        ("system", "you are a helpfull AI bot. Address the user's question in max 2 short sentences."
        "Question: Slughorn teaches his students that Amortentia smells different to each person. What food does Harry smell?\n"
        "Answer: Harry smells the scent of treacle tart when he inhales Amortentia.\n"

        ),
        ("human", "Question: {question}\n"),
    ])
        
    chain = (
        template
        | llm
        | output_parser     
    )
    try:
        result = chain.invoke({"question": question})
        return result
    except Exception as e:
        print(f"Failed to ask: {e}")
        return "An error occurred while processing your request."
    
def ask_with_rag(question: str):
    

    template = ChatPromptTemplate.from_messages([
        ("system", "you are a helpfull AI bot. You are provided with excerpts from books. Use these excerpts to address the user's question in max 2 short sentences."
        "If the necessary information is not contained within the provided context, respond with 'I don't know'.\n" 
        "\n{context_prompt}\n"
        "When you have sufficient information to answer, please adhere to the following format:\n"

        "Question: Slughorn teaches his students that Amortentia smells different to each person. What food does Harry smell?\n"
        "Answer: Harry smells the scent of treacle tart when he inhales Amortentia.\n"
        "source: book6_half_blood_prince_Chapter_9_THE_HALF_BLOOD_PRINCE\n"

        ),
        ("human", "Question: {question}\n"),
    ])


    setup_and_retrieval = RunnableParallel(
    {"question": RunnablePassthrough(), "data": retrieve_data}
     )
    setup_context_prompt = RunnableParallel(
    {"context_prompt": setup_context, "question": RunnablePassthrough()}
    )
    
    rag_chain = (
        setup_and_retrieval
        |setup_context_prompt
        | template
        | llm
        | output_parser
    )
    try:
        result = rag_chain.invoke({"question": question})
        return result
    except Exception as e:
        print(f"Failed to ask: {e}")
        return "An error occurred while processing your rwequest."

multi_question_answer = ask_with_rag("What is the name of the potion that restores a person who has been petrified?")
print(multi_question_answer)