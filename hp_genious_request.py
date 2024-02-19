from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from typing import List
from llama_index.embeddings import HuggingFaceEmbedding
from langchain_community.llms import Ollama
from langchain_core.output_parsers import JsonOutputParser, NumberedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

llm = Ollama(model="mistral:instruct")

collection_name = "hp_genious"
qdrant_client = QdrantClient(host='localhost', port=6333)

embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1", pooling="cls")

top_k = 3


class QdrantRetriever(BaseRetriever):
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            encoded_query = embed_model.get_text_embedding(query)
            result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=encoded_query,
                limit=top_k,
            )
            documents = [Document(
                page_content=x.payload["text"],
                metadata={
                    "id": x.id,
                    "score": x.score,
                    "book": x.payload["book"],
                    "chapter": x.payload["chapter"],
                }
            ) for x in result]
        except Exception as e:
            print(f"Failed to get context: {e}")
            return []
        return documents


retriever = QdrantRetriever()


class responseHP(BaseModel):
    answer: str = Field(description="The answer to the question")
    source: str = Field(description="The source that was used to answer the question")


# Set up parsers
output_parser_response = JsonOutputParser(pydantic_object=responseHP)
output_parser_list = NumberedListOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    """Reciprocal Rank Fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula.

    Each Document in the lists is expected to have a metadata attribute containing an 'id'.
    """

    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs, start=1):  # Ensure rank starts at 1 for correct RRF calculation
            # Use the document's ID from its metadata as a key
            doc_id = doc.metadata["id"]

            # Initialize a score for new documents
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'score': 0, 'doc': doc}

            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_id]['score'] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order
    reranked_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)

    # Limit the results to the top k
    top_k_results = reranked_results[:top_k]

    # Return the reranked results
    return [doc['doc'] for doc in top_k_results]


def generate_questions(question: str):
    format_instructions = output_parser_list.get_format_instructions()
    template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 

Original question: {question}

{format_instructions}
"""
    prompt_perspectives = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )

    generate_queries = (
            prompt_perspectives
            | llm
            | output_parser_list
    )

    return generate_queries.invoke({"question": question})


def answer_question(question: str, use_rag: bool = True, generate_question: bool = True):
    if generate_question and not use_rag:
        raise Exception("need to be using rag for generate_question")

    format_instructions = output_parser_response.get_format_instructions()
    template = PromptTemplate(
        template="Answer the user question as best as possible.\n{format_instructions}\n{question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    ) if not use_rag else PromptTemplate(
        template="Answer the users question as best as possible based only on the following context:"
                 "\nContext: \n {context}"
                 "\n{format_instructions}\n"
                 "\nQuestion: {question}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    simple_chain = ({"question": RunnablePassthrough()} | template | llm | output_parser_response)

    context = retriever if not generate_question else generate_questions | retriever.map() | reciprocal_rank_fusion

    rag_chain = (
            {"context": context | format_docs, "question": RunnablePassthrough()}
            | template
            | llm
            | output_parser_response
    )
    chain = rag_chain if use_rag else simple_chain
    return chain.invoke(question)


# q = "Slughorn teaches his students that Amortentia smells different to each person. What food does Harry smell?"
# a = answer_question(q, use_rag=True, generate_question=True)
# print(a)
