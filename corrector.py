import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = Ollama(model="mistral")
output_parser = StrOutputParser()

def get_correction_score(question, correct_answer, given_answer):
    template= """You are an AI language model assistant. Your task is to grade the answers to questions.
You will each time receive a question, the correct answer to the question and a given anwser. 
Grade the answers with a percentage. A wrong answer should be worst than not responding or saying "I don't know".
Question: {question}
Correct answer: {correct_answer}
Given anwser: {given_answer}
respond with just a number
    """
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    grade = (
        prompt_perspectives 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    result = grade.invoke({"question": question,"correct_answer": correct_answer, "given_answer": given_answer})
    return result

# Read the CSV file into a DataFrame
df = pd.read_csv('/hp_questions_and_answers.csv')  # Replace with the actual path to your CSV file

# Iterate over each row and calculate scores
for index, row in df.iterrows():
    correct_answer = row['Correct Answer']
    question = row['Question']
    for column in ['Simple Answer', 'RAG Answer', 'Multi-Question Answer']:
        given_answer = row[column]
        score = get_correction_score(question, correct_answer, given_answer)
        df.at[index, f'{column} Score'] = score

# Export the updated DataFrame with scores back to a new CSV file
df.to_csv('/path/to/your/scores_csv.csv', index=False)