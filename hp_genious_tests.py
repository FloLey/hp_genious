import threading
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic.v1 import parse_obj_as

from tqdm import tqdm
from hp_genious_create_db import create_db
from hp_genious_request import answer_question
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import pandas as pd
import csv

llm = Ollama(model="mistral")
output_parser = StrOutputParser()


class ResultContainer:
    def __init__(self):
        self.result = None
        self.exception = None


questions = [
    "What does Harry accidentally do when he goes to the zoo?",
    "What are the names of Severus Snape's parents?",
    "What is the name of the Quidditch move where a seeker fake's seeing the snitch and dives to the ground but pulls out of the dive just in time, but the opposing seeker plummets to the ground?",
    "What is the first-ever password to Gryffindor Tower?",
    "Why did Cormac McLaggen miss the Quidditch tryouts in the year previous to Harry Potter and the Half-Blood Prince?",
    "What magical plant does Harry use to breathe underwater during the second task of the Triwizard Tournament?",
    "Who is the ghost of Ravenclaw Tower, also known as 'The Grey Lady'?",
    "What is the core of Harry Potter's wand?",
    "Which potion allows the drinker to assume the form of someone else?",
    "What is the name of the house-elf bound to serve the Black family?",
    "Who destroys the final Horcrux, Nagini?",
    "What is the full name of the spell used to disarm another wizard?",
    "What is Dumbledore's full name?",
    "Who is the Half-Blood Prince?",
    "What creature is Aragog?",
    "What does the spell 'Lumos' do?",
    "Who was the Defense Against the Dark Arts teacher in Harry Potter's first year?",
    "What is the name of Ron Weasley's rat?",
    "What position does Harry Potter play on his Quidditch team?",
    "What is the name of the object used by Dumbledore to review memories?",
    "What is the title of the book Hermione gives to Harry before his first ever Quidditch match?",
    "Name the spell that conjures a Patronus.",
    "What is the model of the first broom Harry ever receives?",
    "What is the name of the goblin who helps Harry, Ron, and Hermione break into Bellatrix Lestrange's vault?",
    "What does S.P.E.W. stand for?",
    "Who is the author of 'The Life and Lies of Albus Dumbledore'?",
    "What is the name of the potion that restores a person who has been petrified?",
    "What is the password to the Prefects' Bathroom in Harry Potter and the Goblet of Fire?",
    "What spell does Harry use to save himself and Dumbledore from the Inferi in the cave?",
    "What is the name of the spell used to open locks?",
    "What is the only book in the Hogwarts library that contains information on Horcruxes, by name?",
    "Who was the original owner of the Elder Wand before it passed to Dumbledore?",
    "In 'Harry Potter and the Chamber of Secrets', which spell does Harry use to kill the Basilisk?",
    "What is the full name of the potion that grants the drinker luck for a period of time?",
    "What spell is used to erase someone's memories?",
    "Which magical creature is known to guard the Chamber of Secrets?",
    "What are the names of the Deathly Hallows?",
    "What language is commonly spoken by goblins?",
    "Who originally created the Marauder's Map?",
    "What is the name of the magical contract that enforces the rules of the Triwizard Tournament?",
    "What magical item is used to destroy Horcruxes that is not a sword or wand?",
    "Name the creature that can transform into a person's worst fear.",
    "What is the name of the curse that causes unbearable pain?",
    "What was the name of the group formed by Dumbledore to fight against Voldemort?",
    "What is the antidote for most poisons?",
    "What is the maximum sentence for breaking the International Statute of Wizarding Secrecy?",
    "Name the wizard who is known for having a chocolate frog card dedicated to him for his work with the Philosopher's Stone.",
    "In which book and chapter does Harry first discover the Mirror of Erised, and what does he see in it",
    "What is the exact location of the entrance to the Ministry of Magic that Harry and Mr. Weasley use in 'Harry Potter and the Order of the Phoenix'? How do they enter?",
    "What specific ingredient is needed to make the Draught of Living Death?"
]

answers = [
    "Make the glass in the snake enclosure disappear",
    "Tobias Snape (father) and Eileen Snape (Prince) (mother)",
    "Wronsky Feint",
    "Caput Draconis",
    "He ate a pound of doxy eggs for a bet.",
    "Gillyweed",
    "Helena Ravenclaw",
    "Phoenix feather",
    "Polyjuice Potion",
    "Kreacher",
    "Neville Longbottom",
    "Expelliarmus",
    "Albus Percival Wulfric Brian Dumbledore",
    "Severus Snape",
    "Acromantula",
    "Produces light from the wand tip",
    "Professor Quirrell",
    "The name of Ron Weasley's rat is Peter Pettigrew, but he was disguised as a rat named Scabbers.",
    "Seeker",
    "Pensieve",
    "Quidditch Through the Ages",
    "Expecto Patronum",
    "Nimbus 2000",
    "Griphook",
    "Society for the Promotion of Elfish Welfare",
    "Rita Skeeter",
    "Mandrake Restorative Draught",
    "Pine Fresh",
    "Fire-making Spell (Incendio)",
    "Alohomora",
    "Secrets of the Darkest Art",
    "Antioch Peverell",
    "Godric Gryffindor's Sword (not a spell, but the means used to kill the Basilisk)",
    "Felix Felicis",
    "Obliviate",
    "Basilisk",
    "The Elder Wand, The Resurrection Stone, The Cloak of Invisibility",
    "Gobbledegook",
    "James Potter, Sirius Black, Remus Lupin, and Peter Pettigrew",
    "The Triwizard Tournament Binding Magical Contract",
    "The Basilisk Fang",
    "Boggart",
    "Cruciatus Curse",
    "The Order of the Phoenix",
    "Bezoar",
    "Azkaban imprisonment (without specifying a maximum, but implies severe punishment)",
    "Nicolas Flamel",
    "Harry Potter and the Philosopher's Stone, Chapter 12 'The Mirror of Erised'. Harry sees his deceased parents",
    "They enter through a telephone booth located on the corner of a street in London. The visitor's entrance code is 62442, spelling out 'MAGIC'.",
    "Powdered root of asphodel",
]


class responseCorrection(BaseModel):
    points: int = Field(description="The number of points for this question")
    justification: str = Field(description="a justification for the given point")

def dict_to_response_correction(correction_dict):
    return parse_obj_as(responseCorrection, correction_dict)

output_parser_correction = JsonOutputParser(pydantic_object=responseCorrection)


def get_correction_score(question, correct_answer, given_answer):
    format_instructions = output_parser_correction.get_format_instructions()

    template = PromptTemplate(
        template="""You are an AI language model assistant. Your task is to grade the answers to questions.
You will each time receive a question, the correct answer to the question and a given answer. 
give: 
- 0 points for a wrong answer
- 1 point if the answer states that no answer can be given 
- 2 points if the answer is incomplete.
- 3 points for a good answer


{format_instructions}


Question: {question}
Correct answer: {correct_answer}
Given answer: {given_answer}
score: 


    """,
        input_variables=["question", "correct_answer", "given_answer"],
        partial_variables={"format_instructions": format_instructions},
    )

    grade = (
            template
            | llm
            | output_parser_correction
    )
    result = grade.invoke({"question": question, "correct_answer": correct_answer, "given_answer": given_answer})
    return result


def answer_question_with_timeout(question, use_rag, generate_question, timeout_seconds=60, retries=3):
    # Function to execute the target function with timeout
    def run_with_timeout():
        try:
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout_seconds)
            if thread.is_alive():
                # If the thread is still alive, it means it's stuck and didn't finish in time
                raise TimeoutError("The function took too long to complete")
        except Exception as e:
            container.exception = e

    # Target function to be executed in a thread
    def target():
        try:
            # Attempt to get the answer to the question
            container.result = answer_question(question=question, use_rag=use_rag, generate_question=generate_question)
            container.exception = None  # Reset exception if successful
        except Exception as e:
            container.exception = e

    # Initial setup before retry loop
    container = ResultContainer()
    attempt = 0

    # Retry loop
    while attempt < retries:
        run_with_timeout()  # Attempt to run the target function with timeout
        if container.exception is None:
            return container.result  # Return result if successful
        else:
            attempt += 1  # Increment attempt counter if there was an exception

    # If we reach here, all retries have failed
    if container.exception:
        raise container.exception  # Raise the last exception if all retries failed


def answer_questions(output_file: str = 'hp_questions_and_answers.csv'):
    results = []
    max_retries = 3

    # Iterate over each question
    for question, correct_answer in tqdm(zip(questions, answers), desc="Answering questions"):
        for attempt in range(max_retries):
            try:
                simple_answer = answer_question_with_timeout(question=question, use_rag=False, generate_question=False)
                rag_answer = answer_question_with_timeout(question=question, use_rag=True, generate_question=False)
                multi_question_answer = answer_question_with_timeout(question=question, use_rag=True,
                                                                     generate_question=True)
                results.append([question, correct_answer, simple_answer, rag_answer, multi_question_answer])
                break  # Break the retry loop on success
            except Exception as e:
                if attempt == max_retries - 1:  # Last retry
                    print(f"Failed to get an answer for '{question}' after {max_retries} attempts due to timeout.")
                    return  # Exit the function on repeated failures

    csv_file_name = output_file
    headers = ['Question', 'Correct Answer', 'Simple Answer', 'RAG Answer', 'Multi-Question Answer']

    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the headers
        writer.writerows(results)  # Write the data rows


def get_correction_score_with_timeout(question, correct_answer, given_answer, timeout_seconds=60):
    # Define a container to hold the function's result or a default error response
    container = [responseCorrection(points=0, justification="Initialization")]  # Use a list for mutability

    def target(con):
        try:
            con[0] = get_correction_score(question, correct_answer, given_answer)
        except Exception as e:
            con[0] = responseCorrection(points=0, justification=f"Error: {str(e)}")

    thread = threading.Thread(target=target, args=(container,))
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        thread.join()
        container[0] = responseCorrection(points=0, justification="Timeout: The function took too long to complete")

    return container[0]


def grade_answers(input_file='hp_questions_and_answers.csv', output_file='hp_questions_and_answers_scores.csv',
                  retries=3, timeout=60):
    df = pd.read_csv(input_file)

    answer_columns = ['Simple Answer', 'RAG Answer', 'Multi-Question Answer']
    total_scores = {column: 0 for column in answer_columns}

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Grading questions"):
        correct_answer = row['Correct Answer']
        question = row['Question']

        for column in answer_columns:
            correction_dict = responseCorrection(points=0, justification="Failed to obtain correction.")
            for attempt in range(retries):
                try:
                    correction_dict = get_correction_score_with_timeout(question, correct_answer, row[column],
                                                                   timeout_seconds=timeout)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt == retries - 1:
                        # Assign a default correction on the final attempt if all retries fail
                        correction_dict = responseCorrection(points=0, justification=f"Failed after {retries} retries: {e}")

            correction = dict_to_response_correction(correction_dict)
            total_scores[column] += correction.points

            # Update DataFrame
            points_col_name = f'{column} Points'
            justification_col_name = f'{column} Justification'
            if points_col_name not in df.columns:
                df[points_col_name] = 0  # Initialize column with 0
            if justification_col_name not in df.columns:
                df[justification_col_name] = ""  # Initialize column with empty string

            df.at[index, points_col_name] = correction.points
            df.at[index, justification_col_name] = correction.justification

    # Calculate and insert overall scores
    entries_count = df.shape[0]
    for column, total_score in total_scores.items():
        overall_score = (total_score / entries_count) * 100
        df[f'{column} Overall Score'] = overall_score

    df.to_csv(output_file, index=False)


# create_db()
answer_questions()
grade_answers()
