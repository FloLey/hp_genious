import re
from tqdm import tqdm
from hp_genious_create_db import create_db
from hp_genious_request import answer_question
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import csv

llm = Ollama(model="mistral")
output_parser = StrOutputParser()

questions = [
    "What does Harry accidentally do when he goes to the zoo?",
    # "What are the names of Severus Snape's parents?",
    # "What is the name of the Quidditch move where a seeker fake's seeing the snitch and dives to the ground but pulls out of the dive just in time, but the opposing seeker plummets to the ground?",
    # "What is the first-ever password to Gryffindor Tower?",
    # "Why did Cormac McLaggen miss the Quidditch tryouts in the year previous to Harry Potter and the Half-Blood Prince?",
    # "What magical plant does Harry use to breathe underwater during the second task of the Triwizard Tournament?",
    # "Who is the ghost of Ravenclaw Tower, also known as 'The Grey Lady'?",
    # "What is the core of Harry Potter's wand?",
    # "Which potion allows the drinker to assume the form of someone else?",
    # "What is the name of the house-elf bound to serve the Black family?",
    # "Who destroys the final Horcrux, Nagini?",
    # "What is the full name of the spell used to disarm another wizard?",
    # "What is Dumbledore's full name?",
    # "Who is the Half-Blood Prince?",
    # "What creature is Aragog?",
    # "What does the spell 'Lumos' do?",
    # "Who was the Defense Against the Dark Arts teacher in Harry Potter's first year?",
    # "What is the name of Ron Weasley's rat?",
    # "What position does Harry Potter play on his Quidditch team?",
    # "What is the name of the object used by Dumbledore to review memories?",
    # "What is the title of the book Hermione gives to Harry before his first ever Quidditch match?",
    # "Name the spell that conjures a Patronus.",
    # "What is the model of the first broom Harry ever receives?",
    # "What is the name of the goblin who helps Harry, Ron, and Hermione break into Bellatrix Lestrange's vault?",
    # "What does S.P.E.W. stand for?",
    # "Who is the author of 'The Life and Lies of Albus Dumbledore'?",
    # "What is the name of the potion that restores a person who has been petrified?",
    # "What is the password to the Prefects' Bathroom in Harry Potter and the Goblet of Fire?",
    # "What spell does Harry use to save himself and Dumbledore from the Inferi in the cave?",
    # "What is the name of the spell used to open locks?",
    # "What is the only book in the Hogwarts library that contains information on Horcruxes, by name?",
    # "Who was the original owner of the Elder Wand before it passed to Dumbledore?",
    # "In 'Harry Potter and the Chamber of Secrets', which spell does Harry use to kill the Basilisk?",
    # "What is the full name of the potion that grants the drinker luck for a period of time?",
    # "What spell is used to erase someone's memories?",
    # "Which magical creature is known to guard the Chamber of Secrets?",
    # "What are the names of the Deathly Hallows?",
    # "What language is commonly spoken by goblins?",
    # "Who originally created the Marauder's Map?",
    # "What is the name of the magical contract that enforces the rules of the Triwizard Tournament?",
    # "What magical item is used to destroy Horcruxes that is not a sword or wand?",
    # "Name the creature that can transform into a person's worst fear.",
    # "What is the name of the curse that causes unbearable pain?",
    # "What was the name of the group formed by Dumbledore to fight against Voldemort?",
    # "What is the antidote for most poisons?",
    # "What is the maximum sentence for breaking the International Statute of Wizarding Secrecy?",
    # "Name the wizard who is known for having a chocolate frog card dedicated to him for his work with the Philosopher's Stone.",
    # "In which book and chapter does Harry first discover the Mirror of Erised, and what does he see in it",
    # "What is the exact location of the entrance to the Ministry of Magic that Harry and Mr. Weasley use in 'Harry Potter and the Order of the Phoenix'? How do they enter?",
    # "What specific ingredient is needed to make the Draught of Living Death?"
]

answers = [
    "Make the glass in the snake enclosure disappear",
    # "Tobias Snape (father) and Eileen Snape (Prince) (mother)",
    # "Wronsky Feint",
    # "Caput Draconis",
    # "He ate a pound of doxy eggs for a bet.",
    # "Gillyweed",
    # "Helena Ravenclaw",
    # "Phoenix feather",
    # "Polyjuice Potion",
    # "Kreacher",
    # "Neville Longbottom",
    # "Expelliarmus",
    # "Albus Percival Wulfric Brian Dumbledore",
    # "Severus Snape",
    # "Acromantula",
    # "Produces light from the wand tip",
    # "Professor Quirrell",
    # "The name of Ron Weasley's rat is Peter Pettigrew, but he was disguised as a rat named Scabbers.",
    # "Seeker",
    # "Pensieve",
    # "Quidditch Through the Ages",
    # "Expecto Patronum",
    # "Nimbus 2000",
    # "Griphook",
    # "Society for the Promotion of Elfish Welfare",
    # "Rita Skeeter",
    # "Mandrake Restorative Draught",
    # "Pine Fresh",
    # "Fire-making Spell (Incendio)",
    # "Alohomora",
    # "Secrets of the Darkest Art",
    # "Antioch Peverell",
    # "Godric Gryffindor's Sword (not a spell, but the means used to kill the Basilisk)",
    # "Felix Felicis",
    # "Obliviate",
    # "Basilisk",
    # "The Elder Wand, The Resurrection Stone, The Cloak of Invisibility",
    # "Gobbledegook",
    # "James Potter, Sirius Black, Remus Lupin, and Peter Pettigrew",
    # "The Triwizard Tournament Binding Magical Contract",
    # "The Basilisk Fang",
    # "Boggart",
    # "Cruciatus Curse",
    # "The Order of the Phoenix",
    # "Bezoar",
    # "Azkaban imprisonment (without specifying a maximum, but implies severe punishment)",
    # "Nicolas Flamel",
    # "Harry Potter and the Philosopher's Stone, Chapter 12 'The Mirror of Erised'. Harry sees his deceased parents",
    # "They enter through a telephone booth located on the corner of a street in London. The visitor's entrance code is 62442, spelling out 'MAGIC'.",
    # "Powdered root of asphodel",
]


def get_correction_score(question, correct_answer, given_answer):
    template = """You are an AI language model assistant. Your task is to grade the answers to questions.
You will each time receive a question, the correct answer to the question and a given anwser. 
give 3 points for a good answer, 0 points for a wrong answer, 1 point it the answer states "I don't know" and 2 points if the answer is incomplete.


Question: What does Harry accidentally do when he goes to the zoo?
Correct answer: Make the glass in the snake enclosure disappear
Given anwser: Harry feeds Myrtle the hippopotamus excessively, causing her to become aggressive and disruptive.
score: 0


Question: {question}
Correct answer: {correct_answer}
Given anwser: {given_answer}
score: 


    """

    def extract_first_number(text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    grade = (
            prompt_perspectives
            | llm
            | StrOutputParser()
            | extract_first_number
    )
    result = grade.invoke({"question": question, "correct_answer": correct_answer, "given_answer": given_answer})
    return result


def answer_questions(output_file: str = 'hp_questions_and_answers.csv'):
    results = []

    # Iterate over each question
    for question, correct_answer in tqdm(zip(questions, answers), "question"):
        # Get responses from each function
        simple_answer = answer_question(question=question, use_rag=False, generate_question=False)
        rag_answer = answer_question(question=question, use_rag=True, generate_question=False)
        multi_question_answer = answer_question(question=question, use_rag=True, generate_question=True)

        results.append([question, correct_answer, simple_answer, rag_answer, multi_question_answer])

    csv_file_name = output_file

    headers = ['Question', 'Correct Answer', 'Simple Answer', 'RAG Answer', 'Multi-Question Answer']

    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the headers
        writer.writerows(results)  # Write the data rows


def grade_answers(input_file: str = 'hp_questions_and_answers.csv',
                  output_file: str = 'hp_questions_and_answers_scores.csv'):
    df = pd.read_csv(input_file)

    answer_columns = ['Simple Answer', 'RAG Answer', 'Multi-Question Answer']

    total_scores = {column: 0 for column in answer_columns}

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Grading questions"):
        correct_answer = row['Correct Answer']
        question = row['Question']

        for column in answer_columns:
            given_answer = row[column]
            score = get_correction_score(question, correct_answer, given_answer)
            total_scores[column] += score if score else 0  # Accumulate total scores

            col_index = df.columns.get_loc(column)
            score_col_name = f'{column} Score'

            if score_col_name not in df.columns:
                df.insert(col_index + 1, score_col_name, None)

            df.at[index, score_col_name] = score

    entries_count = df.shape[0]
    for column, total_score in total_scores.items():
        overall_score = (total_score / (entries_count * 3)) * 100
        overall_score_col_name = f'{column} Overall Score'
        df[overall_score_col_name] = overall_score

    df.to_csv(output_file, index=False)


# create_db()
answer_questions()
# grade_answers()
