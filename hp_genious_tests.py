import threading
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic.v1 import parse_obj_as

from tqdm import tqdm
from hp_genious_create_db import create_db
from hp_genious_request import answer_question, dict_to_responseHP
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import csv

questions_answers = [
    ("What does Harry accidentally do when he goes to the zoo?", "Make the glass in the snake enclosure disappear"),
    ("What are the names of Severus Snape's parents?", "Tobias Snape (father) and Eileen Snape (Prince) (mother)"),
    (
        "What is the name of the Quidditch move where a seeker fake's seeing the snitch and dives to the ground but "
        "pulls"
        "out of the dive just in time, but the opposing seeker plummets to the ground?",
        "Wronsky Feint"),
    ("What is the first-ever password to Gryffindor Tower?", "Caput Draconis"),
    (
        "Why did Cormac McLaggen miss the Quidditch tryouts in the year previous to Harry Potter and the Half-Blood "
        "Prince?",
        "He ate a pound of doxy eggs for a bet."),
    ("What magical plant does Harry use to breathe underwater during the second task of the Triwizard Tournament?",
     "Gillyweed"),
    ("Who is the ghost of Ravenclaw Tower, also known as 'The Grey Lady'?", "Helena Ravenclaw"),
    ("What is the core of Harry Potter's wand?", "Phoenix feather"),
    ("Which potion allows the drinker to assume the form of someone else?", "Polyjuice Potion"),
    ("What is the name of the house-elf bound to serve the Black family?", "Kreacher"),
    ("Who destroys the final Horcrux, Nagini?", "Neville Longbottom"),
    ("What is the full name of the spell used to disarm another wizard?", "Expelliarmus"),
    ("What is Dumbledore's full name?", "Albus Percival Wulfric Brian Dumbledore"),
    ("Who is the Half-Blood Prince?", "Severus Snape"),
    ("What creature is Aragog?", "Acromantula"),
    ("What does the spell 'Lumos' do?", "Produces light from the wand tip"),
    ("Who was the Defense Against the Dark Arts teacher in Harry Potter's first year?", "Professor Quirrell"),
    ("What is the name of Ron Weasley's rat?",
     "The name of Ron Weasley's rat is Peter Pettigrew, but he was disguised as a rat named Scabbers."),
    ("What position does Harry Potter play on his Quidditch team?", "Seeker"),
    ("What is the name of the object used by Dumbledore to review memories?", "Pensieve"),
    ("What is the title of the book Hermione gives to Harry before his first ever Quidditch match?",
     "Quidditch Through the Ages"),
    ("Name the spell that conjures a Patronus.", "Expecto Patronum"),
    ("What is the model of the first broom Harry ever receives?", "Nimbus 2000"),
    ("What is the name of the goblin who helps Harry, Ron, and Hermione break into Bellatrix Lestrange's vault?",
     "Griphook"),
    ("What does S.P.E.W. stand for?", "Society for the Promotion of Elfish Welfare"),
    ("Who is the author of 'The Life and Lies of Albus Dumbledore'?", "Rita Skeeter"),
    ("What is the name of the potion that restores a person who has been petrified?", "Mandrake Restorative Draught"),
    ("What is the password to the Prefects' Bathroom in Harry Potter and the Goblet of Fire?", "Pine Fresh"),
    ("What spell does Harry use to save himself and Dumbledore from the Inferi in the cave?",
     "Fire-making Spell (Incendio)"),
    ("What is the name of the spell used to open locks?", "Alohomora"),
    ("What is the only book in the Hogwarts library that contains information on Horcruxes, by name?",
     "Secrets of the Darkest Art"),
    ("Who was the original owner of the Elder Wand before it passed to Dumbledore?", "Antioch Peverell"),
    ("In 'Harry Potter and the Chamber of Secrets', which spell does Harry use to kill the Basilisk?",
     "Godric Gryffindor's Sword (not a spell, but the means used to kill the Basilisk)"),
    ("What is the full name of the potion that grants the drinker luck for a period of time?", "Felix Felicis"),
    ("What spell is used to erase someone's memories?", "Obliviate"),
    ("Which magical creature is known to guard the Chamber of Secrets?", "Basilisk"),
    ("What are the names of the Deathly Hallows?", "The Elder Wand, The Resurrection Stone, The Cloak of Invisibility"),
    ("What language is commonly spoken by goblins?", "Gobbledegook"),
    ("Who originally created the Marauder's Map?", "James Potter, Sirius Black, Remus Lupin, and Peter Pettigrew"),
    ("What is the name of the magical contract that enforces the rules of the Triwizard Tournament?",
     "The Triwizard Tournament Binding Magical Contract"),
    ("What magical item is used to destroy Horcruxes that is not a sword or wand?", "The Basilisk Fang"),
    ("Name the creature that can transform into a person's worst fear.", "Boggart"),
    ("What is the name of the curse that causes unbearable pain?", "Cruciatus Curse"),
    ("What was the name of the group formed by Dumbledore to fight against Voldemort?", "The Order of the Phoenix"),
    ("What is the antidote for most poisons?", "Bezoar"),
    ("What is the maximum sentence for breaking the International Statute of Wizarding Secrecy?",
     "Azkaban imprisonment (without specifying a maximum, but implies severe punishment)"),
    (
        "Name the wizard who is known for having a chocolate frog card dedicated to him for his work with the "
        "Philosopher's Stone.",
        "Nicolas Flamel"),
    ("In which book and chapter does Harry first discover the Mirror of Erised, and what does he see in it?",
     "Harry Potter and the Philosopher's Stone, Chapter 12 'The Mirror of Erised'. Harry sees his deceased parents"),
    (
        "What is the exact location of the entrance to the Ministry of Magic that Harry and Mr. Weasley use in 'Harry "
        "Potter and the Order of the Phoenix'? How do they enter?",
        "They enter through a telephone booth located on the corner of a street in London. The visitor's entrance "
        "code is 62442, spelling out 'MAGIC'."),
    ("What specific ingredient is needed to make the Draught of Living Death?", "Powdered root of asphodel")
]

llm = Ollama(model="mistral")
output_parser = StrOutputParser()


class ResultContainer:
    def __init__(self):
        self.result = None
        self.exception = None


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
- 0 points for a wrong answer or if an element is wrong in the answer.
- 1 point if the answer states that no answer can be given.
- 2 points if the answer is incomplete but everything in the answer is correct.
- 3 points for a correct, complete answer


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


def run_function_with_timeout(func, timeout_seconds=120, *args, **kwargs):
    container = ResultContainer()

    def target():
        try:
            container.result = func(*args, **kwargs)
            container.exception = None
        except Exception as e:
            container.exception = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        thread.join()
        container.exception = TimeoutError("The function took too long to complete")

    if container.exception:
        raise container.exception
    return container.result


def process_and_grade_questions(questions_answers, output_file='hp_questions_and_answers_scores.csv', retries=3,
                                timeout=120):
    headers = ['Question', 'Correct Answer', 'Simple Answer', 'Simple Answer Score', 'Simple Answer Justification',
               'RAG Answer', 'RAG Answer Score', 'RAG Answer Justification', 'Multi-Question Answer',
               'Multi-Question Score', 'Multi-Question Justification']
    total_scores = {'Simple Answer Score': 0, 'RAG Answer Score': 0, 'Multi-Question Score': 0}
    total_questions = len(questions_answers)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for question, correct_answer in tqdm(questions_answers, desc="Processing and grading questions"):
            row = [question, correct_answer]
            for idx, (use_rag, generate_question) in enumerate([(False, False), (True, False), (True, True)]):
                for attempt in range(retries):
                    try:
                        answer = run_function_with_timeout(answer_question, timeout, question=question, use_rag=use_rag,
                                                           generate_question=generate_question)
                        answer_text = dict_to_responseHP(answer)
                        correction = run_function_with_timeout(get_correction_score, timeout, question, correct_answer,
                                                               answer_text.answer)
                        correction_response = dict_to_response_correction(correction)
                        row.extend([answer_text.answer, correction_response.points, correction_response.justification])
                        total_scores[headers[
                            3 + 3 * idx]] += correction_response.points  # Update total score for this answer type
                        break  # Break the retry loop on success
                    except Exception as e:
                        if attempt == retries - 1:
                            print(f"Failed to process '{question}' after {retries} attempts due to: {e}")
                            row.extend(["Error", 0, "Failed to obtain answer and score."])

            writer.writerow(row)

        # After processing all questions, calculate and write % scores
        writer.writerow([])  # Add empty row before summary
        percentage_scores = {key: (value / (total_questions * 3)) * 100 for key, value in total_scores.items()}
        for key, value in percentage_scores.items():
            writer.writerow([f"{key} % Score", f"{value:.2f}%"])


create_db()
process_and_grade_questions(questions_answers)
