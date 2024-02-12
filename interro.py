from hp_genious_request import ask_with_rag, create_chat_messages


q1 = "Slughorn teaches his students that Amortentia smells different to each person. What food does Harry smell?"
a1="treacle tart, the woody scent of broomstick handle, and 'something flowery that he thought he might have smelled at the Burrow'"
q2 = "What are the names of Severus Snape's parents?"
a2="Tobias Snape (father) and Eileen Snape (Prince) (mother)"
q3 = "What is the name of the Quidditch move where a seeker fake's seeing the snitch and dives to the ground but pulls out of the dive just in time, but the opposing seeker plummets to the ground?"
a3="Wronsky Feint"
q4="What is the first-ever password to Gryffindor Tower?"
a4="Caput Draconis"
q5= "Why did Cormac McLaggen miss the Quidditch tryouts in the year previous to Harry Potter and the Half-Blood Prince?"
a5="He ate a pound of doxy eggs for a bet."
q6 = "What magical plant does Harry use to breathe underwater during the second task of the Triwizard Tournament?"
a6 = "Gillyweed"
q7 = "Who is the ghost of Ravenclaw Tower, also known as 'The Grey Lady'?"
a7 = "Helena Ravenclaw"
q8 = "What is the core of Harry Potter's wand?"
a8 = "Phoenix feather"
q9 = "Which potion allows the drinker to assume the form of someone else?"
a9 = "Polyjuice Potion"
q10 = "What is the name of the house-elf bound to serve the Black family?"
a10 = "Kreacher"
q11 = "Who destroys the final Horcrux, Nagini?"
a11 = "Neville Longbottom"
q12 = "What is the full name of the spell used to disarm another wizard?"
a12 = "Expelliarmus"
q13 = "What is Dumbledore's full name?"
a13 = "Albus Percival Wulfric Brian Dumbledore"
q14 = "Who is the Half-Blood Prince?"
a14 = "Severus Snape"
q15 = "What creature is Aragog?"
a15 = "Acromantula"
q16 = "What does the spell 'Lumos' do?"
a16 = "Produces light from the wand tip"
q17 = "Who was the Defense Against the Dark Arts teacher in Harry Potter's first year?"
a17 = "Professor Quirrell"
q18 = "What is the name of Ron Weasley's rat?"
a18 = "Scabbers"
q19 = "What position does Harry Potter play on his Quidditch team?"
a19 = "Seeker"
q20 = "What is the name of the object used by Dumbledore to review memories?"
a20 = "Pensieve"

questions = [
    q1, q2, q3, q4, q5,
    q6, q7, q8, q9, q10,
    q11, q12, q13, q14, q15,
    q16, q17, q18, q19, q20
]

answers = [
    a1, a2, a3, a4, a5,
    a6, a7, a8, a9, a10,
    a11, a12, a13, a14, a15,
    a16, a17, a18, a19, a20
]

with open('test_results.txt', 'w') as file:
    for i, q in enumerate(questions):
        messages = create_chat_messages(q)  
        response = ask_with_rag(messages)
        
        # Construct the result string
        result = f"=====================\n"
        result += f"Question {i+1}: {q}\n"
        result += f"Answer RAG: {response}\n"
        result += f"Right Answer: {answers[i]}\n"
        
        # Write the result to the file
        file.write(result)