import gradio as gr

from hp_genious_request import answer_question, dict_to_responseHP, answer_question_decomposition


# Gradio interface setup
def gradio_answer_question(question, use_rag, generate_question,decompose):
    use_rag_bool = use_rag == "Yes"
    generate_question_bool = generate_question == "Yes"
    decompose_bool = decompose == "Yes"
    if not decompose_bool:
        answer_dict = answer_question(question, use_rag=use_rag_bool, generate_question=generate_question_bool)
    else:
        answer_dict = answer_question_decomposition(question,generate_question=generate_question_bool)
    answer = dict_to_responseHP(answer_dict)
    return answer.answer, answer.source


with gr.Blocks() as app:
    gr.Markdown("# HP Genius")
    with gr.Row():
        question = gr.Textbox(label="Enter your question")
    with gr.Row():
        use_rag = gr.Radio(choices=["Yes", "No"], label="Use RAG", value="Yes")
        generate_question = gr.Radio(choices=["Yes", "No"], label="Generate Question Variants", value="Yes")
        decompose = gr.Radio(choices=["Yes", "No"], label="Decompose question", value="No")
    answer_button = gr.Button("Get Answer")
    answer = gr.Textbox(label="Answer", lines=4)
    source = gr.Textbox(label="Source", lines=2)

    answer_button.click(
        fn=gradio_answer_question,
        inputs=[question, use_rag, generate_question,decompose],
        outputs=[answer, source]
    )

if __name__ == "__main__":
    app.launch()
