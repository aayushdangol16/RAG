import gradio as gr
import time
from personal_chatbot import chat

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    submit = gr.Button("Submit") 

    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list):
        user_message = history[-1]['content']
        bot_message = chat(user_message)
        history.append({"role": "assistant", "content": ""})
        for character in bot_message:
            history[-1]['content'] += character
            time.sleep(0.05)
            yield history

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.launch(share=True)
