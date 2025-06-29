import gradio as gr
import time
from personal_chatbot import chat

def greet_user(username):
    # Hide the first interface and show the second one with the username passed
    return f"Hello, {username}!", username, gr.update(visible=False), gr.update(visible=True)

def user(user_message, history: list):
    # Append the user's message to the chat history
    return "", history + [{"role": "user", "content": user_message}]

def bot(history: list, username):
    # Generate a bot response and append it to the chat history
    user_message = history[-1]['content']
    bot_message = chat(user_message, username)  # Pass the username to the chat function
    history.append({"role": "assistant", "content": ""})
    for character in bot_message:
        history[-1]['content'] += character
        time.sleep(0.05)  # Simulate typing effect
        yield history

with gr.Blocks() as demo:
    # First interface
    with gr.Column(visible=True) as first_interface:
        username_input = gr.Textbox(label="Enter Username", placeholder="Type your username here...")
        greet_button = gr.Button("Submit")
    greeting_output = gr.Textbox(visible=False)

    # Second interface (hidden initially)
    with gr.Column(visible=False) as second_interface:
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox()
        submit = gr.Button("Submit")

        # Add a state to store the username
        username_state = gr.State()

        # Link the submit button to handle user and bot interactions
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, username_state], chatbot
        )

    # Logic for transitioning from the first interface to the second
    greet_button.click(
        fn=greet_user,
        inputs=username_input,
        outputs=[greeting_output, username_state, first_interface, second_interface]
    )

# Launch the interface
demo.launch(share=True)
