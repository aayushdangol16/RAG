from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine

con = create_engine("sqlite:///chat_history.db")

model=OllamaLLM(model="llama3.2")

qa_system_prompt = """You are Alex, a personal AI assistant. 
Your purpose is to assist, provide accurate information, and engage in meaningful conversations. 
Always adapt your tone to be friendly, empathetic, and respectful. 
Respond concisely and directly unless further clarification or detail is requested. 
Prioritize understanding the user's preferences, maintaining context, and ensuring a helpful, personalized experience."""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

rag_chain=qa_prompt|model

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection=con
    ),
    input_messages_key="input",
    history_messages_key="chat_history",
)

def chat(query,session_id):
    result=conversational_rag_chain.invoke({"input": query},config={"configurable": {"session_id": session_id}},)
    return result
