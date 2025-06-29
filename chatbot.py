from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()

LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="RAG"

file="file.pdf"
loader=PyPDFLoader(file)
docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,add_start_index=True)
all_splits=text_splitter.split_documents(docs)

embeddings=OllamaEmbeddings(model="llama3.2")
vectorestore=Chroma.from_documents(documents=all_splits,embedding=embeddings)

model=OllamaLLM(model="llama3.2")

bm25_retriever = BM25Retriever.from_documents(all_splits,k=2)
retriever=vectorestore.as_retriever(search_type="similarity",k=2)
hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],weights=[0.5, 0.5])

contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    model, hybrid_retriever, contextualize_q_prompt
)

qa_system_prompt = """You are provided with context below to help answer the question at the end.
If you do not know the answer, simply say "I don't know." Do not attempt to fabricate an answer.
Your response should be brief, no more than three sentences.
Always conclude with "Thanks for asking!" at the end of your answer.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def chat(query):
    result=conversational_rag_chain.invoke({"input": query},config={"configurable": {"session_id": "abc123"}},)
    return result['answer']