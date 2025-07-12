
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
import uuid
import os
from dotenv import load_dotenv,find_dotenv


# --- Load Environment Variables ---
load_dotenv()
load_dotenv(find_dotenv())  # Make sure it finds the .env file correctly
print("ENV path used:", find_dotenv())  # <- Add this
print("Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
print("Loaded key:", api_key) 

# --- Embeddings Model ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Streamlit UI Setup ---
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDF(s) and chat with their content")

if not api_key:
    st.warning("GROQ_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

# query_params = st.experimental_get_query_params()
# if "session" in query_params:
#     session_id = query_params["session"][0]
# else:
#     session_id = str(uuid.uuid4())
#     st.experimental_set_query_params(session=session_id)


# st.caption(f"Session ID: `{session_id}`")

# if st.button("➕ Start New Session"):
#     new_session_id = str(uuid.uuid4())
#     st.experimental_set_query_params(session=new_session_id)
#     st.rerun()

query_params = st.query_params

# Read or generate session ID
if "session" in query_params:
    session_id = query_params["session"][0]
else:
    session_id = str(uuid.uuid4())
    st.query_params = {"session": [session_id]}  # Set in browser URL

st.caption(f"Session ID: `{session_id}`")

# New session button
if st.button("➕ Start New Session"):
    new_session_id = str(uuid.uuid4())
    st.query_params = {"session": [new_session_id]}  # Update browser URL
    st.rerun()  # Re-run with new session

# --- In-Memory Chat History Store ---
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str):
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# --- File Upload ---
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# --- Vectorstore ---
vectorstore = Chroma(collection_name="pdf_chunks", embedding_function=embeddings)

# --- Process PDFs ---
import tempfile
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_file_path = tmp.name
        # file_stream = BytesIO(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(docs)

        vectorstore.add_documents(splits)

    st.success("PDFs processed and embedded successfully.")

    retriever = vectorstore.as_retriever()

    # --- Contextual Question Reformulation ---
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- Answering Chain ---
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # --- User Interaction ---
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("Assistant:", response["answer"])
        st.write("Chat History:", session_history.messages)


# Session-based history using session_id and ChatMessageHistory — only in memory (RAM).

# It is not saved to a database — so once the session ends or the app restarts, the history is lost.

#This is a local web app built using Streamlit.

# future : To convert your Streamlit app into a deployable API