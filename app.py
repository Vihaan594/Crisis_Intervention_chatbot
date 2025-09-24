import streamlit as st
import os
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db_connection():
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        database=os.environ['PG_DB'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        cursor_factory=RealDictCursor
    )
    return conn
def hash_password(password):
    return pwd_context.hash(password)
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)
def get_user_by_username(username):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users_1 WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user
def create_user(username, name, password, email=None):
    hashed_pw = hash_password(password)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, name, hashed_password, email) VALUES (%s, %s, %s, %s)",
        (username, name, hashed_pw, email)
    )
    conn.commit()
    cur.close()
    conn.close()

def login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        user = get_user_by_username(username)
        if user and verify_password(password, user['hashed_password']):
            st.session_state['username'] = user['username']
            st.session_state['name'] = user['name']
            st.session_state['authenticated'] = True
            st.success(f"Welcome back, {user['name']}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def signup():
    st.subheader("Sign Up")
    username = st.text_input("Choose a username", key="signup_username")
    name = st.text_input("Your name", key="signup_name")
    email = st.text_input("Email (optional)", key="signup_email")
    password = st.text_input("Choose a password", type="password", key="signup_password")
    password_confirm = st.text_input("Confirm password", type="password", key="signup_password_confirm")
    if st.button("Sign Up"):
        if password != password_confirm:
            st.error("Passwords do not match")
            return
        if get_user_by_username(username):
            st.error("Username already exists")
            return
        create_user(username, name, password, email)
        st.success("User  created! Please login.")

def main_app():
    groq_api_key = os.environ['GROQ_API_KEY']
    if "initialized" not in st.session_state:
        with st.spinner("Loading documents and setting up vector stores..."):
            web_docs = WebBaseLoader("https://docs.smith.langchain.com/").load()
            pdf_docs = PyPDFLoader("resume.pdf").load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            web_chunks = splitter.split_documents(web_docs)
            pdf_chunks = splitter.split_documents(pdf_docs)
            embeddings = OllamaEmbeddings(model="llama2")
            web_vec = FAISS.from_documents(web_chunks, embeddings)
            pdf_vec = FAISS.from_documents(pdf_chunks, embeddings)
            st.session_state.update({
                "web_vec": web_vec,
                "pdf_vec": pdf_vec,
                "initialized": True,
                "chat_history": []
            })
    web_retriever = st.session_state["web_vec"].as_retriever(search_kwargs={"k": 3})
    pdf_retriever = st.session_state["pdf_vec"].as_retriever(search_kwargs={"k": 3})
    doc_prompt = PromptTemplate.from_template("Context:\n{page_content}\n\n(from: {source})")
    react_prompt = hub.pull("hwchase17/react")
    tools = [
        create_retriever_tool(
            retriever=web_retriever,
            name="LangChain_Smith_docs",
            description="Search LangChain documentation page",
            document_prompt=doc_prompt
        ),
        create_retriever_tool(
            retriever=pdf_retriever,
            name="Local_MentalHealth_Guide",
            description="Search local PDF for mental-health guidelines, crises, and coping tips",
            document_prompt=doc_prompt
        )
    ]
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )
    agent_runnable = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )
    agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)
    st.set_page_config(page_title="Mental Health Agent", page_icon="🧠")
    st.title("🧠 Tool‑Aware Mental Health Chatbot")
    st.markdown("Ask anything related to mental health, LangChain docs, or related topics.")

    user_input = st.chat_input("Ask your question…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                start = time.process_time()
                result = agent.invoke({"input": user_input})
                response_time = time.process_time() - start
                answer = result.get("output") if isinstance(result, dict) else result
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "text": answer})
                tool_calls = result.get("tool_calls", [])
                if tool_calls:
                    used = ", ".join(tc["name"] for tc in tool_calls)
                    st.caption(f"Tool(s) used: **{used}** · Completed in {response_time:.2f}s")
    
    with st.sidebar:
        st.header("🔧 Status")
        st.markdown("- Web docs loaded ✅")
        st.markdown("- PDF loaded ✅")
        st.markdown("- Embeddings: `llama2`")
        st.markdown("- Model: `llama-4-scout-17b` via **Groq**")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()
    if st.session_state.chat_history:
        with st.expander("💬 Conversation History"):
            for msg in st.session_state.chat_history:
                st.markdown(f"**{msg['role'].capitalize()}:** {msg['text']}")
    
    st.markdown("---")
    st.subheader("🌟 What our users say")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("💬 *“This bot helped me navigate my anxiety better than any app I’ve tried.”*")
        st.caption("— Anjali, College Student")
    with col2:
        st.markdown("💬 *“Love how it can pull from mental health guides and answer questions instantly.”*")
        st.caption("— Rakesh, Counselor")
    with col3:
        st.markdown("💬 *“Having LangChain docs and local PDFs in one chat is genius.”*")
        st.caption("— Dev Sharma, AI Researcher")


if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if not st.session_state['authenticated']:
    option = st.radio("Choose an option", ["Login", "Sign Up"])
    if option == "Login":
        login()
    else:
        signup()
    st.stop()
else:
    main_app()





