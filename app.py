import os
import streamlit as st
from src.retrieval.retriever import Retriever
from config.config import Config
from src.llm.chat_model import ChatModel
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()  # For local development with .env file
except:
    pass  # If dotenv is not available

try:
    if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
        os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
        os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
except:
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="medibot",
    page_icon="images\icon.png",
    layout="wide"
)

st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src='https://www.reshot.com/preview-assets/icons/3J5PF7MQWY/medical-3J5PF7MQWY.svg' width='40' style='margin-right:10px'/>
        Medical Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

def load_database():
    config = Config()
    embedding = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=config.INDEX_NAME,
        embedding=embedding
        )
    return vector_store

def retrive_context_output(query):
    llm = ChatModel()
    vector_store = load_database()
    retriever = Retriever(vector_store=vector_store)
    context = retriever.retriever_context(query=query)
    answer, _ = llm.generate_answer(query=query, context=context)
    return answer

input = st.text_input(
        "Ask questions about any medical topic — symptoms, conditions, or cures!",
        placeholder="Type your query",
        help="Don't leave this empty"
    )

if st.button("Get Answer", type="secondary"):
     if input:
        with st.spinner("Generating answer..."):
            output = retrive_context_output(input)
            st.markdown(output.content)

# # Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain")