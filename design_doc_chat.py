import os
import fitz  # PyMuPDF
import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

# Load OpenAI API key from .env if exists

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Model and embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"

# Constants
CHROMA_DIR = "chroma_db"
DATA_DIR = "data"
EMBEDDING_MODEL = "text-embedding-3-small"

# Set page config
st.set_page_config(page_title="Design Doc Chat", layout="wide")
st.title("📄🧠 AI-Powered Vendor Datasheet Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a vendor datasheet (PDF)", type=["pdf"])

# Process uploaded file
if uploaded_file:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded and saved!")

    # Load and split document
    st.info("Parsing and embedding document...")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Embedding and storing
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    st.success("Document indexed successfully.")

    # Query input
    question = st.text_input("Ask a question about this datasheet:", placeholder="e.g. What is the design pressure?")

    if question:
        # RetrievalQA chain
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        with st.spinner("Searching and thinking..."):
            result = qa_chain({"query": question})
            st.subheader("🧠 Answer")
            st.write(result["result"])

            with st.expander("📎 Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page {doc.metadata.get('page', '?')}**: {doc.page_content[:500]}...")



