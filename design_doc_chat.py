from dotenv import load_dotenv
import os
import streamlit as st
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file in the root directory.")

# Set API key for langchain-openai compatibility
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define constants
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"
PERSIST_DIRECTORY = "db"
CHROMA_COLLECTION_NAME = "design_doc_collection"

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL)

# Initialize Streamlit UI
st.set_page_config(page_title="Design Document Intelligence")
st.title("📄 Design Document Intelligence Chat")

uploaded_file = st.file_uploader("Upload your PDF design document", type="pdf")
query = st.text_input("Ask a question about the document:")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split document
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create vectorstore
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=CHROMA_COLLECTION_NAME
    )

    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    if query:
        with st.spinner("Searching your document..."):
            result = qa_chain.run(query)
            st.write("**Answer:**", result)
