# design_doc_chat.py

import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableMap
from langchain.prompts import PromptTemplate

# Streamlit UI setup
st.set_page_config(page_title="Design Document Intelligence", layout="wide")
st.title("📄 AI-Powered Design Document Chat")

# Load API key from secrets or environment
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Model and embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4"

# File uploader
uploaded_file = st.file_uploader("Upload a design document (PDF)", type="pdf")

if uploaded_file:
    file_path = os.path.join("temp_docs", uploaded_file.name)
    os.makedirs("temp_docs", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split the document
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Embeddings and vector store
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectordb.as_retriever()

    # Prompt template
    template = """
    You are an expert in oil & gas engineering design. Use the context below to answer the question accurately and precisely.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # QA Chain setup
    llm = ChatOpenAI(model_name=CHAT_MODEL, api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    # User input for query
    query = st.text_input("Ask a question about the uploaded document:")

    if query:
        with st.spinner("Processing..."):
            answer = qa_chain.run(query)
            st.markdown(f"**Answer:** {answer}")

        # Optional: show sources
        # st.write("\n**Retrieved Chunks:**")
        # for doc in retriever.get_relevant_documents(query):
        #     st.write(doc.page_content[:300])
