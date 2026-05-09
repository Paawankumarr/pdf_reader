import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# =========================
# Load Environment Variables
# =========================
load_dotenv()

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Chat With PDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat With Your PDF")

st.write("Upload a PDF and ask questions from it ")


# =========================
# Upload PDF
# =========================
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)


# =========================
# Session State
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================
# Process PDF
# =========================
if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)

        documents = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Create vector store
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        st.session_state.vectorstore = vectorstore

    st.success("PDF Processed Successfully!")


# =========================
# User Question
# =========================
question = st.text_input("Ask a question from the PDF")


# =========================
# Chat Logic
# =========================
if question and st.session_state.vectorstore:

    with st.spinner("Generating Answer..."):

        # Retrieve relevant chunks
        docs = st.session_state.vectorstore.similarity_search(
            question,
            k=4
        )

        # Convert docs into context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

        # Prompt Template
        prompt = PromptTemplate(
            template="""
You are a helpful AI assistant.

Answer the question only from the provided PDF context.

If answer is not available in the context,
say "Answer not found in PDF."

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=["context", "question"]
        )

        # LCEL Chain
        chain = prompt | llm

        # Generate response
        response = chain.invoke({
            "context": context,
            "question": question
        })

        # Extract final text
        answer = response.content

        # Save chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })


# =========================
# Display Chat History
# =========================
if st.session_state.chat_history:

    st.subheader("💬 Chat History")

    for chat in reversed(st.session_state.chat_history):

        st.markdown("### 🙋 Question:")
        st.write(chat["question"])

        st.markdown("### 🤖 Answer:")
        st.write(chat["answer"])

        st.divider()