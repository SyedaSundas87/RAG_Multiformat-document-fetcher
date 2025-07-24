import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stFileUploader>div>div>div>button {
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #e9f5ff;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    .header {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .file-info {
        background-color: #e2f0fd;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class RAGSystem:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = None
        self.documents = []
        self.file_info = None

        # Initialize Groq client
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def load_and_process_document(self, file_path: str, file_type: str):
        """Load and process different file types"""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path)
            elif file_type == "csv":
                loader = CSVLoader(file_path)
            elif file_type == "excel":
                loader = UnstructuredExcelLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            pages = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.documents = text_splitter.split_documents(pages)

            # Store file info for display
            self.file_info = {
                "type": file_type.upper(),
                "pages": len(pages),
                "chunks": len(self.documents)
            }

            # Create embeddings
            texts = [doc.page_content for doc in self.documents]
            embeddings = self.embedding_model.encode(texts)

            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_db = faiss.IndexFlatL2(dimension)
            self.vector_db.add(embeddings)
            return True
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False

    def query(self, question: str, k: int = 3) -> str:
        """Search and answer question"""
        if not self.vector_db:
            return "Please load a document first."

        try:
            # Embed question
            question_embedding = self.embedding_model.encode([question])

            # Search in vector DB
            distances, indices = self.vector_db.search(question_embedding, k)

            # Get relevant docs
            relevant_docs = [self.documents[i] for i in indices[0]]
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Generate answer
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant. Provide detailed, accurate answers based on the context."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in detail with proper formatting:"
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def main():
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("### Document Processing")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This RAG application supports:
        - PDF documents
        - CSV files
        - Text files
        - Excel spreadsheets
        """)

    # Main content
    st.markdown('<h1 class="header">📂 Multi-Format Document Q&A</h1>', unsafe_allow_html=True)

    # Initialize RAG system
    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSystem()

    # File upload section
    st.markdown("### 📤 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "csv", "xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Determine file type
        file_type = uploaded_file.name.split(".")[-1].lower()

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Process document with nice UI feedback
        with st.spinner(f"Processing {uploaded_file.name}..."):
            success = st.session_state.rag.load_and_process_document(tmp_path, file_type)
            os.unlink(tmp_path)

        if success:
            st.markdown(f"""
            <div class="file-info">
                <h4>📄 {uploaded_file.name}</h4>
                <p><strong>Type:</strong> {st.session_state.rag.file_info['type']}</p>
                <p><strong>Pages/Records:</strong> {st.session_state.rag.file_info['pages']}</p>
                <p><strong>Text Chunks:</strong> {st.session_state.rag.file_info['chunks']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.balloons()
            st.success("Document processed successfully! You can now ask questions.")

    # Question input section
    st.markdown("### ❓ Ask About Your Document")
    question = st.text_area(
        "Enter your question here...",
        height=100,
        placeholder="What would you like to know about the document?"
    )

    if st.button("Get Answer", use_container_width=True):
        if question:
            with st.spinner("Analyzing document and generating answer..."):
                answer = st.session_state.rag.query(question)

            st.markdown("### 📝 Answer")
            st.markdown(answer)

            # Add some visual separation
            st.markdown("---")
            st.markdown("### 🔍 Relevant Context")
            st.info("The answer above was generated based on the most relevant sections of your document.")
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()
