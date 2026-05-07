"""
DocQuery - Intelligent PDF Document Query System
A Retrieval-Augmented Generation (RAG) pipeline for interactive document analysis.
"""

import os
import time
import logging
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
import markdown

from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.chat_engine import ChatEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DocQuery - Intelligent PDF Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
    <style>
    /* Main Layout Improvements */
    .main {
        background-color: #fcfcfc;
    }
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Stat Cards */
    .stat-card {
        background-color: white;
        color: #1f1f1f;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #448aff;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Chat Message Styling - Premium Look */
    .chat-container {
        max-width: 850px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin: 1.2rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        position: relative;
        transition: all 0.2s ease;
    }
    .user-message {
        background-color: #e3f2fd;
        color: #1565c0;
        border-bottom-right-radius: 2px;
        margin-left: 2rem;
        border: 1px solid #bbdefb;
    }
    .assistant-message {
        background-color: #ffffff;
        color: #2c3e50;
        border-bottom-left-radius: 2px;
        margin-right: 2rem;
        border: 1px solid #eceff1;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }
    .message-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: block;
        letter-spacing: 0.5px;
    }
    .user-label { color: #1976d2; }
    .assistant-label { color: #2e7d32; }



    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .user-message { margin-left: 0.5rem; }
        .assistant-message { margin-right: 0.5rem; }
        .chat-message { padding: 0.8rem 1rem; font-size: 0.95rem; }
        .main-header { padding: 1.5rem 1rem; margin-bottom: 1.5rem; }
        .main-header h1 { font-size: 1.8rem !important; }
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """
    Initialize all session state variables.
    """
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    
    if "processing_time" not in st.session_state:
        st.session_state.processing_time = None
    
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0


def process_document(uploaded_file) -> bool:
    """
    Process uploaded PDF document and create vector store.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        start_time = time.time()
        
        with st.spinner("Processing document..."):
            processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
            chunks = processor.process_pdf(uploaded_file)
            
            st.session_state.chunk_count = len(chunks)
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("Google API key not found. Please check your environment variables.")
                return False
            
            vector_manager = VectorStoreManager(api_key=api_key)
            st.session_state.vector_store = vector_manager.create_vector_store(chunks)
            
            st.session_state.chat_engine = ChatEngine(api_key=api_key)
            
            processing_time = time.time() - start_time
            st.session_state.processing_time = processing_time
            
            st.session_state.document_processed = True
            st.session_state.document_name = uploaded_file.name
            
            logger.info(f"Document processed: {uploaded_file.name}, chunks: {len(chunks)}, time: {processing_time:.2f}s")
            
            return True
            
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        st.error(f"Failed to process document: {str(e)}")
        return False


def generate_answer(question: str) -> str:
    """
    Generate answer using RAG pipeline.
    
    Args:
        question: User's question
        
    Returns:
        Generated answer
    """
    if st.session_state.vector_store is None:
        return "Please upload and process a document first."
    
    try:
        relevant_docs = st.session_state.vector_store.similarity_search(question, k=4)
        answer = st.session_state.chat_engine.generate_response(question, relevant_docs)
        return answer
        
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return f"An error occurred while generating response: {str(e)}"


def main() -> None:
    """
    Main application entry point.
    """
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## DocQuery")
        st.markdown("Intelligent PDF Document Query System")
        st.markdown("---")
        
        st.markdown("### Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload a PDF document to enable intelligent querying"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary", use_container_width=True):
                success = process_document(uploaded_file)
                if success:
                    st.success("Document processed successfully!")
                    st.session_state.messages = []
                    if st.session_state.chat_engine:
                        st.session_state.chat_engine.clear_memory()
        
        st.markdown("---")
        
        if st.session_state.document_processed:
            st.markdown("### Document Status")
            st.markdown(f"**Active Document:** {st.session_state.document_name}")
            st.markdown(f"**Processing Time:** {st.session_state.processing_time:.2f} seconds")
            st.markdown(f"**Text Chunks Generated:** {st.session_state.chunk_count}")
            
            st.markdown("---")
            st.markdown("### 📊 Document Insights")
            st.markdown(f"""
            <div class="stat-card">
                <strong>Density:</strong> {st.session_state.chunk_count / max(1, st.session_state.processing_time):.1f} chunks/sec<br>
                <strong>Search Depth:</strong> Top 4 Matches<br>
                <strong>Memory:</strong> Context-Aware
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 💡 Pro Tips")
            st.info("Ask 'Summarize the document' for a quick overview, or ask specific technical questions for detailed extraction.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        DocQuery uses Retrieval-Augmented Generation (RAG) to provide accurate 
        answers from your PDF documents. The system combines:
        
        - Semantic search with FAISS
        - Context-aware responses using Gemini AI
        - Conversation memory for follow-up questions
        """)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.chat_engine:
                st.session_state.chat_engine.clear_memory()
            st.success("Chat history cleared!")
    
    # Main content area
    st.markdown(f"""
        <div class="main-header">
            <h1>📄 DocQuery AI</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">
                Intelligent Retrieval-Augmented Generation (RAG) for PDF Analysis
            </p>
            <div style="margin-top: 10px; font-size: 0.9rem; font-weight: 300;">
                Powered by Gemini 1.5 & FAISS Vector Search
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.document_processed:
        st.info("""
        Welcome to DocQuery. To get started:
        
        1. Use the sidebar to upload a PDF document
        2. Click 'Process Document' to index the content
        3. Start asking questions about your document
        
        The system will provide accurate answers based strictly on the document content.
        """)
        
        # Example use cases
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Use Case 1**\n\nResearch Papers\nExtract methodologies, results, and conclusions")
        with col2:
            st.markdown("**Use Case 2**\n\nLegal Documents\nQuickly locate clauses and provisions")
        with col3:
            st.markdown("**Use Case 3**\n\nTechnical Manuals\nFind specific procedures and specifications")
    
    else:
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            for idx, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    user_html = markdown.markdown(message["content"])
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<span class="message-label user-label">You</span>'
                        f'{user_html}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Convert markdown content to HTML for proper rendering inside our styled div
                    html_content = markdown.markdown(message["content"], extensions=['tables'])
                    
                    st.markdown(
                        f'<div class="chat-message assistant-message">'
                        f'<span class="message-label assistant-label">DocQuery</span>'
                        f'{html_content}</div>',
                        unsafe_allow_html=True
                    )
        
        # Question input
        with st.container():
            st.markdown("---")
            question = st.text_input(
                "Ask a question about your document:",
                placeholder="Example: What are the main findings of this research?",
                key="question_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit_button = st.button("Submit", type="primary", use_container_width=True)
            with col2:
                if st.button("Clear", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            
            if submit_button and question:
                st.session_state.messages.append({"role": "user", "content": question})
                
                with st.spinner("Analyzing document and generating response..."):
                    answer = generate_answer(question)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()


if __name__ == "__main__":
    main()