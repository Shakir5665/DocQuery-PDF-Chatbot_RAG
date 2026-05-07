"""
PDF Processing Module for Document Loading and Text Extraction
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



class PDFProcessor:
    """
    Handles PDF document loading and text chunking for RAG pipeline.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with configurable chunking parameters.
        
        Args:
            chunk_size: Number of characters per text chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_file) -> List[Document]:
        """
        Load and parse a PDF file into LangChain Document objects.
        
        Args:
            pdf_file: Uploaded PDF file object from Streamlit
            
        Returns:
            List of Document objects containing the PDF content
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_path = temp_file.name
        
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            return documents
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        for idx, doc in enumerate(chunked_docs):
            doc.metadata["chunk_id"] = idx
        
        return chunked_docs
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """
        Complete pipeline: load PDF and chunk into documents.
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        documents = self.load_pdf(pdf_file)
        chunked_documents = self.chunk_documents(documents)
        return chunked_documents