"""
Chat Engine Module for LLM Integration and Response Generation
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage


class ChatEngine:
    """
    Manages LLM interactions with context-aware responses.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-flash-latest"):
        """
        Initialize chat engine with Gemini LLM.
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model version to use
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=2048
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.rag_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a professional document analysis assistant. "
                "Answer questions based strictly on the provided context. "
                "Format your responses using structured Markdown: "
                "- Use **bold** for key terms and project names. "
                "- Use bullet points or numbered lists for multiple items. "
                "- Use tables for structured data like technologies or dates. "
                "If the answer is not in the context, say 'I cannot find this information in the document.' "
                "Do not hallucinate or invent information. "
                "Provide concise, accurate responses. "
                "Cite relevant sections when possible.\n\n"
                "Context from document:\n{context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    
    def create_qa_chain(self, retriever):
        """
        Create a retrieval-augmented generation chain.
        
        Args:
            retriever: Vector store retriever object
            
        Returns:
            Runnable chain for QA
        """
        document_chain = create_stuff_documents_chain(
            self.llm, 
            self.rag_prompt
        )
        retrieval_chain = create_retrieval_chain(
            retriever, 
            document_chain
        )
        return retrieval_chain
    
    def generate_response(
        self, 
        query: str, 
        relevant_docs: List[Document]
    ) -> str:
        """
        Generate response using retrieved context.
        
        Args:
            query: User question
            relevant_docs: Retrieved relevant document chunks
            
        Returns:
            Generated response string
        """
        if not relevant_docs:
            return "No relevant content found in the uploaded document. Please try a different question."
        
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = self.rag_prompt.format_messages(
            context=context,
            chat_history=self.memory.load_memory_variables({})["chat_history"],
            input=query
        )
        
        response = self.llm.invoke(prompt)
        
        # Ensure we only get the text content if response is a list or dict
        answer = ""
        if isinstance(response.content, str):
            answer = response.content
        elif isinstance(response.content, list):
            for part in response.content:
                if isinstance(part, dict) and "text" in part:
                    answer += part["text"]
                elif isinstance(part, str):
                    answer += part
        
        self.memory.chat_memory.add_message(HumanMessage(content=query))
        self.memory.chat_memory.add_message(AIMessage(content=answer))
        
        return answer
    
    def clear_memory(self) -> None:
        """
        Clear conversation memory for new document session.
        """
        self.memory.clear()