import logging
from typing import Dict, Any, List, Optional
import os

# Local imports
from .chains import LegalQAChain, ConversationalLegalQAChain
from .utils import get_relevant_segments

# Import from indexing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indexing import LegalDocumentIndexer

logger = logging.getLogger(__name__)

class LegalDocumentQAEngine:
    """
    Main engine for legal document question answering.
    Combines LlamaIndex for retrieval and LangChain for answer generation.
    """
    
    def __init__(
        self,
        index_dir: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_k_documents: int = 3,
        top_k_segments: int = 5,
        use_conversation_memory: bool = False
    ):
        """
        Initialize the QA engine.
        
        Args:
            index_dir: Directory containing the indices
            model_name: Name of the OpenAI model to use
            temperature: Temperature for answer generation
            max_tokens: Maximum tokens in the response
            top_k_documents: Number of documents to retrieve in first stage
            top_k_segments: Number of segments to retrieve in second stage
            use_conversation_memory: Whether to use conversation memory
        """
        self.index_dir = index_dir
        self.top_k_documents = top_k_documents
        self.top_k_segments = top_k_segments
        
        # Load indices
        self.indexer = LegalDocumentIndexer(processed_data_dir="", index_dir=index_dir)
        self.indices = self.indexer.load_indices()
        
        # Check if indices are loaded
        if not self.indices:
            logger.error("No indices loaded. Please build indices first.")
        
        # Initialize QA chain
        if use_conversation_memory:
            self.qa_chain = ConversationalLegalQAChain(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            self.qa_chain = LegalQAChain(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        logger.info(f"Initialized LegalDocumentQAEngine with model: {model_name}")
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a legal question using RAG.
        
        Args:
            query: User question
            
        Returns:
            Answer and source documents
        """
        if not self.indices:
            return {"answer": "No document indices available. Please index documents first.", "source_documents": []}
        
        try:
            # Retrieve relevant segments
            relevant_segments = get_relevant_segments(
                query=query,
                indices=self.indices,
                top_k_documents=self.top_k_documents,
                top_k_segments=self.top_k_segments
            )
            
            if not relevant_segments:
                return {"answer": "No relevant documents found for the query.", "source_documents": []}
            
            # Generate answer
            result = self.qa_chain.run(query, relevant_segments)
            
            # Format the output
            answer = result.get("text", "")
            source_documents = result.get("source_documents", [])
            
            return {"answer": answer, "source_documents": source_documents}
        
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"answer": f"Error answering question: {str(e)}", "source_documents": []}