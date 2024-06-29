import logging
from typing import Dict, Any, List

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Import utilities
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import get_api_key
from query_engine.utils import (
    LEGAL_QA_SYSTEM_PROMPT,
    LEGAL_QA_PROMPT,
    format_document_context
)

logger = logging.getLogger(__name__)

class LegalQAChain:
    """
    RAG Chain for legal document QA using LangChain and OpenAI.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=1000):
        """
        Initialize the chain.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for LLM sampling (lower for more deterministic outputs)
            max_tokens: Maximum tokens in the LLM response
        """
        # Get API key
        api_key = get_api_key()
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            system=LEGAL_QA_SYSTEM_PROMPT
        )
        
        # Initialize the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=LEGAL_QA_PROMPT,
            verbose=True
        )
        
        logger.info(f"Initialized LegalQAChain with model: {model_name}")
    
    def run(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        """
        Run the chain to generate an answer.
        
        Args:
            query: User query
            documents: Retrieved relevant documents
        
        Returns:
            Generated answer
        """
        # Format documents into context
        context = format_document_context(documents)
        
        try:
            # Run the chain
            result = self.chain.invoke({
                "question": query,
                "context": context
            })
            
            # Add the source documents to the output
            result["source_documents"] = documents
            
            return result
        except Exception as e:
            logger.error(f"Error running LegalQAChain: {e}")
            return {"text": f"Error generating answer: {str(e)}", "source_documents": []}


class ConversationalLegalQAChain:
    """
    Conversational RAG Chain for legal document QA that maintains conversation history.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.1, max_tokens=1000):
        """
        Initialize the chain.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for LLM sampling (lower for more deterministic outputs)
            max_tokens: Maximum tokens in the LLM response
        """
        # Get API key
        api_key = get_api_key()
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            system=LEGAL_QA_SYSTEM_PROMPT
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        
        # Initialize the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=LEGAL_QA_PROMPT,
            verbose=True,
            memory=self.memory
        )
        
        logger.info(f"Initialized ConversationalLegalQAChain with model: {model_name}")
    
    def run(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        """
        Run the chain to generate an answer.
        
        Args:
            query: User query
            documents: Retrieved relevant documents
        
        Returns:
            Generated answer
        """
        # Format documents into context
        context = format_document_context(documents)
        
        try:
            # Run the chain
            result = self.chain.invoke({
                "question": query,
                "context": context
            })
            
            # Add the source documents to the output
            result["source_documents"] = documents
            
            return result
        except Exception as e:
            logger.error(f"Error running ConversationalLegalQAChain: {e}")
            return {"text": f"Error generating answer: {str(e)}", "source_documents": []}