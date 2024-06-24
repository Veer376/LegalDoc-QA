import logging
from typing import List, Dict, Any, Optional

# LlamaIndex imports
from llama_index.core import QueryBundle
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

class HierarchicalRetriever:
    """
    Hierarchical retriever that first retrieves relevant documents at a coarse level,
    then retrieves fine-grained segments from those documents.
    """
    
    def __init__(
        self, 
        document_index: VectorStoreIndex,
        segment_index: VectorStoreIndex,
        top_k_documents: int = 3,
        top_k_segments: int = 5
    ):
        """
        Initialize the hierarchical retriever.
        
        Args:
            document_index: Document-level index for coarse retrieval
            segment_index: Segment-level index for fine-grained retrieval
            top_k_documents: Number of documents to retrieve in first stage
            top_k_segments: Number of segments to retrieve in second stage
        """
        self.document_index = document_index
        self.segment_index = segment_index
        self.top_k_documents = top_k_documents
        self.top_k_segments = top_k_segments
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Perform hierarchical retrieval.
        
        Args:
            query: Query string
            
        Returns:
            List of retrieved nodes with relevance scores
        """
        logger.info(f"Performing hierarchical retrieval for query: {query}")
        
        # First stage: retrieve relevant documents
        doc_retriever = self.document_index.as_retriever(
            similarity_top_k=self.top_k_documents
        )
        document_nodes = doc_retriever.retrieve(query)
        
        if not document_nodes:
            logger.warning("No documents retrieved in first stage")
            return []
        
        # Extract relevant document IDs
        doc_ids = [node.metadata.get("doc_id") for node in document_nodes]
        logger.info(f"Retrieved document IDs: {doc_ids}")
        
        # Second stage: retrieve relevant segments from the documents
        # Create a filter to only get segments from the retrieved documents
        segment_retriever = self.segment_index.as_retriever(
            similarity_top_k=self.top_k_segments,
            filters=lambda node: node.metadata.get("doc_id") in doc_ids
        )
        
        # Retrieve segments
        segment_nodes = segment_retriever.retrieve(query)
        
        if not segment_nodes:
            logger.warning("No segments retrieved in second stage")
            # Fall back to document nodes
            return document_nodes
        
        logger.info(f"Retrieved {len(segment_nodes)} relevant segments")
        return segment_nodes


def get_relevant_segments(query: str, indices: Dict[str, VectorStoreIndex], top_k_documents: int = 3, top_k_segments: int = 5) -> List[NodeWithScore]:
    """
    Retrieve relevant document segments for a given query using hierarchical retrieval.
    
    Args:
        query: Query string
        indices: Dictionary containing document and segment indices
        top_k_documents: Number of documents to retrieve in first stage
        top_k_segments: Number of segments to retrieve in second stage
        
    Returns:
        List of relevant document segments
    """
    # Check if both indices are available
    if "document_index" not in indices or "segment_index" not in indices:
        logger.error("Document or segment index not available")
        return []
    
    # Create hierarchical retriever
    retriever = HierarchicalRetriever(
        document_index=indices["document_index"],
        segment_index=indices["segment_index"],
        top_k_documents=top_k_documents,
        top_k_segments=top_k_segments
    )
    
    # Retrieve relevant segments
    return retriever.retrieve(query)