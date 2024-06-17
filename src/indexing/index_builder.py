import os
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, ServiceContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

# Import from utils
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_api_key

logger = logging.getLogger(__name__)

class LegalDocumentIndexer:
    """
    Creates and manages vector indices for legal documents using LlamaIndex.
    Implements hierarchical indexing for both document-level and segment-level retrieval.
    """
    
    def __init__(self, processed_data_dir: str, index_dir: str):
        """
        Initialize the indexer.
        
        Args:
            processed_data_dir: Directory containing processed documents
            index_dir: Directory to store the index
        """
        self.processed_data_dir = processed_data_dir
        self.index_dir = index_dir
        
        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Get OpenAI API key
        api_key = get_api_key()
        
        # Set up embedding model
        self.embed_model = OpenAIEmbedding(
            api_key=api_key,
            model_name="text-embedding-3-small"  # Using OpenAI's efficient embedding model
        )
        
        # Configure LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        
    def build_index(self) -> None:
        """Build hierarchical vector indices for legal documents"""
        logger.info("Starting to build document index")
        
        # Load processed documents
        documents = self._load_processed_documents()
        
        if not documents:
            logger.warning("No documents found to index")
            return
        
        # Convert to LlamaIndex Document objects
        llama_docs = self._convert_to_llama_documents(documents)
        
        # Build document-level index
        document_index = self._build_document_level_index(llama_docs)
        
        # Build segment-level index
        segment_index = self._build_segment_level_index(llama_docs)
        
        # Save indices
        self._save_indices(document_index, segment_index)
        
        logger.info("Successfully built document and segment indices")
        
    def _load_processed_documents(self) -> List[Dict]:
        """Load processed documents from JSON file"""
        json_path = os.path.join(self.processed_data_dir, "documents.json")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} processed documents")
            return documents
        except FileNotFoundError:
            logger.error(f"Processed documents file not found: {json_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_path}")
            return []
    
    def _convert_to_llama_documents(self, documents: List[Dict]) -> List[Document]:
        """Convert processed documents to LlamaIndex Document objects"""
        llama_docs = []
        
        for doc in documents:
            # Create LlamaIndex Document with text and metadata
            llama_doc = Document(
                text=doc["text"],
                metadata={
                    "doc_id": doc["doc_id"],
                    "segment_id": doc["segment_id"],
                    "source": doc["metadata"]["source"],
                    "segment_index": doc["metadata"]["segment_index"]
                }
            )
            llama_docs.append(llama_doc)
        
        return llama_docs
    
    def _build_document_level_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build document-level index for coarse retrieval"""
        logger.info("Building document-level index")
        
        # Create document-level documents by concatenating segments from same document
        doc_level_docs = {}
        
        for doc in documents:
            doc_id = doc.metadata["doc_id"]
            
            if doc_id not in doc_level_docs:
                doc_level_docs[doc_id] = {
                    "text": doc.text,
                    "metadata": {
                        "doc_id": doc_id,
                        "source": doc.metadata["source"]
                    }
                }
            else:
                doc_level_docs[doc_id]["text"] += f"\n\n{doc.text}"
        
        # Convert to LlamaIndex Documents
        doc_level_documents = [
            Document(text=doc["text"], metadata=doc["metadata"])
            for doc in doc_level_docs.values()
        ]
        
        # Build vector index
        storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
        document_index = VectorStoreIndex.from_documents(
            doc_level_documents,
            storage_context=storage_context
        )
        
        return document_index
    
    def _build_segment_level_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build segment-level index for fine-grained retrieval"""
        logger.info("Building segment-level index")
        
        # Build vector index
        storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
        segment_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        
        return segment_index
    
    def _save_indices(self, document_index: VectorStoreIndex, segment_index: VectorStoreIndex) -> None:
        """Save indices to disk"""
        # Save document-level index
        document_index_path = os.path.join(self.index_dir, "document_index")
        document_index.storage_context.persist(persist_dir=document_index_path)
        logger.info(f"Saved document-level index to {document_index_path}")
        
        # Save segment-level index
        segment_index_path = os.path.join(self.index_dir, "segment_index")
        segment_index.storage_context.persist(persist_dir=segment_index_path)
        logger.info(f"Saved segment-level index to {segment_index_path}")
    
    def load_indices(self) -> Dict[str, VectorStoreIndex]:
        """Load indices from disk"""
        logger.info("Loading indices from disk")
        
        indices = {}
        
        # Load document-level index
        document_index_path = os.path.join(self.index_dir, "document_index")
        if os.path.exists(document_index_path):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=document_index_path
                )
                indices["document_index"] = VectorStoreIndex.from_storage_context(
                    storage_context=storage_context
                )
                logger.info(f"Loaded document-level index from {document_index_path}")
            except Exception as e:
                logger.error(f"Error loading document-level index: {e}")
        
        # Load segment-level index
        segment_index_path = os.path.join(self.index_dir, "segment_index")
        if os.path.exists(segment_index_path):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=segment_index_path
                )
                indices["segment_index"] = VectorStoreIndex.from_storage_context(
                    storage_context=storage_context
                )
                logger.info(f"Loaded segment-level index from {segment_index_path}")
            except Exception as e:
                logger.error(f"Error loading segment-level index: {e}")
        
        return indices