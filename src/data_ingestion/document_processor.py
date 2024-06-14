import os
import logging
from typing import List, Dict, Any
import pandas as pd
import json
import re
from pathlib import Path

from .document_loaders import get_loader_for_file

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class for processing legal documents and standardizing them into segments"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def process_documents(self) -> None:
        """Process all documents in raw data directory"""
        logger.info("Starting document processing")
        
        all_documents = []
        document_files = self._get_document_files()
        
        for file_path in document_files:
            logger.info(f"Processing document: {file_path}")
            
            # Get loader based on file extension
            loader = get_loader_for_file(file_path)
            
            # Load and clean document text
            text = loader.load(file_path)
            
            # Skip empty documents
            if not text.strip():
                logger.warning(f"Skipping empty document: {file_path}")
                continue
            
            # Process document into segments
            segments = self.segment_document(text, os.path.basename(file_path))
            
            all_documents.extend(segments)
        
        # Save processed documents
        if all_documents:
            self._save_processed_documents(all_documents)
            logger.info(f"Successfully processed {len(document_files)} documents into {len(all_documents)} segments")
        else:
            logger.warning("No documents were processed")
    
    def _get_document_files(self) -> List[str]:
        """Get all document files from raw data directory"""
        document_files = []
        
        for root, _, files in os.walk(self.raw_data_dir):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                
                if file_extension in ['.pdf', '.docx', '.html', '.htm']:
                    document_files.append(os.path.join(root, file))
        
        return document_files
    
    def segment_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Segment document into semantically coherent chunks"""
        segments = []
        
        # Split text by paragraphs
        paragraphs = text.split("\n\n")
        
        # Process each paragraph into segments
        current_segment = ""
        segment_idx = 0
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # If adding this paragraph would make the segment too long,
            # save current segment and start a new one
            if len(current_segment) + len(paragraph) > 1000:  # Max segment size of ~1000 chars
                if current_segment:
                    segments.append({
                        "doc_id": doc_id,
                        "segment_id": f"{doc_id}_segment_{segment_idx}",
                        "text": current_segment.strip(),
                        "metadata": {
                            "source": doc_id,
                            "segment_index": segment_idx
                        }
                    })
                    segment_idx += 1
                    current_segment = paragraph
                else:
                    # If a single paragraph is too long, we need to split it further
                    # (simplified approach - in practice, would need more sophisticated splitting)
                    segments.append({
                        "doc_id": doc_id,
                        "segment_id": f"{doc_id}_segment_{segment_idx}",
                        "text": paragraph[:1000].strip(),
                        "metadata": {
                            "source": doc_id,
                            "segment_index": segment_idx
                        }
                    })
                    segment_idx += 1
            else:
                current_segment += "\n\n" + paragraph if current_segment else paragraph
        
        # Don't forget the last segment
        if current_segment:
            segments.append({
                "doc_id": doc_id,
                "segment_id": f"{doc_id}_segment_{segment_idx}",
                "text": current_segment.strip(),
                "metadata": {
                    "source": doc_id,
                    "segment_index": segment_idx
                }
            })
        
        return segments
    
    def _save_processed_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Save processed documents to CSV and JSON files"""
        # Create DataFrame from documents
        df = pd.DataFrame(documents)
        
        # Save as CSV
        csv_path = os.path.join(self.processed_data_dir, "documents.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed documents to {csv_path}")
        
        # Save as JSON (for LlamaIndex)
        json_path = os.path.join(self.processed_data_dir, "documents.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed documents to {json_path}")