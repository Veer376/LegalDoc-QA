import os
import logging
from typing import List, Dict, Any
import pandas as pd

# Document loaders
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Base class for document loaders"""
    
    def load(self, file_path: str) -> str:
        """Load document and return text content"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def clean_text(self, text: str) -> str:
        """Clean the extracted text to preserve clause boundaries and citations"""
        # Basic text cleanup
        text = text.replace('\n\n', ' [PARAGRAPH_BREAK] ')
        text = text.replace('\n', ' ')
        
        # Preserving legal citations (common patterns)
        # This is a simplified version - real implementation would be more comprehensive
        citation_patterns = [
            r'\d+ U\.S\. \d+',  # US Reports
            r'\d+ F\.\d+d \d+',  # Federal Reporter
            r'\d+ S\.Ct\. \d+',  # Supreme Court Reporter
            r'[A-Za-z]+ v\. [A-Za-z]+',  # Case names
        ]
        
        # Preserve clause boundaries by maintaining paragraph structure
        text = text.replace('[PARAGRAPH_BREAK]', '\n\n')
        
        return text

class PdfLoader(DocumentLoader):
    """Loader for PDF documents"""
    
    def load(self, file_path: str) -> str:
        """Load PDF document and extract text"""
        try:
            logger.info(f"Loading PDF: {file_path}")
            reader = PdfReader(file_path)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""

class DocxLoader(DocumentLoader):
    """Loader for Word documents"""
    
    def load(self, file_path: str) -> str:
        """Load Word document and extract text"""
        try:
            logger.info(f"Loading DOCX: {file_path}")
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            return ""

class HtmlLoader(DocumentLoader):
    """Loader for HTML documents"""
    
    def load(self, file_path: str) -> str:
        """Load HTML document and extract text"""
        try:
            logger.info(f"Loading HTML: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Extract text from paragraph elements
            paragraphs = soup.find_all('p')
            text = "\n\n".join([p.get_text() for p in paragraphs])
            
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error loading HTML {file_path}: {e}")
            return ""

def get_loader_for_file(file_path: str) -> DocumentLoader:
    """Return appropriate loader based on file extension"""
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.pdf':
        return PdfLoader()
    elif extension == '.docx':
        return DocxLoader()
    elif extension in ['.html', '.htm']:
        return HtmlLoader()
    else:
        raise ValueError(f"Unsupported file type: {extension}")