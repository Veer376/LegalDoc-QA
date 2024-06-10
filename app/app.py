import streamlit as st
import os
import sys
import logging
import datetime
import time
from pathlib import Path

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import components
from data_ingestion import DocumentProcessor
from indexing import LegalDocumentIndexer
from query_engine.legal_qa_engine import LegalDocumentQAEngine
from utils.config import get_data_path, get_raw_data_path, get_processed_data_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = get_data_path()
RAW_DATA_DIR = get_raw_data_path()
PROCESSED_DATA_DIR = get_processed_data_path()
INDEX_DIR = os.path.join(DATA_DIR, "index")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Legal Document QA System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_qa_engine():
    """Initialize QA engine as a cached resource"""
    return LegalDocumentQAEngine(
        index_dir=INDEX_DIR,
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000,
        top_k_documents=3,
        top_k_segments=5,
        use_conversation_memory=True
    )

def check_openai_api_key():
    """Check if OpenAI API key is set"""
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key is not set. Please enter your API key.")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set!")
            return True
        return False
    return True

def process_uploaded_file(uploaded_file, file_dir):
    """Process an uploaded file and save it"""
    file_path = os.path.join(file_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def main():
    # Header
    st.title("⚖️ Legal Document QA System")
    
    # Check API key
    if not check_openai_api_key():
        st.stop()
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "qa_engine" not in st.session_state:
        st.session_state.qa_engine = get_qa_engine()
    
    # Sidebar
    with st.sidebar:
        st.header("Document Management")
        
        # Document upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload legal documents", 
            type=["pdf", "docx", "html", "htm"],
            accept_multiple_files=True
        )
        
        # Process documents
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                with st.spinner("Processing documents..."):
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        process_uploaded_file(uploaded_file, RAW_DATA_DIR)
                    
                    # Process documents
                    processor = DocumentProcessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
                    processor.process_documents()
                    
                    st.success(f"✅ Processed {len(uploaded_files)} documents")
        
        # Index documents
        st.subheader("Build Document Index")
        if st.button("Build Index"):
            with st.spinner("Building document index..."):
                try:
                    indexer = LegalDocumentIndexer(PROCESSED_DATA_DIR, INDEX_DIR)
                    indexer.build_index()
                    
                    # Refresh QA engine to use new index
                    st.session_state.qa_engine = get_qa_engine()
                    
                    st.success("✅ Document index built successfully")
                except Exception as e:
                    st.error(f"Error building index: {e}")
        
        # Settings
        st.subheader("Settings")
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
        )
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    # Main content area
    tab1, tab2 = st.tabs(["Q&A", "Document Explorer"])
    
    # Q&A Tab
    with tab1:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show citations if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** {source['doc_id']}")
                            st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your legal documents..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get answer from QA engine
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if hasattr(st.session_state, "qa_engine") and st.session_state.qa_engine:
                        qa_response = st.session_state.qa_engine.answer_question(prompt)
                        answer = qa_response.get("answer", "")
                        sources = qa_response.get("source_documents", [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": [
                                {"doc_id": doc.metadata.get("doc_id", "Unknown"), "text": doc.text}
                                for doc in sources
                            ]
                        })
                        
                        # Display sources
                        if sources:
                            with st.expander("Sources"):
                                for i, doc in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('doc_id', 'Unknown')}")
                                    st.text(doc.text[:300] + "..." if len(doc.text) > 300 else doc.text)
                    else:
                        st.error("QA engine not initialized. Please build the document index first.")
    
    # Document Explorer Tab
    with tab2:
        st.header("Document Explorer")
        
        # List raw documents
        st.subheader("Raw Documents")
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if os.path.isfile(os.path.join(RAW_DATA_DIR, f))]
        if raw_files:
            st.write(f"Found {len(raw_files)} raw documents:")
            for file in raw_files:
                st.write(f"📄 {file}")
        else:
            st.write("No raw documents found. Please upload some documents.")
        
        # List processed segments
        st.subheader("Processed Document Segments")
        if os.path.exists(os.path.join(PROCESSED_DATA_DIR, "documents.csv")):
            import pandas as pd
            try:
                df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "documents.csv"))
                st.write(f"Found {len(df)} document segments:")
                st.dataframe(df[["doc_id", "segment_id"]].head(20))
                
                if len(df) > 20:
                    st.write(f"... and {len(df) - 20} more segments.")
            except Exception as e:
                st.error(f"Error loading processed documents: {e}")
        else:
            st.write("No processed document segments found. Please process your documents.")
        
        # Check if index exists
        st.subheader("Document Index")
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            st.write("✅ Document index exists.")
        else:
            st.write("❌ Document index does not exist. Please build the index.")

if __name__ == "__main__":
    main()