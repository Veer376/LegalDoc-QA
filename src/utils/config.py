import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_api_key():
    """Retrieve OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in .env file.")
    return api_key

def get_data_path():
    """Returns the path to the data directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data")

def get_raw_data_path():
    """Returns the path to the raw data directory."""
    return os.path.join(get_data_path(), "raw")

def get_processed_data_path():
    """Returns the path to the processed data directory."""
    return os.path.join(get_data_path(), "processed")