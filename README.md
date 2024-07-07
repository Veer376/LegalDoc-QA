# Legal Document QA System

A sophisticated legal document question-answering system built with Streamlit, LangChain, LlamaIndex, and OpenAI's ChatGPT.

## Features

- **Document Ingestion & Processing**: Supports PDF, Word, and HTML legal documents
- **Advanced Retrieval**: Hierarchical vector index for both coarse and fine-grained document retrieval
- **Conversational QA**: Chat with your legal documents using natural language
- **Citation Support**: All answers include relevant citations to source documents
- **Document Explorer**: View and manage your documents and their processed segments

## Architecture

### Data Ingestion & Preprocessing

- Supports diverse legal sources (PDFs, Word docs, HTML)
- Standardizes content into plain-text segments
- Applies rule-based cleaning to preserve clause boundaries and legal citations

### Index Construction with LlamaIndex

- Chunks documents by semantic units
- Embeds them using OpenAI's embedding models
- Builds a hierarchical vector index enabling both coarse- and fine-grained retrieval

### Query Pipeline via LangChain

- Implements a two-stage retrieval chain:
  - First retrieves top-k relevant documents
  - Then retrieves the most relevant segments from those documents
- Uses prompt templates that condition the model on citation formatting, neutrality, and legal tone

### Answer Generation with ChatGPT API

- Tunes temperature and max-token settings for optimal precision
- Enforces guardrails to avoid hallucinations by verifying citations against the index

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/LegalDoc-QA.git
cd LegalDoc-QA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp template.env .env
```
Then edit the `.env` file to add your OpenAI API key.

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run app/app.py
```

### Using the System

1. **Upload Documents**: Use the sidebar to upload legal documents (PDF, DOCX, HTML)
2. **Process Documents**: Click "Process Uploaded Documents" to extract and segment text
3. **Build Index**: Click "Build Index" to create the vector index for retrieval
4. **Ask Questions**: Type your legal questions in the chat input to get answers with citations

## Project Structure

```
.
├── app/
│   └── app.py              # Streamlit application
├── data/
│   ├── processed/          # Processed document segments
│   ├── raw/                # Original uploaded documents
│   └── index/              # Vector indices
├── src/
│   ├── data_ingestion/     # Document loading and processing
│   ├── indexing/           # Vector index construction
│   ├── query_engine/       # Question answering logic
│   └── utils/              # Utility functions
├── requirements.txt        # Project dependencies
├── template.env            # Environment variable template
└── README.md               # Project documentation
```

## Dependencies

- Streamlit: Web interface
- LangChain: Chain-of-thought prompting and LLM orchestration
- LlamaIndex: Document indexing and retrieval
- OpenAI API: ChatGPT for answer generation
- Various document parsers (pypdf, python-docx, beautifulsoup4)

## Future Improvements

- Support for additional document formats
- Advanced document pre-processing for complex legal documents
- Implementation of re-ranking to improve retrieval precision
- Integration with legal knowledge bases
- Citation verification and fact-checking