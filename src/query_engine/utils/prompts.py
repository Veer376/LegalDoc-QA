from langchain.prompts import PromptTemplate

# System prompt for legal document QA
LEGAL_QA_SYSTEM_PROMPT = """You are an AI legal assistant trained to answer questions about legal documents.
Always maintain a neutral, professional tone in your responses.
Format your answers in clear legal language, similar to a legal brief.
When citing from documents, use the following format: [Source: document_name, segment_id].
Only make assertions that are directly supported by the provided context.
If the context doesn't contain information to answer the question, acknowledge this limitation.
Do not fabricate citations or references.
Avoid providing legal advice or opinions not directly supported by the documents."""

# Prompt template for retrieval augmented generation with citations
LEGAL_QA_PROMPT_TEMPLATE = """
Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question: {question}

Remember to:
1. Answer only from the provided context.
2. Include specific citations to the source documents using [Source: document_name, segment_id] format.
3. Maintain a neutral, professional tone appropriate for legal context.
4. If you don't have enough information to answer, say so clearly.
"""

# Create a LangChain prompt template
LEGAL_QA_PROMPT = PromptTemplate(
    template=LEGAL_QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# Template for generating document context from retrieved segments
DOCUMENT_CONTEXT_TEMPLATE = """Document: {doc_id}
Segment ID: {segment_id}
Content:
{text}
"""

def format_document_context(documents):
    """
    Format retrieved documents into a single context string with proper citations.
    
    Args:
        documents: List of retrieved document segments.
        
    Returns:
        Formatted context string.
    """
    context_parts = []
    
    for i, doc in enumerate(documents):
        doc_id = doc.metadata.get("doc_id", f"Document {i+1}")
        segment_id = doc.metadata.get("segment_id", f"Segment {i+1}")
        
        formatted_doc = DOCUMENT_CONTEXT_TEMPLATE.format(
            doc_id=doc_id,
            segment_id=segment_id,
            text=doc.text
        )
        
        context_parts.append(formatted_doc)
    
    return "\n\n".join(context_parts)