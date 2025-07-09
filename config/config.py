import os
from dotenv import load_dotenv

# load env
load_dotenv()

class Config:
    # load api
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

    # llm and embedding
    LLM_MODEL = 'models/gemini-2.5-pro'
    EMBEDDING_MODEL = 'models/text-embedding-004'

    # pinecone config
    INDEX_NAME = 'medibot'
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
    METRIC = "cosine"

    # document config
    DATA_PATH = "data"
    PDF_GLOB = '*.pdf'
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # retriever config
    RETRIEVAL_K = 5

    # vector store config
    BATCH_SIZE = 200

    # System Prompt
    SYSTEM_PROMPT = (
        "You are a helpful and knowledgeable medical assistant. "
        "Answer the user's following medical question in simple and accurate terms. "
        "It would be preferred if you add some bullet points when answering the question.\n\n"
        "Use the following retrieved context to answer the question:\n"
        "{context}\n\n"
        "If the answer is not in the context, say you don't know."
    )

