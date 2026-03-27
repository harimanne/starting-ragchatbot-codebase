import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # LLM backend: "anthropic" or "ollama"
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")

    # Anthropic API settings (used when LLM_BACKEND=anthropic)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"

    # Ollama settings (used when LLM_BACKEND=ollama)
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3")
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


