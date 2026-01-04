"""Configuration management for PulmoRAG"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Config:
    """Base configuration"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")

    # Vector Store
    VECTOR_STORE = os.getenv("VECTOR_STORE", "chromadb")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "vectordb"))
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "copd_documents")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pulmorag-index")

    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")

    # Reranker Configuration
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 20))  # Docs to retrieve before rerank

    # Local LLM Configuration
    LOCAL_MODEL_PATH = os.getenv(
        "LOCAL_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "mistrallite.Q4_K_M.gguf")
    )
    LOCAL_MODEL_CTX = int(os.getenv("LOCAL_MODEL_CTX", 4096))
    LOCAL_MODEL_GPU_LAYERS = int(os.getenv("LOCAL_MODEL_GPU_LAYERS", 0))
    LOCAL_MODEL_BATCH_SIZE = int(os.getenv("LOCAL_MODEL_BATCH_SIZE", 512))
    LOCAL_MODEL_THREADS = int(os.getenv("LOCAL_MODEL_THREADS", 0))  # 0 = auto-detect
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "auto")  # auto, true, false

    # RAG Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))
    TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

    # Hybrid Retrieval Configuration
    BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(PROJECT_ROOT / "data" / "bm25_index"))
    HYBRID_FUSION_METHOD = os.getenv("HYBRID_FUSION_METHOD", "rrf")  # rrf or weighted
    HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", 0.7))
    USE_HYBRID_RETRIEVAL = os.getenv("USE_HYBRID_RETRIEVAL", "true").lower() == "true"

    # Application Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False


def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
