"""Embedding generation for RAG retrieval"""

from typing import List, Optional
from abc import ABC, abstractmethod

from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedders"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension"""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings using sentence-transformers"""
    
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dim={self._dimension})")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple texts with batching"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API embeddings"""
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (uses env var if not provided)
        """
        from openai import OpenAI
        
        config = get_config()
        api_key = api_key or config.OPENAI_API_KEY
        
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dimension = self.DIMENSIONS.get(model_name, 1536)
        logger.info(f"OpenAI embedder initialized: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed multiple texts with batching"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedder(
    embedder_type: str = "local",
    model_name: Optional[str] = None,
) -> BaseEmbedder:
    """
    Factory function to get an embedder.
    
    Args:
        embedder_type: 'local' or 'openai'
        model_name: Model name override
        
    Returns:
        Embedder instance
    """
    if embedder_type == "local":
        model = model_name or "nomic-ai/nomic-embed-text-v1.5"
        return SentenceTransformerEmbedder(model_name=model)
    elif embedder_type == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedder(model_name=model)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
