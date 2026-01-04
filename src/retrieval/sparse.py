"""Sparse retrieval using BM25 with persistence"""

import pickle
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.logger import setup_logger
from src.config import get_config
from src.retrieval.vectorstore import SearchResult

logger = setup_logger(__name__)


@dataclass
class BM25Document:
    """Document stored in BM25 index"""
    chunk_id: str
    text: str
    tokens: List[str]
    metadata: Dict[str, Any]


class SparseRetriever:
    """
    BM25-based sparse retrieval with persistent index.
    
    Index is saved to disk for fast loading across restarts.
    """
    
    # Simple tokenization pattern for medical text
    TOKEN_PATTERN = re.compile(r'\b\w+\b')
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        load_on_init: bool = True,
    ):
        """
        Initialize the sparse retriever.
        
        Args:
            index_path: Path to persisted index directory
            load_on_init: Whether to load existing index on initialization
        """
        config = get_config()
        self.index_path = Path(index_path or config.BM25_INDEX_PATH)
        
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[BM25Document] = []
        self._is_loaded = False
        
        if load_on_init and self.index_exists():
            self.load_index()
    
    def index_exists(self) -> bool:
        """Check if persisted index exists."""
        index_file = self.index_path / "bm25_index.pkl"
        docs_file = self.index_path / "bm25_docs.pkl"
        return index_file.exists() and docs_file.exists()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Simple lowercase tokenization optimized for medical terms.
        Preserves medical abbreviations like FEV1, COPD, etc.
        """
        # Lowercase and extract word tokens
        tokens = self.TOKEN_PATTERN.findall(text.lower())
        # Filter very short tokens (except medical abbreviations with numbers)
        return [t for t in tokens if len(t) > 1 or t.isdigit()]
    
    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        save: bool = True,
    ) -> int:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dicts with 'chunk_id', 'text', 'metadata'
            save: Whether to persist index to disk
            
        Returns:
            Number of documents indexed
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        
        self.documents = []
        corpus = []
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.get("text", ""))
            if not tokens:
                continue
            
            doc = BM25Document(
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                tokens=tokens,
                metadata=chunk.get("metadata", {}),
            )
            self.documents.append(doc)
            corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(corpus)
        self._is_loaded = True
        
        logger.info(f"BM25 index built with {len(self.documents)} documents")
        
        if save:
            self.save_index()
        
        return len(self.documents)
    
    def save_index(self) -> None:
        """Persist index to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / "bm25_index.pkl"
        docs_file = self.index_path / "bm25_docs.pkl"
        
        with open(index_file, "wb") as f:
            pickle.dump(self.bm25, f)
        
        with open(docs_file, "wb") as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"BM25 index saved to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load persisted index from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        index_file = self.index_path / "bm25_index.pkl"
        docs_file = self.index_path / "bm25_docs.pkl"
        
        if not (index_file.exists() and docs_file.exists()):
            logger.warning(f"BM25 index not found at {self.index_path}")
            return False
        
        try:
            with open(index_file, "rb") as f:
                self.bm25 = pickle.load(f)
            
            with open(docs_file, "rb") as f:
                self.documents = pickle.load(f)
            
            self._is_loaded = True
            logger.info(f"BM25 index loaded: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Retrieve documents using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult sorted by BM25 score
        """
        if not self._is_loaded or self.bm25 is None:
            logger.warning("BM25 index not loaded, attempting to load...")
            if not self.load_index():
                logger.error("No BM25 index available")
                return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Build results
        results = []
        max_score = max(scores) if scores.max() > 0 else 1.0
        
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            
            doc = self.documents[idx]
            # Normalize score to 0-1 range for consistency with dense retrieval
            normalized_score = scores[idx] / max_score
            
            result = SearchResult(
                chunk_id=doc.chunk_id,
                text=doc.text,
                score=normalized_score,
                metadata=doc.metadata,
            )
            results.append(result)
        
        return results
    
    @property
    def count(self) -> int:
        """Number of documents in index."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear the index."""
        self.bm25 = None
        self.documents = []
        self._is_loaded = False
        
        # Remove persisted files
        if self.index_path.exists():
            for f in self.index_path.glob("bm25_*.pkl"):
                f.unlink()
            logger.info("BM25 index cleared")
