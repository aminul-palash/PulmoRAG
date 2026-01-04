"""Cross-encoder reranking for improved retrieval precision"""

from typing import List, Optional
import time

from src.logger import setup_logger
from src.config import get_config
from src.retrieval.vectorstore import SearchResult

logger = setup_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improving retrieval precision.
    
    Uses a cross-encoder model that scores (query, document) pairs together,
    capturing semantic interactions that bi-encoders miss.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name (default from config)
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        from sentence_transformers import CrossEncoder
        
        config = get_config()
        self.model_name = model_name or config.RERANKER_MODEL
        
        logger.info(f"Loading reranker model: {self.model_name}")
        start_time = time.time()
        
        # Initialize cross-encoder
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=device,
        )
        
        load_time = time.time() - start_time
        logger.info(f"Reranker loaded in {load_time:.2f}s")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.
        
        Args:
            query: Search query
            results: List of SearchResult from initial retrieval
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of SearchResult (top_k items)
        """
        if not results:
            return []
        
        if len(results) <= top_k:
            # No need to rerank if we have fewer results than requested
            logger.debug(f"Skipping rerank: only {len(results)} results")
            return results
        
        start_time = time.time()
        
        # Create (query, document) pairs for scoring
        pairs = [(query, result.text) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine results with new scores
        scored_results = list(zip(results, scores))
        
        # Sort by cross-encoder score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Build reranked results with new scores
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked_result = SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),  # Use cross-encoder score
                metadata=result.metadata,
            )
            reranked.append(reranked_result)
        
        rerank_time = time.time() - start_time
        logger.info(f"Reranked {len(results)} → {len(reranked)} in {rerank_time*1000:.1f}ms")
        
        return reranked
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair.
        
        Args:
            query: Search query
            document: Document text
            
        Returns:
            Relevance score (higher = more relevant)
        """
        return float(self.model.predict([(query, document)])[0])


# Lazy-loaded singleton for efficiency
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: Optional[str] = None,
    force_new: bool = False,
) -> CrossEncoderReranker:
    """
    Get or create a reranker instance (singleton pattern).
    
    Args:
        model_name: Model name (uses default if None)
        force_new: Force creation of new instance
        
    Returns:
        CrossEncoderReranker instance
    """
    global _reranker_instance
    
    if _reranker_instance is None or force_new:
        _reranker_instance = CrossEncoderReranker(model_name=model_name)
    
    return _reranker_instance
