"""Fusion algorithms for combining retrieval results"""

from typing import List, Dict
from collections import defaultdict

from src.logger import setup_logger
from src.retrieval.vectorstore import SearchResult

logger = setup_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60,
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    RRF Score = Σ 1 / (k + rank_i) for each result list
    
    Args:
        result_lists: List of SearchResult lists from different retrievers
        k: RRF constant (default 60, as per original paper)
        top_k: Number of final results to return
        
    Returns:
        Fused and re-ranked list of SearchResult
    """
    if not result_lists:
        return []
    
    # Accumulate RRF scores by chunk_id
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, SearchResult] = {}
    
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] += 1.0 / (k + rank)
            
            # Keep the result with highest original score for metadata
            if chunk_id not in doc_map or result.score > doc_map[chunk_id].score:
                doc_map[chunk_id] = result
    
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Build final results with RRF score
    final_results = []
    for chunk_id in sorted_ids[:top_k]:
        original = doc_map[chunk_id]
        # Create new result with RRF score
        result = SearchResult(
            chunk_id=chunk_id,
            text=original.text,
            score=rrf_scores[chunk_id],  # Use RRF score
            metadata=original.metadata,
        )
        final_results.append(result)
    
    logger.debug(f"RRF fusion: {sum(len(r) for r in result_lists)} inputs → {len(final_results)} outputs")
    return final_results


def weighted_fusion(
    result_lists: List[List[SearchResult]],
    weights: List[float],
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Combine results using weighted score fusion.
    
    Final Score = Σ weight_i * normalized_score_i
    
    Args:
        result_lists: List of SearchResult lists
        weights: Weight for each result list (should sum to 1.0)
        top_k: Number of final results
        
    Returns:
        Fused list of SearchResult
    """
    if not result_lists or len(result_lists) != len(weights):
        return []
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Accumulate weighted scores
    weighted_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, SearchResult] = {}
    
    for results, weight in zip(result_lists, weights):
        if not results:
            continue
            
        # Normalize scores within this result list
        max_score = max(r.score for r in results) if results else 1.0
        max_score = max_score if max_score > 0 else 1.0
        
        for result in results:
            chunk_id = result.chunk_id
            normalized = result.score / max_score
            weighted_scores[chunk_id] += weight * normalized
            
            if chunk_id not in doc_map:
                doc_map[chunk_id] = result
    
    # Sort and build results
    sorted_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
    
    final_results = []
    for chunk_id in sorted_ids[:top_k]:
        original = doc_map[chunk_id]
        result = SearchResult(
            chunk_id=chunk_id,
            text=original.text,
            score=weighted_scores[chunk_id],
            metadata=original.metadata,
        )
        final_results.append(result)
    
    return final_results


def linear_combination(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    alpha: float = 0.7,
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Simple linear combination of dense and sparse scores.
    
    Final Score = alpha * dense_score + (1 - alpha) * sparse_score
    
    Args:
        dense_results: Results from dense retriever
        sparse_results: Results from sparse retriever
        alpha: Weight for dense scores (0-1)
        top_k: Number of results
        
    Returns:
        Combined results
    """
    return weighted_fusion(
        [dense_results, sparse_results],
        [alpha, 1 - alpha],
        top_k=top_k,
    )
