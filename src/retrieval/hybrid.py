"""Hybrid retrieval combining dense and sparse search with optional reranking"""

from typing import List, Optional, Literal

from src.logger import setup_logger
from src.config import get_config
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion, linear_combination
from src.retrieval.vectorstore import SearchResult

logger = setup_logger(__name__)

FusionMethod = Literal["rrf", "weighted"]


class HybridRetriever:
    """
    Hybrid retrieval combining dense (semantic) and sparse (BM25) search.
    
    Optionally applies cross-encoder reranking for improved precision.
    
    Architecture:
        Query → ┬→ Dense Retriever → Semantic Results ─┐
                │                                       ├→ Fusion → Reranker → Final Results
                └→ Sparse Retriever → BM25 Results ────┘
    """
    
    def __init__(
        self,
        collection_name: str = "pulmonary_documents",
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion_method: FusionMethod = "rrf",
        dense_weight: float = 0.7,
        use_reranker: Optional[bool] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            collection_name: ChromaDB collection name
            dense_retriever: Pre-configured dense retriever (creates new if None)
            sparse_retriever: Pre-configured sparse retriever (creates new if None)
            fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted'
            dense_weight: Weight for dense results when using weighted fusion (0-1)
            use_reranker: Enable cross-encoder reranking (default from config)
        """
        config = get_config()
        
        # Initialize retrievers
        self.dense = dense_retriever or DenseRetriever(collection_name=collection_name)
        self.sparse = sparse_retriever or SparseRetriever()
        
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        
        # Check if sparse index is available
        self._sparse_available = self.sparse.index_exists()
        if not self._sparse_available:
            logger.warning("BM25 index not found. Run 'build_vector_db.py build' to create it.")
        
        # Initialize reranker (lazy loading)
        self._use_reranker = use_reranker if use_reranker is not None else config.USE_RERANKER
        self._reranker = None
        self._rerank_candidates = config.RERANK_CANDIDATES
        
        logger.info(f"HybridRetriever initialized (fusion={fusion_method}, sparse_ready={self._sparse_available}, reranker={self._use_reranker})")
    
    def _get_reranker(self):
        """Lazy load the reranker."""
        if self._reranker is None and self._use_reranker:
            from src.retrieval.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Retrieve documents using hybrid search with optional reranking.
        
        Args:
            query: Search query
            top_k: Final number of results
            dense_k: Number of dense results to retrieve
            sparse_k: Number of sparse results to retrieve
            use_reranker: Override reranker setting for this query
            
        Returns:
            List of SearchResult sorted by relevance
        """
        # Determine if we should rerank
        should_rerank = use_reranker if use_reranker is not None else self._use_reranker
        
        # If reranking, retrieve more candidates
        if should_rerank:
            candidates_k = self._rerank_candidates
            dense_k = dense_k or candidates_k
            sparse_k = sparse_k or candidates_k
        else:
            dense_k = dense_k or top_k * 2
            sparse_k = sparse_k or top_k * 2
        
        # Dense retrieval (always available)
        dense_results = self.dense.retrieve(query, top_k=dense_k)
        logger.debug(f"Dense: {len(dense_results)} results")
        
        # Sparse retrieval (may not be available)
        if self._sparse_available:
            sparse_results = self.sparse.retrieve(query, top_k=sparse_k)
            logger.debug(f"Sparse: {len(sparse_results)} results")
        else:
            # Fallback to dense-only
            logger.debug("Sparse not available, using dense only")
            fused = dense_results
            if should_rerank and fused:
                reranker = self._get_reranker()
                if reranker:
                    return reranker.rerank(query, fused, top_k=top_k)
            return fused[:top_k]
        
        # Fusion - get more candidates if reranking
        fusion_top_k = self._rerank_candidates if should_rerank else top_k
        
        if self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=60,
                top_k=fusion_top_k,
            )
        else:  # weighted
            fused = linear_combination(
                dense_results,
                sparse_results,
                alpha=self.dense_weight,
                top_k=fusion_top_k,
            )
        
        # Apply reranking if enabled
        if should_rerank and fused:
            reranker = self._get_reranker()
            if reranker:
                fused = reranker.rerank(query, fused, top_k=top_k)
                logger.info(f"Hybrid+Rerank: {len(dense_results)} dense + {len(sparse_results)} sparse → {fusion_top_k} fused → {len(fused)} reranked")
                return fused
        
        logger.info(f"Hybrid retrieval: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused)} fused")
        return fused[:top_k]
    
    def retrieve_dense_only(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Retrieve using only dense (semantic) search."""
        return self.dense.retrieve(query, top_k=top_k)
    
    def retrieve_sparse_only(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Retrieve using only sparse (BM25) search."""
        if not self._sparse_available:
            logger.error("BM25 index not available")
            return []
        return self.sparse.retrieve(query, top_k=top_k)
    
    def compare_methods(
        self,
        query: str,
        top_k: int = 5,
        include_reranked: bool = True,
    ) -> dict:
        """
        Compare dense, sparse, hybrid, and reranked results for analysis.
        
        Returns:
            Dict with 'dense', 'sparse', 'hybrid', 'reranked' result lists
        """
        dense_results = self.dense.retrieve(query, top_k=top_k)
        
        if self._sparse_available:
            sparse_results = self.sparse.retrieve(query, top_k=top_k)
            # Get hybrid without reranking for fair comparison
            hybrid_results = self.retrieve(query, top_k=top_k, use_reranker=False)
        else:
            sparse_results = []
            hybrid_results = dense_results
        
        result = {
            "dense": dense_results,
            "sparse": sparse_results,
            "hybrid": hybrid_results,
            "query": query,
        }
        
        # Add reranked results if available
        if include_reranked and self._use_reranker:
            reranked_results = self.retrieve(query, top_k=top_k, use_reranker=True)
            result["reranked"] = reranked_results
        
        return result
    
    @property
    def is_hybrid_ready(self) -> bool:
        """Check if both dense and sparse are available."""
        return self._sparse_available
    
    @property
    def is_reranker_enabled(self) -> bool:
        """Check if reranker is enabled."""
        return self._use_reranker
