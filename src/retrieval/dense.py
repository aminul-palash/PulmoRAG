"""Dense retrieval using vector similarity search"""

from typing import List, Optional, Dict, Any

from src.logger import setup_logger
from src.retrieval.vectorstore import ChromaVectorStore, SearchResult
from src.retrieval.embeddings import BaseEmbedder

logger = setup_logger(__name__)


class DenseRetriever:
    """Dense retrieval using vector embeddings"""
    
    def __init__(
        self,
        vectorstore: Optional[ChromaVectorStore] = None,
        collection_name: str = "pulmonary_documents",
        embedder: Optional[BaseEmbedder] = None,
    ):
        """
        Initialize the dense retriever.
        
        Args:
            vectorstore: Pre-configured vector store (creates new if None)
            collection_name: Collection name (used if vectorstore is None)
            embedder: Embedder instance (used if vectorstore is None)
        """
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = ChromaVectorStore(
                collection_name=collection_name,
                embedder=embedder,
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results sorted by relevance
        """
        results = self.vectorstore.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve and format results as context string for LLM.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("filename", "Unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{result.text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    @property
    def document_count(self) -> int:
        """Get number of indexed documents"""
        return self.vectorstore.count
