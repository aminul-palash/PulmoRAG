"""RAG Pipeline with optimized context injection for local LLM"""

from typing import List, Optional, Generator
from dataclasses import dataclass

from src.logger import setup_logger
from src.config import get_config
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.vectorstore import SearchResult

logger = setup_logger(__name__)

__all__ = ["RAGPipeline", "RAGResponse"]


@dataclass
class RAGResponse:
    """RAG pipeline response with sources"""
    answer: str
    sources: List[dict]
    context_used: str
    token_count: int


class RAGPipeline:
    """
    Main RAG pipeline orchestrator with optimized context injection.
    
    Settings:
    - Top K: 3 chunks
    - Score cutoff: 0.5 minimum (for dense-only)
    - Max context: 1500 tokens
    - Format: Source-attributed
    - Retrieval: Hybrid (dense + BM25) by default
    """
    
    # Configuration constants
    TOP_K = 3
    SCORE_CUTOFF = 0.5
    MAX_CONTEXT_TOKENS = 1500
    CHARS_PER_TOKEN = 4  # Approximate for token estimation
    
    def __init__(
        self,
        collection_name: str = "copd_documents",
        llm = None,
        use_hybrid: bool = True,
        use_reranker: Optional[bool] = None,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            collection_name: ChromaDB collection name
            llm: LLM instance (LocalLLM or API-based)
            use_hybrid: Use hybrid retrieval (dense + BM25) if available
            use_reranker: Use cross-encoder reranking (default from config)
        """
        config = get_config()
        use_hybrid = use_hybrid and config.USE_HYBRID_RETRIEVAL
        
        if use_hybrid:
            self.retriever = HybridRetriever(
                collection_name=collection_name,
                fusion_method=config.HYBRID_FUSION_METHOD,
                dense_weight=config.HYBRID_DENSE_WEIGHT,
                use_reranker=use_reranker,
            )
            self._is_hybrid = self.retriever.is_hybrid_ready
            self._is_reranker = self.retriever.is_reranker_enabled
            if not self._is_hybrid:
                logger.warning("Hybrid not ready (BM25 index missing), using dense-only")
        else:
            self.retriever = DenseRetriever(collection_name=collection_name)
            self._is_hybrid = False
            self._is_reranker = False
        
        self.llm = llm
        logger.info(f"RAGPipeline initialized (hybrid={self._is_hybrid}, reranker={self._is_reranker})")
    
    def set_llm(self, llm) -> None:
        """Set the LLM instance."""
        self.llm = llm
    
    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: User query
            top_k: Number of results (default: self.TOP_K)
            
        Returns:
            List of search results
        """
        top_k = top_k or self.TOP_K
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # For dense-only, apply score cutoff
        # For hybrid with RRF, scores are already normalized differently
        if not self._is_hybrid:
            filtered = [r for r in results if r.score >= self.SCORE_CUTOFF]
            logger.info(f"Retrieved {len(results)} chunks, {len(filtered)} above cutoff ({self.SCORE_CUTOFF})")
            return filtered
        
        logger.info(f"Hybrid retrieved {len(results)} chunks")
        return results
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return len(text) // self.CHARS_PER_TOKEN
    
    def _smart_truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        max_chars = max_tokens * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        
        # Truncate at sentence boundary if possible
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.7:  # Keep at least 70%
            return truncated[:last_period + 1] + "..."
        return truncated + "..."
    
    def build_context(self, results: List[SearchResult]) -> tuple[str, List[dict]]:
        """
        Build source-attributed context from search results.
        
        Returns:
            Tuple of (formatted context string, list of source metadata)
        """
        if not results:
            return "", []
        
        context_parts = []
        sources = []
        remaining_tokens = self.MAX_CONTEXT_TOKENS
        
        for i, result in enumerate(results, 1):
            # Calculate available tokens for this chunk
            # Reserve some tokens for formatting
            chunk_budget = remaining_tokens - 50
            if chunk_budget <= 0:
                break
            
            # Smart truncate if needed
            text = self._smart_truncate(result.text, chunk_budget)
            token_count = self._estimate_tokens(text)
            
            # Format with source attribution
            source_name = result.metadata.get("filename", "Unknown source")
            context_parts.append(
                f"[{i}] Source: {source_name}\n{text}"
            )
            
            sources.append({
                "index": i,
                "source": source_name,
                "score": round(result.score, 4),
                "chunk_id": result.chunk_id,
            })
            
            remaining_tokens -= token_count + 20  # Account for formatting
            
            logger.info(f"Chunk {i}: {source_name} (score={result.score:.3f}, ~{token_count} tokens)")
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def build_prompt(self, query: str, context: str) -> str:
        """
        Build the final prompt with context injection.
        
        Args:
            query: User query
            context: Formatted context from retrieval
            
        Returns:
            Complete prompt for LLM
        """
        if not context:
            # No relevant context found
            return f"""You are PulmoRAG, a medical information assistant for COPD.

The user asked: "{query}"

I could not find relevant information in my medical knowledge base for this query.
Please provide a helpful response explaining that:
1. No specific sources were found for this query
2. Suggest rephrasing or asking about specific COPD topics
3. Remind them to consult healthcare professionals

Answer:"""
        
        return f"""You are PulmoRAG, a medical information assistant for COPD (Chronic Obstructive Pulmonary Disease).

Use ONLY the following medical sources to answer the question. Cite sources using [1], [2], etc.
If the sources don't contain enough information, say so honestly.

{context}

Question: {query}

Instructions:
- Answer based on the sources above
- Cite sources with [1], [2], etc.
- Be concise but thorough
- If unsure, acknowledge uncertainty
- Always recommend consulting healthcare professionals for medical decisions

Answer:"""
    
    def run(
        self,
        query: str,
        stream: bool = False,
    ) -> RAGResponse | Generator[str, None, RAGResponse]:
        """
        Run the complete RAG pipeline.
        
        Args:
            query: User query
            stream: Whether to stream the response
            
        Returns:
            RAGResponse with answer and sources, or generator if streaming
        """
        if not self.llm:
            raise ValueError("LLM not set. Call set_llm() first.")
        
        # Step 1: Retrieve relevant chunks
        results = self.retrieve(query)
        
        # Step 2: Build context
        context, sources = self.build_context(results)
        context_tokens = self._estimate_tokens(context)
        
        # Step 3: Build prompt
        prompt = self.build_prompt(query, context)
        
        logger.info(f"Context: {len(sources)} sources, ~{context_tokens} tokens")
        
        # Step 4: Generate response
        if stream:
            return self._stream_response(prompt, sources, context, context_tokens)
        else:
            answer = self.llm.generate(
                prompt,
                max_tokens=512,
                temperature=0.7,
                stream=False,
            )
            return RAGResponse(
                answer=answer,
                sources=sources,
                context_used=context,
                token_count=context_tokens,
            )
    
    def _stream_response(
        self,
        prompt: str,
        sources: List[dict],
        context: str,
        context_tokens: int,
    ) -> Generator[str, None, RAGResponse]:
        """Stream response and return final RAGResponse."""
        full_response = ""
        
        for chunk in self.llm.generate(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stream=True,
        ):
            full_response += chunk
            yield chunk
        
        # Return final response (caller can access via generator.send() or just iterate)
        return RAGResponse(
            answer=full_response,
            sources=sources,
            context_used=context,
            token_count=context_tokens,
        )
