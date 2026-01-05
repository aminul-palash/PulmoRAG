"""Vector store implementation using ChromaDB"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.logger import setup_logger
from src.config import get_config
from src.data.processor import Chunk
from src.retrieval.embeddings import BaseEmbedder, get_embedder

logger = setup_logger(__name__)


@dataclass
class SearchResult:
    """A single search result"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    
    @property
    def source(self) -> str:
        return self.metadata.get("filename", "unknown")


class ChromaVectorStore:
    """ChromaDB vector store for document retrieval"""
    
    def __init__(
        self,
        collection_name: str = "pulmonary_documents",
        embedder: Optional[BaseEmbedder] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            embedder: Embedder instance (creates default if None)
            persist_directory: Local persistence path (for embedded mode)
            host: ChromaDB server host (for client-server mode)
            port: ChromaDB server port
        """
        import chromadb
        
        config = get_config()
        
        # Initialize embedder
        self.embedder = embedder or get_embedder("local")
        
        # Initialize ChromaDB client
        if host or config.CHROMA_HOST != "localhost":
            # Client-server mode
            host = host or config.CHROMA_HOST
            port = port or config.CHROMA_PORT
            logger.info(f"Connecting to ChromaDB server: {host}:{port}")
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Persistent embedded mode (default)
            persist_dir = persist_directory or config.CHROMA_PERSIST_DIR
            logger.info(f"Using persistent ChromaDB: {persist_dir}")
            self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        logger.info(f"Collection '{collection_name}' ready ({self.collection.count()} documents)")
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
    ) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunks to add
            batch_size: Batch size for embedding and insertion
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data
            ids = [chunk.chunk_id for chunk in batch]
            texts = [chunk.text for chunk in batch]
            metadatas = [self._prepare_metadata(chunk.metadata) for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedder.embed_texts(texts)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            
            total_added += len(batch)
            logger.info(f"Added {total_added}/{len(chunks)} chunks")
        
        return total_added
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (must be flat dict with simple types)"""
        prepared = {}
        for key, value in metadata.items():
            # ChromaDB only supports str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                prepared[key] = ", ".join(str(v) for v in value)
            else:
                prepared[key] = str(value)
        return prepared
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # For cosine, similarity = 1 - distance
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                search_results.append(result)
        
        return search_results
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def clear(self) -> None:
        """Clear all documents from collection"""
        # Get all IDs and delete them
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
            logger.info(f"Cleared {len(all_ids)} documents from collection")
    
    @property
    def count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
