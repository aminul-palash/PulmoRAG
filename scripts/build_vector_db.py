#!/usr/bin/env python
"""
Build Vector Database Script
Indexes processed chunks into ChromaDB for retrieval
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import setup_logger
from src.config import get_config
from src.data.utils import load_chunks_jsonl, get_chunks_statistics
from src.retrieval.vectorstore import ChromaVectorStore
from src.retrieval.embeddings import get_embedder
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever

logger = setup_logger(__name__)


def build_vector_db(
    chunks_file: str = "data/processed/chunks.jsonl",
    collection_name: str = "copd_documents",
    embedder_type: str = "local",
    model_name: str = None,
    clear_existing: bool = False,
) -> None:
    """
    Build vector database from processed chunks.
    
    Args:
        chunks_file: Path to chunks JSONL file
        collection_name: ChromaDB collection name
        embedder_type: 'local' or 'openai'
        model_name: Embedding model name
        clear_existing: Whether to clear existing collection
    """
    chunks_path = Path(chunks_file)
    
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.info("Run 'python scripts/setup_data.py' first to process documents")
        return
    
    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}...")
    chunks = load_chunks_jsonl(chunks_path)
    
    if not chunks:
        logger.warning("No chunks found!")
        return
    
    # Print statistics
    stats = get_chunks_statistics(chunks)
    logger.info(f"Loaded {stats['total_chunks']} chunks from {stats['unique_sources']} sources")
    
    # Initialize embedder
    logger.info(f"Initializing {embedder_type} embedder...")
    embedder = get_embedder(embedder_type, model_name)
    
    # Initialize vector store
    logger.info(f"Connecting to ChromaDB collection: {collection_name}")
    vectorstore = ChromaVectorStore(
        collection_name=collection_name,
        embedder=embedder,
    )
    
    # Clear existing if requested
    if clear_existing and vectorstore.count > 0:
        logger.info(f"Clearing existing {vectorstore.count} documents...")
        vectorstore.clear()
    
    # Check if already indexed
    if vectorstore.count > 0:
        logger.info(f"Collection already has {vectorstore.count} documents")
        response = input("Add more? (y/n): ").strip().lower()
        if response != "y":
            logger.info("Aborted")
            return
    
    # Add chunks to vector store
    logger.info("Indexing chunks into ChromaDB (this may take a while)...")
    added = vectorstore.add_chunks(chunks, batch_size=50)
    
    logger.info(f"Successfully indexed {added} chunks in ChromaDB")
    logger.info(f"Total documents in collection: {vectorstore.count}")
    
    # Build BM25 index for hybrid retrieval
    logger.info("Building BM25 index for hybrid retrieval...")
    sparse_retriever = SparseRetriever(load_on_init=False)
    
    # Convert Chunk objects to dicts for BM25
    chunk_dicts = [
        {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
        for c in chunks
    ]
    bm25_count = sparse_retriever.build_index(chunk_dicts, save=True)
    logger.info(f"Successfully indexed {bm25_count} chunks in BM25")


def test_search(
    query: str = "What is COPD treatment?",
    collection_name: str = "copd_documents",
    top_k: int = 3,
    mode: str = "hybrid",
) -> None:
    """Test search functionality"""
    logger.info(f"Testing {mode} search: '{query}'")
    
    if mode == "hybrid":
        retriever = HybridRetriever(collection_name=collection_name)
        if not retriever.is_hybrid_ready:
            logger.warning("BM25 index not found, falling back to dense-only")
        results = retriever.retrieve(query, top_k=top_k)
    elif mode == "dense":
        from src.retrieval.dense import DenseRetriever
        retriever = DenseRetriever(collection_name=collection_name)
        results = retriever.retrieve(query, top_k=top_k)
    elif mode == "sparse":
        retriever = SparseRetriever()
        results = retriever.retrieve(query, top_k=top_k)
    else:
        logger.error(f"Unknown mode: {mode}")
        return
    
    if not results:
        logger.warning("No results found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Mode: {mode}")
    print(f"Found {len(results)} results:")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"    Source: {result.source}")
        print(f"    Text: {result.text[:200]}...")


def compare_retrieval(
    query: str = "What is COPD treatment?",
    collection_name: str = "copd_documents",
    top_k: int = 3,
    include_reranked: bool = True,
) -> None:
    """Compare dense, sparse, hybrid, and reranked retrieval results"""
    logger.info(f"Comparing retrieval methods for: '{query}'")
    
    retriever = HybridRetriever(collection_name=collection_name)
    comparison = retriever.compare_methods(query, top_k=top_k, include_reranked=include_reranked)
    
    methods = ["dense", "sparse", "hybrid"]
    if "reranked" in comparison:
        methods.append("reranked")
    
    for method in methods:
        results = comparison.get(method, [])
        print(f"\n{'='*60}")
        print(f"{method.upper()} Results ({len(results)} found):")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result.score:.4f} - {result.source}")


def main():
    parser = argparse.ArgumentParser(description="Build vector database for PulmoRAG")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build vector database")
    build_parser.add_argument(
        "--chunks", "-c",
        default="data/processed/chunks.jsonl",
        help="Path to chunks file"
    )
    build_parser.add_argument(
        "--collection", "-n",
        default="copd_documents",
        help="Collection name"
    )
    build_parser.add_argument(
        "--embedder", "-e",
        choices=["local", "openai"],
        default="local",
        help="Embedder type"
    )
    build_parser.add_argument(
        "--model", "-m",
        default=None,
        help="Embedding model name"
    )
    build_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection first"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test search")
    test_parser.add_argument(
        "query",
        nargs="?",
        default="What is COPD treatment?",
        help="Search query"
    )
    test_parser.add_argument(
        "--collection", "-n",
        default="copd_documents",
        help="Collection name"
    )
    test_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of results"
    )
    test_parser.add_argument(
        "--mode", "-m",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="Retrieval mode"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare retrieval methods")
    compare_parser.add_argument(
        "query",
        nargs="?",
        default="What is COPD treatment?",
        help="Search query"
    )
    compare_parser.add_argument(
        "--collection", "-n",
        default="copd_documents",
        help="Collection name"
    )
    compare_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of results"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "build":
            build_vector_db(
                chunks_file=args.chunks,
                collection_name=args.collection,
                embedder_type=args.embedder,
                model_name=args.model,
                clear_existing=args.clear,
            )
        elif args.command == "test":
            test_search(
                query=args.query,
                collection_name=args.collection,
                top_k=args.top_k,
                mode=args.mode,
            )
        elif args.command == "compare":
            compare_retrieval(
                query=args.query,
                collection_name=args.collection,
                top_k=args.top_k,
            )
        else:
            # Default: build
            build_vector_db()
            
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
