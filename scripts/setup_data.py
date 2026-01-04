#!/usr/bin/env python
"""
Data Setup Script for PulmoRAG
Processes documents from data/raw/ and creates chunks for RAG
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import setup_logger
from src.config import get_config
from src.data.loader import DocumentLoader
from src.data.processor import DocumentProcessor, MedicalTextEnricher
from src.data.utils import (
    save_chunks_jsonl,
    save_processing_manifest,
    get_chunks_statistics,
)

logger = setup_logger(__name__)


def setup_directories() -> None:
    """Create necessary directories"""
    dirs = [
        "data/raw/guidelines",
        "data/raw/research",
        "data/raw/patient_education",
        "data/processed",
        "logs",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified: {dir_path}")


def process_documents(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> None:
    """
    Process all documents in input directory and save chunks.
    
    Args:
        input_dir: Directory containing raw documents
        output_dir: Directory to save processed chunks
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check for documents
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return
    
    # Load documents
    logger.info(f"Loading documents from {input_path}...")
    loader = DocumentLoader()
    documents = loader.load_directory(input_path, recursive=True)
    
    if not documents:
        logger.warning("No documents found! Add documents to data/raw/ first.")
        logger.info("Supported formats: .pdf, .docx, .txt, .md")
        return
    
    # Process into chunks
    logger.info(f"Processing {len(documents)} documents...")
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = processor.process_documents(documents)
    
    # Enrich with medical metadata
    logger.info("Enriching chunks with medical metadata...")
    chunks = MedicalTextEnricher.enrich(chunks)
    
    # Save chunks
    output_path.mkdir(parents=True, exist_ok=True)
    chunks_file = output_path / "chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_file)
    
    # Save manifest
    save_processing_manifest(documents, chunks, output_path)
    
    # Print statistics
    stats = get_chunks_statistics(chunks)
    logger.info("Processing complete!")
    logger.info(f"  Documents: {stats['total_chunks']} chunks from {stats['unique_sources']} files")
    logger.info(f"  Avg chunk length: {stats['character_length']['avg']} chars")
    
    if stats.get("medical_categories"):
        logger.info(f"  Medical categories: {stats['medical_categories']}")


def main():
    parser = argparse.ArgumentParser(description="Process documents for PulmoRAG")
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory containing documents (default: data/raw)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory for processed chunks (default: data/processed)"
    )
    parser.add_argument(
        "--chunk-size", "-s",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap", "-l",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only create directories, don't process"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("PulmoRAG Data Setup")
        logger.info("=" * 40)
        
        # Always ensure directories exist
        setup_directories()
        
        if args.setup_only:
            logger.info("Directory setup complete!")
            return
        
        # Process documents
        process_documents(
            input_dir=args.input,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
