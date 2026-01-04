"""Data utilities for document processing"""

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.data.loader import Document
from src.data.processor import Chunk
from src.logger import setup_logger

logger = setup_logger(__name__)


def save_chunks_jsonl(chunks: List[Chunk], output_path: str | Path) -> None:
    """Save chunks to JSONL format (one JSON per line)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_jsonl(input_path: str | Path) -> List[Chunk]:
    """Load chunks from JSONL format"""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {input_path}")
    
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                chunk = Chunk(
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    metadata=record.get("metadata", {}),
                )
                chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


def save_processing_manifest(
    documents: List[Document],
    chunks: List[Chunk],
    output_dir: str | Path,
) -> None:
    """Save a manifest of processed documents"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "processed_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "documents": [
            {
                "source": doc.source,
                "filename": doc.metadata.get("filename", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "page_count": doc.page_count,
            }
            for doc in documents
        ],
        "chunk_stats": {
            "total": len(chunks),
            "avg_length": sum(len(c.text) for c in chunks) // max(len(chunks), 1),
            "min_length": min(len(c.text) for c in chunks) if chunks else 0,
            "max_length": max(len(c.text) for c in chunks) if chunks else 0,
        },
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved manifest to {manifest_path}")


def get_chunk_by_id(chunks: List[Chunk], chunk_id: str) -> Optional[Chunk]:
    """Find a chunk by its ID"""
    for chunk in chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    return None


def filter_chunks_by_source(chunks: List[Chunk], source_pattern: str) -> List[Chunk]:
    """Filter chunks by source filename pattern"""
    return [c for c in chunks if source_pattern.lower() in c.source.lower()]


def get_chunks_statistics(chunks: List[Chunk]) -> dict:
    """Get statistics about chunks"""
    if not chunks:
        return {"total": 0}
    
    lengths = [len(c.text) for c in chunks]
    word_counts = [c.word_count for c in chunks]
    
    sources = set(c.source for c in chunks)
    
    # Medical category distribution
    categories = {}
    for chunk in chunks:
        for cat in chunk.metadata.get("medical_categories", []):
            categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_chunks": len(chunks),
        "unique_sources": len(sources),
        "character_length": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) // len(lengths),
        },
        "word_count": {
            "min": min(word_counts),
            "max": max(word_counts),
            "avg": sum(word_counts) // len(word_counts),
        },
        "medical_categories": categories,
    }


def print_chunk_preview(chunk: Chunk, max_length: int = 200) -> None:
    """Print a preview of a chunk"""
    text_preview = chunk.text[:max_length] + "..." if len(chunk.text) > max_length else chunk.text
    
    print(f"\n{'='*60}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Source: {chunk.metadata.get('filename', 'unknown')}")
    print(f"Words: {chunk.word_count}")
    print(f"Medical: {chunk.metadata.get('medical_categories', [])}")
    print(f"{'='*60}")
    print(text_preview)
