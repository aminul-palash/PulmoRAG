"""Data loading and processing module"""

from src.data.loader import Document, DocumentLoader, PDFLoader, DocxLoader, TextLoader
from src.data.processor import Chunk, DocumentProcessor, TextChunker, MedicalTextEnricher
from src.data.utils import (
    save_chunks_jsonl,
    load_chunks_jsonl,
    save_processing_manifest,
    get_chunks_statistics,
)

__all__ = [
    # Loader
    "Document",
    "DocumentLoader",
    "PDFLoader",
    "DocxLoader",
    "TextLoader",
    # Processor
    "Chunk",
    "DocumentProcessor",
    "TextChunker",
    "MedicalTextEnricher",
    # Utils
    "save_chunks_jsonl",
    "load_chunks_jsonl",
    "save_processing_manifest",
    "get_chunks_statistics",
]
