"""Document processing and chunking for RAG"""

import re
from typing import List, Optional
from dataclasses import dataclass, field

from src.data.loader import Document
from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """A text chunk ready for embedding and retrieval"""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class TextCleaner:
    """Clean and normalize text"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Clean text while preserving structure"""
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
        text = re.sub(r' +\n', '\n', text)       # Remove trailing spaces
        
        # Remove page markers if needed (optional)
        # text = re.sub(r'\[Page \d+\]', '', text)
        
        return text.strip()
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """Remove common header/footer patterns"""
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            # Skip page numbers
            if re.match(r'^\s*\d+\s*$', line):
                continue
            # Skip common footer patterns
            if re.match(r'^\s*(page|©|copyright)', line.lower()):
                continue
            cleaned.append(line)
        
        return '\n'.join(cleaned)


class TextChunker:
    """Split text into chunks with overlap"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size (skip smaller chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Separators in order of preference
        self.separators = [
            "\n\n",     # Paragraph break
            "\n",       # Line break
            ". ",       # Sentence end
            "? ",       # Question end
            "! ",       # Exclamation end
            "; ",       # Semicolon
            ", ",       # Comma
            " ",        # Word break
        ]
    
    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        chunks = self._recursive_split(text, self.separators)
        
        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)
        
        # Add overlap
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        if not separators:
            # No separators left, force split at chunk_size
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk_size
            if len(current_chunk) + len(split) + len(separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If split itself is too large, recursively split it
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
            else:
                current_chunk = current_chunk + separator + split if current_chunk else split
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small"""
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            if len(current) < self.min_chunk_size:
                current = current + "\n\n" + chunk
            else:
                merged.append(current)
                current = chunk
        
        if current:
            merged.append(current)
        
        return merged
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # Find a clean break point (sentence or word boundary)
            clean_break = overlap_text.rfind('. ')
            if clean_break == -1:
                clean_break = overlap_text.rfind(' ')
            
            if clean_break > 0:
                overlap_text = overlap_text[clean_break + 1:].strip()
            
            # Prepend overlap to current chunk
            if overlap_text:
                overlapped.append(f"...{overlap_text}\n\n{curr_chunk}")
            else:
                overlapped.append(curr_chunk)
        
        return overlapped


class DocumentProcessor:
    """Process documents into chunks ready for embedding"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        clean_text: bool = True,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.cleaner = TextCleaner()
        self.clean_text = clean_text
        self._chunk_counter = 0
    
    def process(self, document: Document) -> List[Chunk]:
        """Process a single document into chunks"""
        text = document.content
        
        # Clean text
        if self.clean_text:
            text = self.cleaner.clean(text)
        
        # Split into chunks
        text_chunks = self.chunker.chunk(text)
        
        # Create Chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            self._chunk_counter += 1
            chunk = Chunk(
                chunk_id=f"chunk_{self._chunk_counter}",
                text=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {document.metadata.get('filename', 'unknown')}")
        return chunks
    
    def process_documents(self, documents: List[Document]) -> List[Chunk]:
        """Process multiple documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.process(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Total: {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


class MedicalTextEnricher:
    """Enrich chunks with medical domain metadata"""
    
    # Medical term categories
    MEDICAL_PATTERNS = {
        "spirometry": [r"fev1", r"fvc", r"spirometry", r"lung\s+function"],
        "medications": [r"laba", r"lama", r"ics", r"inhaler", r"bronchodilator", r"corticosteroid"],
        "symptoms": [r"dyspnea", r"cough", r"sputum", r"wheeze", r"breathless"],
        "severity": [r"gold\s+\d", r"mild", r"moderate", r"severe", r"very\s+severe"],
        "exacerbation": [r"exacerbation", r"acute", r"flare", r"hospitalization"],
    }
    
    @classmethod
    def enrich(cls, chunks: List[Chunk]) -> List[Chunk]:
        """Add medical metadata to chunks"""
        for chunk in chunks:
            text_lower = chunk.text.lower()
            
            # Detect medical categories
            categories = []
            for category, patterns in cls.MEDICAL_PATTERNS.items():
                if any(re.search(p, text_lower) for p in patterns):
                    categories.append(category)
            
            chunk.metadata["medical_categories"] = categories
            
            # Check for measurements
            has_measurements = bool(re.search(r'\d+\s*%|\d+\s*ml|\d+\s*mg', text_lower))
            chunk.metadata["has_measurements"] = has_measurements
            
            # COPD relevance score (simple heuristic)
            copd_terms = ["copd", "chronic obstructive", "pulmonary disease", "emphysema", "chronic bronchitis"]
            relevance = sum(1 for term in copd_terms if term in text_lower)
            chunk.metadata["copd_relevance"] = relevance
        
        return chunks
