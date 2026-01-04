"""Document loaders for various file formats"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Document:
    """Represents a loaded document"""
    content: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")
    
    @property
    def page_count(self) -> int:
        return self.metadata.get("page_count", 1)


class PDFLoader:
    """Load PDF documents using pdfplumber"""
    
    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables
    
    def load(self, file_path: str | Path) -> Document:
        """Load a PDF file and extract text"""
        import pdfplumber
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        pages_text = []
        tables_text = []
        
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            pdf_metadata = pdf.metadata or {}
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(f"[Page {page_num}]\n{text}")
                
                # Extract tables
                if self.extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = self._table_to_text(table)
                        if table_text:
                            tables_text.append(f"[Table on Page {page_num}]\n{table_text}")
        
        # Combine all content
        content_parts = pages_text + tables_text
        content = "\n\n".join(content_parts)
        
        return Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "pdf",
                "page_count": page_count,
                "title": pdf_metadata.get("Title", ""),
                "author": pdf_metadata.get("Author", ""),
            }
        )
    
    def _table_to_text(self, table: List[List]) -> str:
        """Convert table to readable text format"""
        if not table:
            return ""
        
        rows = []
        for row in table:
            # Clean cells and join
            cells = [str(cell).strip() if cell else "" for cell in row]
            if any(cells):  # Skip empty rows
                rows.append(" | ".join(cells))
        
        return "\n".join(rows)


class DocxLoader:
    """Load Word documents"""
    
    def load(self, file_path: str | Path) -> Document:
        """Load a DOCX file and extract text"""
        from docx import Document as DocxDocument
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX not found: {file_path}")
        
        doc = DocxDocument(file_path)
        
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        
        # Extract tables
        tables_text = []
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    table_rows.append(" | ".join(cells))
            if table_rows:
                tables_text.append("\n".join(table_rows))
        
        content = "\n\n".join(paragraphs + tables_text)
        
        return Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": "docx",
                "page_count": 1,  # DOCX doesn't have explicit pages
            }
        )


class TextLoader:
    """Load plain text files"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load(self, file_path: str | Path) -> Document:
        """Load a text file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = file_path.read_text(encoding=self.encoding)
        
        return Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "file_type": file_path.suffix.lstrip("."),
                "page_count": 1,
            }
        )


class DocumentLoader:
    """Unified document loader - auto-detects file type"""
    
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
    
    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.docx_loader = DocxLoader()
        self.text_loader = TextLoader()
    
    def load(self, file_path: str | Path) -> Document:
        """Load a document based on file extension"""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            return self.pdf_loader.load(file_path)
        elif ext in {".docx", ".doc"}:
            return self.docx_loader.load(file_path)
        elif ext in {".txt", ".md"}:
            return self.text_loader.load(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def load_directory(
        self, 
        directory: str | Path, 
        recursive: bool = True
    ) -> List[Document]:
        """Load all supported documents from a directory"""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load(file_path)
                    documents.append(doc)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
