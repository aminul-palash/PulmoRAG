"""Test Dataset Management for RAG Evaluation"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)


@dataclass
class TestQuery:
    """A single test query with optional ground truth"""
    query: str
    category: str = "general"
    expected_sources: List[str] = None
    ground_truth: str = None  # Optional ground truth answer
    id: int = None  # Unique query identifier
    disease: str = None  # Disease category (COPD, Asthma, etc.)
    
    def __post_init__(self):
        if self.expected_sources is None:
            self.expected_sources = []
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "query": self.query,
            "category": self.category,
            "expected_sources": self.expected_sources,
            "ground_truth": self.ground_truth,
        }
        if self.id is not None:
            result["id"] = self.id
        if self.disease is not None:
            result["disease"] = self.disease
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestQuery":
        return cls(
            query=data["query"],
            category=data.get("category", "general"),
            expected_sources=data.get("expected_sources", []),
            ground_truth=data.get("ground_truth"),
            id=data.get("id"),
            disease=data.get("disease"),
        )


class TestDataset:
    """Manages test queries for evaluation"""
    
    DEFAULT_PATH = "data/evaluation/test_queries.json"
    
    def __init__(self, queries: List[TestQuery] = None):
        """
        Initialize dataset.
        
        Args:
            queries: List of TestQuery objects
        """
        self.queries = queries or []
    
    def add_query(self, query: TestQuery) -> None:
        """Add a query to the dataset"""
        self.queries.append(query)
    
    def get_queries(self, category: Optional[str] = None) -> List[str]:
        """
        Get query strings, optionally filtered by category.
        
        Args:
            category: Filter by category (None = all)
            
        Returns:
            List of query strings
        """
        if category:
            return [q.query for q in self.queries if q.category == category]
        return [q.query for q in self.queries]
    
    def get_by_category(self) -> Dict[str, List[TestQuery]]:
        """Group queries by category"""
        result = {}
        for q in self.queries:
            if q.category not in result:
                result[q.category] = []
            result[q.category].append(q)
        return result
    
    def get_by_disease(self) -> Dict[str, List[TestQuery]]:
        """Group queries by disease"""
        result = {}
        for q in self.queries:
            disease = q.disease or "Unknown"
            if disease not in result:
                result[disease] = []
            result[disease].append(q)
        return result
    
    def get_queries_by_disease(self, disease: str) -> List[str]:
        """
        Get query strings filtered by disease.
        
        Args:
            disease: Disease name to filter by
            
        Returns:
            List of query strings for that disease
        """
        return [q.query for q in self.queries if q.disease == disease]
    
    def get_diseases(self) -> List[str]:
        """Get list of unique diseases in the dataset"""
        diseases = set(q.disease for q in self.queries if q.disease)
        return sorted(list(diseases))
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories in the dataset"""
        categories = set(q.category for q in self.queries if q.category)
        return sorted(list(categories))
    
    def save(self, path: str = None) -> str:
        """
        Save dataset to JSON file.
        
        Args:
            path: Output path (default: DEFAULT_PATH)
            
        Returns:
            Path to saved file
        """
        path = Path(path or self.DEFAULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "queries": [q.to_dict() for q in self.queries],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Dataset saved to {path} ({len(self.queries)} queries)")
        return str(path)
    
    @classmethod
    def load(cls, path: str = None) -> "TestDataset":
        """
        Load dataset from JSON file.
        
        Args:
            path: Input path (default: DEFAULT_PATH)
            
        Returns:
            TestDataset instance
        """
        path = Path(path or cls.DEFAULT_PATH)
        
        if not path.exists():
            logger.warning(f"Dataset not found at {path}, returning empty dataset")
            return cls()
        
        with open(path, "r") as f:
            data = json.load(f)
        
        queries = [TestQuery.from_dict(q) for q in data.get("queries", [])]
        logger.info(f"Loaded {len(queries)} queries from {path}")
        
        return cls(queries=queries)
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __iter__(self):
        return iter(self.queries)


def create_default_pulmonary_dataset() -> TestDataset:
    """
    Create default pulmonary disease test dataset with sample queries.
    
    Returns:
        TestDataset with pulmonary disease-related queries
    """
    queries = [
        # COPD questions
        TestQuery(
            query="What are the main treatments for COPD?",
            category="copd",
        ),
        TestQuery(
            query="What inhalers are used for COPD management?",
            category="copd",
        ),
        TestQuery(
            query="What is the GOLD classification for COPD?",
            category="copd",
        ),
        TestQuery(
            query="How should COPD exacerbations be managed?",
            category="copd",
        ),
        
        # Asthma questions
        TestQuery(
            query="What are the main treatments for asthma?",
            category="asthma",
        ),
        TestQuery(
            query="What is the difference between COPD and asthma?",
            category="asthma",
        ),
        TestQuery(
            query="How are asthma exacerbations managed?",
            category="asthma",
        ),
        
        # Pneumonia questions
        TestQuery(
            query="What antibiotics are used to treat pneumonia?",
            category="pneumonia",
        ),
        TestQuery(
            query="What are the symptoms of pneumonia?",
            category="pneumonia",
        ),
        
        # Tuberculosis questions
        TestQuery(
            query="What is the standard treatment regimen for tuberculosis?",
            category="tuberculosis",
        ),
        TestQuery(
            query="How is tuberculosis diagnosed?",
            category="tuberculosis",
        ),
        
        # Lung cancer questions
        TestQuery(
            query="What are the treatment options for non-small cell lung cancer?",
            category="lung_cancer",
        ),
        TestQuery(
            query="How is lung cancer staged?",
            category="lung_cancer",
        ),
        
        # Pulmonary fibrosis questions
        TestQuery(
            query="What treatments are available for pulmonary fibrosis?",
            category="pulmonary_fibrosis",
        ),
        
        # General pulmonary questions
        TestQuery(
            query="What is pulmonary rehabilitation and who benefits from it?",
            category="general",
        ),
        TestQuery(
            query="What lifestyle changes help manage chronic lung diseases?",
            category="general",
        ),
    ]
    
    dataset = TestDataset(queries=queries)
    logger.info(f"Created default pulmonary dataset with {len(queries)} queries")
    
    return dataset


# Backward compatibility alias
def create_default_copd_dataset() -> TestDataset:
    """Deprecated: Use create_default_pulmonary_dataset instead."""
    return create_default_pulmonary_dataset()
