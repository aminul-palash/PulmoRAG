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
    
    def __post_init__(self):
        if self.expected_sources is None:
            self.expected_sources = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "category": self.category,
            "expected_sources": self.expected_sources,
            "ground_truth": self.ground_truth,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestQuery":
        return cls(
            query=data["query"],
            category=data.get("category", "general"),
            expected_sources=data.get("expected_sources", []),
            ground_truth=data.get("ground_truth"),
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


def create_default_copd_dataset() -> TestDataset:
    """
    Create default COPD test dataset with sample queries.
    
    Returns:
        TestDataset with COPD-related queries
    """
    queries = [
        # Treatment questions
        TestQuery(
            query="What are the main treatments for COPD?",
            category="treatment",
        ),
        TestQuery(
            query="What inhalers are used for COPD management?",
            category="treatment",
        ),
        TestQuery(
            query="When should corticosteroids be used in COPD?",
            category="treatment",
        ),
        TestQuery(
            query="What is the role of bronchodilators in COPD treatment?",
            category="treatment",
        ),
        
        # Medication-specific questions
        TestQuery(
            query="What is dupilumab and how does it help COPD patients?",
            category="medication",
        ),
        TestQuery(
            query="What are LAMA and LABA medications?",
            category="medication",
        ),
        TestQuery(
            query="What is the recommended dose of tiotropium for COPD?",
            category="medication",
        ),
        
        # Diagnosis questions
        TestQuery(
            query="How is COPD diagnosed?",
            category="diagnosis",
        ),
        TestQuery(
            query="What is the GOLD classification for COPD?",
            category="diagnosis",
        ),
        TestQuery(
            query="What FEV1 values indicate COPD severity?",
            category="diagnosis",
        ),
        
        # Exacerbation questions
        TestQuery(
            query="How should COPD exacerbations be managed?",
            category="exacerbation",
        ),
        TestQuery(
            query="What are the signs of a COPD exacerbation?",
            category="exacerbation",
        ),
        TestQuery(
            query="When should a COPD patient be hospitalized?",
            category="exacerbation",
        ),
        
        # Lifestyle questions
        TestQuery(
            query="What lifestyle changes help manage COPD?",
            category="lifestyle",
        ),
        TestQuery(
            query="Is pulmonary rehabilitation effective for COPD?",
            category="lifestyle",
        ),
    ]
    
    dataset = TestDataset(queries=queries)
    logger.info(f"Created default COPD dataset with {len(queries)} queries")
    
    return dataset
