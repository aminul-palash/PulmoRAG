"""
RAG Evaluation Module

RAGAS-style evaluation metrics for RAG pipelines:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the query?
- Context Relevancy: Is the retrieved context relevant?
"""

from src.logger import setup_logger
from src.evaluation.metrics import (
    BaseLLMJudge,
    LocalLLMJudge,
    OpenAIJudge,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextRelevancyMetric,
)
from src.evaluation.evaluator import (
    EvaluationResult,
    EvaluationReport,
    RAGEvaluator,
)
from src.evaluation.dataset import (
    TestQuery,
    TestDataset,
    create_default_copd_dataset,
)

logger = setup_logger(__name__)

__all__ = [
    # LLM Judges
    "BaseLLMJudge",
    "LocalLLMJudge",
    "OpenAIJudge",
    # Metrics
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextRelevancyMetric",
    # Evaluator
    "EvaluationResult",
    "EvaluationReport",
    "RAGEvaluator",
    # Dataset
    "TestQuery",
    "TestDataset",
    "create_default_copd_dataset",
]
