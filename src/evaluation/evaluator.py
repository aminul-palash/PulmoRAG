"""RAG Evaluation Orchestrator

Main evaluator class that coordinates metrics evaluation
and generates reports.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from src.logger import setup_logger
from src.config import get_config
from src.evaluation.metrics import (
    get_judge,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextRelevancyMetric,
)

logger = setup_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single query"""
    query: str
    context: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    faithfulness_reasoning: str = ""
    answer_relevancy_reasoning: str = ""
    context_relevancy_reasoning: str = ""
    retrieval_method: str = ""
    latency_ms: float = 0.0
    
    @property
    def average_score(self) -> float:
        """Average of all metrics"""
        return (self.faithfulness + self.answer_relevancy + self.context_relevancy) / 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "context": self.context[:200] + "..." if len(self.context) > 200 else self.context,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
            "average_score": self.average_score,
            "retrieval_method": self.retrieval_method,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report"""
    results: List[EvaluationResult]
    retrieval_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def num_queries(self) -> int:
        return len(self.results)
    
    @property
    def avg_faithfulness(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.faithfulness for r in self.results) / len(self.results)
    
    @property
    def avg_answer_relevancy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.answer_relevancy for r in self.results) / len(self.results)
    
    @property
    def avg_context_relevancy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.context_relevancy for r in self.results) / len(self.results)
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)
    
    @property
    def overall_score(self) -> float:
        return (self.avg_faithfulness + self.avg_answer_relevancy + self.avg_context_relevancy) / 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_method": self.retrieval_method,
            "timestamp": self.timestamp,
            "num_queries": self.num_queries,
            "metrics": {
                "faithfulness": round(self.avg_faithfulness, 4),
                "answer_relevancy": round(self.avg_answer_relevancy, 4),
                "context_relevancy": round(self.avg_context_relevancy, 4),
                "overall": round(self.overall_score, 4),
            },
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }
    
    def summary(self) -> str:
        """Generate text summary"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  RAG Evaluation Report - {self.retrieval_method.upper():^20}         ║
╠══════════════════════════════════════════════════════════════╣
║  Queries Evaluated: {self.num_queries:<10}                           ║
║  Average Latency:   {self.avg_latency_ms:.0f}ms                              ║
╠══════════════════════════════════════════════════════════════╣
║  METRICS                                                     ║
║  ────────────────────────────────────────────────────────    ║
║  Faithfulness:       {self.avg_faithfulness:.2%}                              ║
║  Answer Relevancy:   {self.avg_answer_relevancy:.2%}                              ║
║  Context Relevancy:  {self.avg_context_relevancy:.2%}                              ║
║  ────────────────────────────────────────────────────────    ║
║  OVERALL SCORE:      {self.overall_score:.2%}                              ║
╚══════════════════════════════════════════════════════════════╝
"""


class RAGEvaluator:
    """
    Main RAG evaluation orchestrator.
    
    Evaluates RAG pipeline using:
    - Faithfulness: Is answer grounded in context?
    - Answer Relevancy: Does answer address the query?
    - Context Relevancy: Is retrieved context relevant?
    """
    
    def __init__(
        self,
        judge=None,
        use_local_llm: bool = True,
        llm=None,
    ):
        """
        Initialize evaluator.
        
        Args:
            judge: Pre-initialized LLM judge (takes precedence)
            use_local_llm: Use local LLM as judge (vs OpenAI)
            llm: Pre-loaded LLM instance
        """
        self.use_local_llm = use_local_llm
        self._judge = judge
        self._llm = llm
        self._metrics_initialized = False
        
        logger.info(f"RAGEvaluator initialized (local_llm={use_local_llm})")
    
    def _init_metrics(self):
        """Lazy initialization of metrics"""
        if self._metrics_initialized:
            return
        
        if self._judge is None:
            self._judge = get_judge(use_local=self.use_local_llm, llm=self._llm)
        self.faithfulness = FaithfulnessMetric(self._judge)
        self.answer_relevancy = AnswerRelevancyMetric(self._judge)
        self.context_relevancy = ContextRelevancyMetric(self._judge)
        self._metrics_initialized = True
    
    def evaluate_single(
        self,
        query: str,
        context: str,
        answer: str,
        retrieval_method: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a single query-context-answer tuple.
        
        Args:
            query: User query
            context: Retrieved context
            answer: Generated answer
            retrieval_method: Name of retrieval method used
            
        Returns:
            EvaluationResult with all metrics
        """
        self._init_metrics()
        
        start_time = time.time()
        
        # Evaluate all metrics
        faith_result = self.faithfulness.evaluate(context, answer)
        relevancy_result = self.answer_relevancy.evaluate(query, answer)
        context_result = self.context_relevancy.evaluate(query, context)
        
        latency = (time.time() - start_time) * 1000
        
        result = EvaluationResult(
            query=query,
            context=context,
            answer=answer,
            faithfulness=faith_result["score"],
            answer_relevancy=relevancy_result["score"],
            context_relevancy=context_result["score"],
            faithfulness_reasoning=faith_result.get("reasoning", ""),
            answer_relevancy_reasoning=relevancy_result.get("reasoning", ""),
            context_relevancy_reasoning=context_result.get("reasoning", ""),
            retrieval_method=retrieval_method,
            latency_ms=latency,
        )
        
        logger.info(
            f"Evaluated: faith={result.faithfulness:.2f}, "
            f"ans_rel={result.answer_relevancy:.2f}, "
            f"ctx_rel={result.context_relevancy:.2f}"
        )
        
        return result
    
    def evaluate_rag_pipeline(
        self,
        queries: List[str],
        rag_pipeline,
        retrieval_method: str = "default",
    ) -> EvaluationReport:
        """
        Evaluate RAG pipeline on a list of queries.
        
        Args:
            queries: List of test queries
            rag_pipeline: RAGPipeline instance
            retrieval_method: Name of retrieval method
            
        Returns:
            EvaluationReport with aggregated metrics
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Evaluating query {i}/{len(queries)}: {query[:50]}...")
            
            try:
                # Run RAG pipeline
                rag_response = rag_pipeline.run(query, stream=False)
                
                # Evaluate
                result = self.evaluate_single(
                    query=query,
                    context=rag_response.context_used,
                    answer=rag_response.answer,
                    retrieval_method=retrieval_method,
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate query '{query}': {e}")
                continue
        
        report = EvaluationReport(results=results, retrieval_method=retrieval_method)
        return report
    
    def compare_retrieval_methods(
        self,
        queries: List[str],
        methods: List[str] = ["dense", "hybrid", "reranked"],
        collection_name: str = "pulmonary_documents",
    ) -> Dict[str, EvaluationReport]:
        """
        Compare different retrieval methods.
        
        Args:
            queries: List of test queries
            methods: List of retrieval methods to compare
            collection_name: ChromaDB collection name
            
        Returns:
            Dict mapping method name to EvaluationReport
        """
        from src.rag import RAGPipeline
        from src.generation.llm import LLMFactory
        
        # Get LLM for generation
        llm = LLMFactory.get_llm(prefer_local=True)
        if llm is None:
            raise RuntimeError("No LLM available for RAG pipeline")
        
        reports = {}
        
        for method in methods:
            logger.info(f"Evaluating retrieval method: {method}")
            
            # Configure pipeline based on method
            if method == "dense":
                pipeline = RAGPipeline(
                    collection_name=collection_name,
                    llm=llm,
                    use_hybrid=False,
                    use_reranker=False,
                )
            elif method == "hybrid":
                pipeline = RAGPipeline(
                    collection_name=collection_name,
                    llm=llm,
                    use_hybrid=True,
                    use_reranker=False,
                )
            elif method == "reranked":
                pipeline = RAGPipeline(
                    collection_name=collection_name,
                    llm=llm,
                    use_hybrid=True,
                    use_reranker=True,
                )
            else:
                logger.warning(f"Unknown method: {method}, skipping")
                continue
            
            report = self.evaluate_rag_pipeline(queries, pipeline, method)
            reports[method] = report
            
            print(report.summary())
        
        return reports
    
    def evaluate_by_disease(
        self,
        dataset,  # TestDataset
        method: str = "reranked",
        collection_name: str = "pulmonary_documents",
    ) -> Dict[str, EvaluationReport]:
        """
        Evaluate RAG pipeline grouped by disease category.
        
        Args:
            dataset: TestDataset with disease-labeled queries
            method: Retrieval method to use
            collection_name: ChromaDB collection name
            
        Returns:
            Dict mapping disease name to EvaluationReport
        """
        from src.rag import RAGPipeline
        from src.generation.llm import LLMFactory
        
        # Get LLM for generation
        llm = LLMFactory.get_llm(prefer_local=True)
        if llm is None:
            raise RuntimeError("No LLM available for RAG pipeline")
        
        # Configure pipeline
        if method == "dense":
            pipeline = RAGPipeline(
                collection_name=collection_name,
                llm=llm,
                use_hybrid=False,
                use_reranker=False,
            )
        elif method == "hybrid":
            pipeline = RAGPipeline(
                collection_name=collection_name,
                llm=llm,
                use_hybrid=True,
                use_reranker=False,
            )
        else:  # reranked
            pipeline = RAGPipeline(
                collection_name=collection_name,
                llm=llm,
                use_hybrid=True,
                use_reranker=True,
            )
        
        # Group queries by disease
        disease_groups = dataset.get_by_disease()
        disease_reports = {}
        
        for disease, queries in disease_groups.items():
            logger.info(f"\nEvaluating {disease} ({len(queries)} queries)...")
            query_strings = [q.query for q in queries]
            
            report = self.evaluate_rag_pipeline(query_strings, pipeline, f"{method}_{disease}")
            disease_reports[disease] = report
            
            print(f"\n{disease}: Faithfulness={report.avg_faithfulness:.2%}, "
                  f"Answer Rel={report.avg_answer_relevancy:.2%}, "
                  f"Context Rel={report.avg_context_relevancy:.2%}")
        
        return disease_reports
    
    def generate_disease_summary_table(
        self,
        disease_reports: Dict[str, 'EvaluationReport'],
    ) -> str:
        """
        Generate a summary table suitable for research paper.
        
        Args:
            disease_reports: Dict mapping disease to EvaluationReport
            
        Returns:
            Formatted table string
        """
        header = f"\n{'Disease':<25} {'Queries':>8} {'Faithful':>10} {'Ans Rel':>10} {'Ctx Rel':>10} {'Overall':>10}"
        separator = "-" * 83
        
        rows = [header, separator]
        
        total_queries = 0
        total_faith = 0
        total_ans = 0
        total_ctx = 0
        
        for disease, report in sorted(disease_reports.items()):
            n = report.num_queries
            total_queries += n
            total_faith += report.avg_faithfulness * n
            total_ans += report.avg_answer_relevancy * n
            total_ctx += report.avg_context_relevancy * n
            
            overall = (report.avg_faithfulness + report.avg_answer_relevancy + report.avg_context_relevancy) / 3
            rows.append(
                f"{disease:<25} {n:>8} {report.avg_faithfulness:>10.2%} "
                f"{report.avg_answer_relevancy:>10.2%} {report.avg_context_relevancy:>10.2%} "
                f"{overall:>10.2%}"
            )
        
        rows.append(separator)
        
        # Overall averages
        if total_queries > 0:
            avg_faith = total_faith / total_queries
            avg_ans = total_ans / total_queries
            avg_ctx = total_ctx / total_queries
            avg_overall = (avg_faith + avg_ans + avg_ctx) / 3
            rows.append(
                f"{'OVERALL':<25} {total_queries:>8} {avg_faith:>10.2%} "
                f"{avg_ans:>10.2%} {avg_ctx:>10.2%} {avg_overall:>10.2%}"
            )
        
        return "\n".join(rows)
    
    def save_report(
        self,
        report: EvaluationReport,
        output_dir: str = "data/evaluation/reports",
    ) -> str:
        """
        Save evaluation report to JSON file.
        
        Args:
            report: EvaluationReport to save
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{report.retrieval_method}_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Report saved to {filepath}")
        return str(filepath)
