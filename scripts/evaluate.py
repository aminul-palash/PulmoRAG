#!/usr/bin/env python3
"""
RAG Evaluation Script

Run evaluation on the RAG pipeline with different retrieval methods.

Usage:
    python scripts/evaluate.py                     # Run full evaluation
    python scripts/evaluate.py --method hybrid     # Evaluate specific method
    python scripts/evaluate.py --compare           # Compare all methods
    python scripts/evaluate.py --quick             # Quick eval (5 queries)
    python scripts/evaluate.py --use-openai        # Use OpenAI as judge
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import setup_logger
from src.config import get_config
from src.evaluation.dataset import TestDataset, create_default_copd_dataset
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.metrics import LocalLLMJudge, OpenAIJudge

logger = setup_logger(__name__)
config = get_config()


def ensure_dataset_exists() -> TestDataset:
    """Ensure test dataset exists, create if not"""
    dataset_path = Path("data/evaluation/test_queries.json")
    
    if dataset_path.exists():
        dataset = TestDataset.load(str(dataset_path))
    else:
        logger.info("Creating default COPD test dataset...")
        dataset = create_default_copd_dataset()
        dataset.save(str(dataset_path))
    
    return dataset


def run_single_method_eval(
    evaluator: RAGEvaluator,
    queries: List[str],
    method: str
) -> Dict[str, Any]:
    """
    Run evaluation for a single retrieval method.
    
    Args:
        evaluator: RAGEvaluator instance
        queries: List of test queries
        method: Retrieval method name
        
    Returns:
        Evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating method: {method}")
    logger.info(f"{'='*60}")
    
    # Use compare_retrieval_methods with single method
    reports = evaluator.compare_retrieval_methods(
        queries=queries,
        methods=[method],
    )
    
    if method in reports:
        report = reports[method]
        print(report.summary())
        return report
    else:
        logger.error(f"Method {method} not found in results")
        return None


def run_comparison(
    evaluator: RAGEvaluator,
    queries: List[str],
) -> Dict[str, Any]:
    """
    Compare all retrieval methods.
    
    Args:
        evaluator: RAGEvaluator instance
        queries: List of test queries
        
    Returns:
        Comparison results
    """
    logger.info("\n" + "="*60)
    logger.info("Running Retrieval Method Comparison")
    logger.info("="*60)
    
    methods = ["dense", "hybrid", "reranked"]
    
    comparison = evaluator.compare_retrieval_methods(
        queries=queries,
        methods=methods,
    )
    
    # Print comparison table
    print("\n" + "="*70)
    print("RETRIEVAL METHOD COMPARISON")
    print("="*70)
    print(f"{'Method':<20} {'Faithfulness':<15} {'Ans Rel':<15} {'Ctx Rel':<15}")
    print("-"*70)
    
    for method, report in comparison.items():
        if hasattr(report, 'avg_faithfulness'):
            print(f"{method:<20} {report.avg_faithfulness:<15.3f} "
                  f"{report.avg_answer_relevancy:<15.3f} "
                  f"{report.avg_context_relevancy:<15.3f}")
    
    print("="*70)
    
    return comparison


def generate_html_report(
    reports: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Generate HTML report with visualizations.
    
    Args:
        reports: Dict of method -> EvaluationReport
        output_path: Output HTML path
        
    Returns:
        Path to generated report
    """
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .summary-cards {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; }}
        .card h3 {{ margin-top: 0; color: #007bff; }}
        .metric {{ font-size: 2em; font-weight: bold; color: #333; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .good {{ color: #28a745; }}
        .medium {{ color: #ffc107; }}
        .poor {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 RAG Evaluation Report</h1>
        <p>Generated: {timestamp}</p>
        
        <div class="summary-cards">
            {summary_cards}
        </div>
        
        <div class="chart-container">
            <h2>📊 Metrics Comparison</h2>
            <canvas id="metricsChart" width="600" height="300"></canvas>
        </div>
        
        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Queries</th>
                    <th>Faithfulness</th>
                    <th>Answer Relevancy</th>
                    <th>Context Relevancy</th>
                    <th>Avg Time (s)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    
    <script>
        const ctx = document.getElementById('metricsChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {chart_labels},
                datasets: [
                    {{
                        label: 'Faithfulness',
                        data: {faithfulness_data},
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    }},
                    {{
                        label: 'Answer Relevancy',
                        data: {answer_rel_data},
                        backgroundColor: 'rgba(75, 192, 192, 0.8)',
                    }},
                    {{
                        label: 'Context Relevancy',
                        data: {context_rel_data},
                        backgroundColor: 'rgba(255, 159, 64, 0.8)',
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
    """
    
    # Build summary cards
    summary_cards = ""
    for method, report in reports.items():
        if hasattr(report, 'avg_faithfulness'):
            avg_score = (report.avg_faithfulness + report.avg_answer_relevancy + report.avg_context_relevancy) / 3
            score_class = "good" if avg_score > 0.7 else ("medium" if avg_score > 0.4 else "poor")
            summary_cards += f"""
            <div class="card">
                <h3>{method}</h3>
                <div class="metric {score_class}">{avg_score:.2f}</div>
                <p>Average Score</p>
            </div>
            """
    
    # Build table rows
    table_rows = ""
    chart_labels = []
    faithfulness_data = []
    answer_rel_data = []
    context_rel_data = []
    
    for method, report in reports.items():
        if hasattr(report, 'avg_faithfulness'):
            chart_labels.append(method)
            faithfulness_data.append(round(report.avg_faithfulness, 3))
            answer_rel_data.append(round(report.avg_answer_relevancy, 3))
            context_rel_data.append(round(report.avg_context_relevancy, 3))
            
            table_rows += f"""
            <tr>
                <td><strong>{method}</strong></td>
                <td>{len(report.results)}</td>
                <td>{report.avg_faithfulness:.3f}</td>
                <td>{report.avg_answer_relevancy:.3f}</td>
                <td>{report.avg_context_relevancy:.3f}</td>
                <td>{report.avg_latency:.2f}</td>
            </tr>
            """
    
    # Fill template
    html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary_cards=summary_cards,
        table_rows=table_rows,
        chart_labels=json.dumps(chart_labels),
        faithfulness_data=json.dumps(faithfulness_data),
        answer_rel_data=json.dumps(answer_rel_data),
        context_rel_data=json.dumps(context_rel_data),
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(html)
    
    logger.info(f"HTML report saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--method",
        choices=["dense", "hybrid", "reranked"],
        default="reranked",
        help="Retrieval method to evaluate (default: reranked)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all retrieval methods"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with 5 queries"
    )
    
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI as LLM judge (requires OPENAI_API_KEY)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/reports",
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report with visualizations"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        help="Filter queries by category (treatment, medication, diagnosis, etc.)"
    )
    
    args = parser.parse_args()
    
    # Ensure dataset exists
    dataset = ensure_dataset_exists()
    
    # Get queries
    if args.category:
        queries = dataset.get_queries(category=args.category)
        logger.info(f"Using {len(queries)} queries from category: {args.category}")
    else:
        queries = dataset.get_queries()
        logger.info(f"Using all {len(queries)} queries")
    
    if args.quick:
        queries = queries[:5]
        logger.info(f"Quick mode: using first 5 queries")
    
    if not queries:
        logger.error("No queries found!")
        return 1
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    
    judge = None
    if args.use_openai:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set!")
            return 1
        judge = OpenAIJudge(api_key=api_key)
        logger.info("Using OpenAI as LLM judge")
    else:
        judge = LocalLLMJudge()
        logger.info("Using local LLM as judge")
    
    evaluator = RAGEvaluator(judge=judge)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run evaluation
    start_time = time.time()
    
    if args.compare:
        reports = run_comparison(evaluator, queries)
    else:
        report = run_single_method_eval(evaluator, queries, args.method)
        reports = {args.method: report}
    
    elapsed = time.time() - start_time
    logger.info(f"\nTotal evaluation time: {elapsed:.1f}s")
    
    # Save reports
    for method, report in reports.items():
        if hasattr(report, 'to_dict'):
            json_path = output_dir / f"eval_{method}_{timestamp}.json"
            evaluator.save_report(report, str(json_path))
    
    # Generate HTML report
    if args.html:
        html_path = output_dir / f"report_{timestamp}.html"
        generate_html_report(reports, str(html_path))
        print(f"\n📊 HTML report: {html_path}")
    
    print(f"\n✅ Evaluation complete! Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
