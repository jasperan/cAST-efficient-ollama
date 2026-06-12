from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class Reporter:
    def _render_method(self, label: str, metrics: Dict[str, Any], issues: List[str]) -> List[str]:
        lines = [f"{label} RESULTS:"]
        if "error" in metrics:
            lines.append(f"└─ Status: {metrics['error']}")
            if metrics.get("processing_time") is not None:
                lines.append(f"   Processing Time: {metrics['processing_time']:.2f}s")
            lines.append("")
            return lines

        lines.append(f"├─ Chunks Generated: {metrics['num_chunks']}")
        lines.append(f"├─ Avg Chunk Size: {metrics['avg_chunk_size']:.0f} chars")
        lines.append(f"├─ Processing Time: {metrics['processing_time']:.2f}s")
        lines.append(f"├─ Avg Top-5 Score: {metrics['avg_top5_score']:.0f}")
        lines.append(f"├─ Avg Score: {metrics['avg_score']:.2f}")
        lines.append("└─ Notes:")
        lines.extend(f"   • {issue}" for issue in issues)
        lines.append("")
        return lines

    def generate_console_report(self, query: str, analysis: Dict[str, Any]) -> str:
        report = [
            "┌─────────────────────────────────────────────────────────────────────┐",
            "│                  CHUNKING STRATEGY COMPARISON REPORT                │",
            "└─────────────────────────────────────────────────────────────────────┘",
            f"\nQuery: \"{query}\"\n",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        ]

        if "random" in analysis:
            report.extend(
                self._render_method(
                    "RANDOM CHUNKING",
                    analysis["random"],
                    [
                        "Function boundaries may be split across chunks",
                        "Context can be incomplete for precise retrieval",
                    ],
                )
            )

        if "ast" in analysis:
            report.extend(
                self._render_method(
                    "cAST (AST-BASED)",
                    analysis["ast"],
                    [
                        "Function and class boundaries are preserved",
                        "Semantic retrieval quality improves with structured chunks",
                        "Optional reranking improves top-result relevance",
                    ],
                )
            )

        if "improvement" in analysis:
            improvement = analysis["improvement"]
            report.append("RERANKING IMPACT (cAST only):")
            report.append(f"├─ Avg Top-5 Score Delta: +{improvement['score_delta']:.1f}")
            report.append(f"├─ Score Improvement: +{improvement['score_improvement']:.1f}%")
            report.append(f"├─ Chunk Reduction: {improvement['chunk_reduction']:.1f}%")
            report.append("└─ Processing Time: Additional reranking cost depends on backend\n")

        report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(report)

    def generate_full_report(self, full_results: List[Dict[str, Any]]) -> None:
        for item in full_results:
            print(self.generate_console_report(item["query"], item["analysis"]))

    def export_to_csv(self, full_results: List[Dict[str, Any]], filename: str = "comparison_report.csv") -> None:
        data = []
        for item in full_results:
            query = item["query"]
            analysis = item["analysis"]
            for method in ["random", "ast"]:
                metrics = analysis.get(method)
                if not metrics or "error" in metrics:
                    continue
                data.append(
                    {
                        "query": query,
                        "method": method,
                        "num_chunks": metrics["num_chunks"],
                        "avg_chunk_size": metrics["avg_chunk_size"],
                        "processing_time": metrics["processing_time"],
                        "avg_top5_score": metrics["avg_top5_score"],
                        "avg_score": metrics["avg_score"],
                        "completeness": metrics["completeness"],
                    }
                )

        pd.DataFrame(data).to_csv(filename, index=False)
        logger.info("Exported to %s", filename)

    def export_to_json(self, full_results: List[Dict[str, Any]], filename: str = "comparison_report.json") -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(full_results, handle, indent=4)
        logger.info("Exported to %s", filename)
