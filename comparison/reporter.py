from typing import List, Dict, Any
from tabulate import tabulate
import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)

class Reporter:
    def generate_console_report(self, query: str, analysis: Dict[str, Any]) -> str:
        report = []
        report.append("┌─────────────────────────────────────────────────────────────────────┐")
        report.append("│                  CHUNKING STRATEGY COMPARISON REPORT                │")
        report.append("└─────────────────────────────────────────────────────────────────────┘")
        report.append(f"\nQuery: \"{query}\"\n")
        report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        if 'random' in analysis:
            random = analysis['random']
            report.append("RANDOM CHUNKING RESULTS:")
            report.append(f"├─ Chunks Generated: {random['num_chunks']}")
            report.append(f"├─ Avg Chunk Size: {random['avg_chunk_size']:.0f} chars")
            report.append(f"├─ Processing Time: {random['processing_time']:.2f}s")
            report.append(f"├─ Top-5 Accuracy: {random['top_5_accuracy']:.0f}%")
            report.append(f"├─ Avg Score: {random['avg_score']:.2f}")
            report.append("└─ Issues: ")
            report.append("   • Function split across chunks")
            report.append("   • Missing context for complete understanding\n")

        if 'ast' in analysis:
            ast = analysis['ast']
            report.append("CASTS (AST-BASED) RESULTS:")
            report.append(f"├─ Chunks Generated: {ast['num_chunks']}")
            report.append(f"├─ Avg Chunk Size: {ast['avg_chunk_size']:.0f} chars")
            report.append(f"├─ Processing Time: {ast['processing_time']:.2f}s (with reranking)")
            report.append(f"├─ Top-5 Accuracy: {ast['top_5_accuracy']:.0f}%")
            report.append(f"├─ Avg Score: {ast['avg_score']:.2f}")
            report.append("└─ Advantages:")
            report.append("   • Complete functions preserved")
            report.append("   • Better semantic relevance")
            report.append("   • Reduced chunks to search\n")

        if 'improvement' in analysis:
            imp = analysis['improvement']
            report.append("RERANKING IMPACT (cAST only):")
            report.append(f"├─ Accuracy Improvement: +{imp['accuracy_improvement']:.1f}%")
            report.append(f"├─ Score Improvement: +{imp['score_improvement']:.1f}%")
            report.append(f"├─ Chunk Reduction: {imp['chunk_reduction']:.1f}%")
            report.append("└─ Processing Time: Additional ~0.12s (simulated)\n")

        report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(report)

    def generate_full_report(self, full_results: List[Dict[str, Any]]) -> None:
        for item in full_results:
            print(self.generate_console_report(item['query'], item['analysis']))

    def export_to_csv(self, full_results: List[Dict[str, Any]], filename: str = 'comparison_report.csv') -> None:
        data = []
        for item in full_results:
            query = item['query']
            analysis = item['analysis']
            for method in ['random', 'ast']:
                if method in analysis:
                    metrics = analysis[method]
                    data.append({
                        'query': query,
                        'method': method,
                        'num_chunks': metrics['num_chunks'],
                        'avg_chunk_size': metrics['avg_chunk_size'],
                        'processing_time': metrics['processing_time'],
                        'top_5_accuracy': metrics['top_5_accuracy'],
                        'avg_score': metrics['avg_score'],
                        'completeness': metrics['completeness']
                    })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported to {filename}")

    def export_to_json(self, full_results: List[Dict[str, Any]], filename: str = 'comparison_report.json') -> None:
        with open(filename, 'w') as f:
            json.dump(full_results, f, indent=4)
        logger.info(f"Exported to {filename}")
