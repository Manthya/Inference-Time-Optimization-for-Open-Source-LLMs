"""
Metrics collection, aggregation, and analysis.
Measures performance and quality metrics across all stages.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from scipy import stats


@dataclass
class StageMetrics:
    """Aggregated metrics for a single stage."""
    stage_name: str
    
    # Latency metrics (in seconds)
    avg_ttft: float
    p50_ttft: float
    p99_ttft: float
    
    avg_per_token_latency: float
    p50_per_token_latency: float
    p99_per_token_latency: float
    
    avg_total_time: float
    
    # Throughput metrics
    avg_tokens_per_second: float
    total_tokens_generated: int
    
    # Memory metrics
    avg_memory_mb: float
    avg_gpu_memory_mb: float
    
    # Quality metrics
    num_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collect and analyze metrics from inference results."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def aggregate_stage_metrics(self, 
                                results: List[Any],
                                stage_name: str) -> StageMetrics:
        """
        Aggregate metrics from multiple inference results.
        
        Args:
            results: List of InferenceResult objects
            stage_name: Name of the stage
            
        Returns:
            Aggregated StageMetrics
        """
        if not results:
            raise ValueError("No results to aggregate")
        
        # Extract metrics
        ttfts = [r.time_to_first_token for r in results]
        total_times = [r.total_time for r in results]
        tokens_per_sec = [r.tokens_per_second for r in results]
        memory_usage = [r.memory_used_mb for r in results]
        gpu_memory = [r.gpu_memory_mb for r in results if r.gpu_memory_mb is not None]
        
        # Per-token latencies (flatten all)
        all_per_token_latencies = []
        for r in results:
            if r.per_token_latency:
                all_per_token_latencies.extend(r.per_token_latency)
        
        total_tokens = sum(r.tokens_generated for r in results)
        
        # Calculate aggregate metrics
        metrics = StageMetrics(
            stage_name=stage_name,
            
            # TTFT
            avg_ttft=np.mean(ttfts),
            p50_ttft=np.percentile(ttfts, 50),
            p99_ttft=np.percentile(ttfts, 99),
            
            # Per-token latency
            avg_per_token_latency=np.mean(all_per_token_latencies) if all_per_token_latencies else 0,
            p50_per_token_latency=np.percentile(all_per_token_latencies, 50) if all_per_token_latencies else 0,
            p99_per_token_latency=np.percentile(all_per_token_latencies, 99) if all_per_token_latencies else 0,
            
            # Total time
            avg_total_time=np.mean(total_times),
            
            # Throughput
            avg_tokens_per_second=np.mean(tokens_per_sec),
            total_tokens_generated=total_tokens,
            
            # Memory
            avg_memory_mb=np.mean(memory_usage),
            avg_gpu_memory_mb=np.mean(gpu_memory) if gpu_memory else 0,
            
            # Sample count
            num_samples=len(results),
        )
        
        return metrics
    
    def compare_stages(self, 
                      stage_metrics: List[StageMetrics]) -> pd.DataFrame:
        """
        Compare metrics across stages.
        
        Returns:
            DataFrame with comparison
        """
        if not stage_metrics:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([m.to_dict() for m in stage_metrics])
        
        # Calculate speedups relative to baseline
        baseline = df.iloc[0]
        
        df['ttft_speedup'] = baseline['avg_ttft'] / df['avg_ttft']
        df['throughput_speedup'] = df['avg_tokens_per_second'] / baseline['avg_tokens_per_second']
        df['latency_improvement'] = (baseline['avg_per_token_latency'] - df['avg_per_token_latency']) / baseline['avg_per_token_latency'] * 100
        
        return df
    
    def save_results(self, 
                    stage_metrics: List[StageMetrics],
                    comparison_df: pd.DataFrame,
                    experiment_name: str = "experiment"):
        """Save results to disk."""
        
        # Save individual stage metrics
        for metrics in stage_metrics:
            filename = f"{experiment_name}_{metrics.stage_name}_metrics.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            print(f"Saved metrics: {filepath}")
        
        # Save comparison
        comparison_file = self.output_dir / f"{experiment_name}_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Saved comparison: {comparison_file}")
        
        # Save human-readable summary
        summary_file = self.output_dir / f"{experiment_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Experiment Summary: {experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            
            for metrics in stage_metrics:
                f.write(f"\n{metrics.stage_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Avg TTFT: {metrics.avg_ttft*1000:.2f} ms\n")
                f.write(f"  Avg Per-Token Latency: {metrics.avg_per_token_latency*1000:.2f} ms\n")
                f.write(f"  Throughput: {metrics.avg_tokens_per_second:.2f} tokens/s\n")
                f.write(f"  Total Tokens: {metrics.total_tokens_generated}\n")
                f.write(f"  GPU Memory: {metrics.avg_gpu_memory_mb:.2f} MB\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Speedup Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(comparison_df[['stage_name', 'ttft_speedup', 'throughput_speedup', 'latency_improvement']].to_string(index=False))
        
        print(f"Saved summary: {summary_file}")


class QualityEvaluator:
    """
    Evaluate quality metrics to ensure optimizations don't degrade output.
    """
    
    @staticmethod
    def exact_match(output1: str, output2: str) -> bool:
        """Check if two outputs match exactly."""
        return output1.strip() == output2.strip()
    
    @staticmethod
    def token_agreement(output1: str, output2: str) -> float:
        """
        Calculate token-level agreement.
        
        Returns:
            Agreement ratio (0-1)
        """
        tokens1 = output1.split()
        tokens2 = output2.split()
        
        # Calculate Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    @staticmethod
    def edit_distance(output1: str, output2: str) -> int:
        """
        Calculate Levenshtein edit distance.
        
        Returns:
            Edit distance
        """
        m, n = len(output1), len(output2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if output1[i-1] == output2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def compare_outputs(baseline_outputs: List[str],
                       stage_outputs: List[str],
                       stage_name: str) -> Dict[str, Any]:
        """
        Compare outputs from a stage against baseline.
        
        Returns:
            Dictionary of quality metrics
        """
        if len(baseline_outputs) != len(stage_outputs):
            raise ValueError("Output lists must have same length")
        
        exact_matches = sum(
            QualityEvaluator.exact_match(b, s) 
            for b, s in zip(baseline_outputs, stage_outputs)
        )
        
        token_agreements = [
            QualityEvaluator.token_agreement(b, s)
            for b, s in zip(baseline_outputs, stage_outputs)
        ]
        
        edit_distances = [
            QualityEvaluator.edit_distance(b[:500], s[:500])  # Limit for performance
            for b, s in zip(baseline_outputs, stage_outputs)
        ]
        
        return {
            'stage_name': stage_name,
            'exact_match_rate': exact_matches / len(baseline_outputs),
            'avg_token_agreement': np.mean(token_agreements),
            'avg_edit_distance': np.mean(edit_distances),
            'num_samples': len(baseline_outputs),
        }
