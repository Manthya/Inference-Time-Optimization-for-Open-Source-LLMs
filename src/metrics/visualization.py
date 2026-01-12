"""
Visualization utilities for experiment results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class MetricsVisualizer:
    """Create visualizations for experiment results."""
    
    def __init__(self, output_dir: str = "./results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_throughput_comparison(self, 
                                   comparison_df: pd.DataFrame,
                                   experiment_name: str = "experiment"):
        """Plot throughput comparison across stages."""
        fig, ax = plt.subplots()
        
        stages = comparison_df['stage_name']
        throughput = comparison_df['avg_tokens_per_second']
        
        bars = ax.bar(range(len(stages)), throughput, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title('Throughput Comparison Across Stages', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=15, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        filepath = self.output_dir / f"{experiment_name}_throughput.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filepath}")
    
    def plot_latency_comparison(self,
                               comparison_df: pd.DataFrame,
                               experiment_name: str = "experiment"):
        """Plot latency comparison across stages."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stages = comparison_df['stage_name']
        
        # TTFT
        ttft = comparison_df['avg_ttft'] * 1000  # Convert to ms
        ax1.bar(range(len(stages)), ttft, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_xlabel('Stage', fontsize=11)
        ax1.set_ylabel('Time to First Token (ms)', fontsize=11)
        ax1.set_title('TTFT Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels(stages, rotation=15, ha='right')
        
        # Per-token latency
        per_token = comparison_df['avg_per_token_latency'] * 1000  # Convert to ms
        ax2.bar(range(len(stages)), per_token, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_xlabel('Stage', fontsize=11)
        ax2.set_ylabel('Per-Token Latency (ms)', fontsize=11)
        ax2.set_title('Per-Token Latency Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels(stages, rotation=15, ha='right')
        
        plt.tight_layout()
        filepath = self.output_dir / f"{experiment_name}_latency.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filepath}")
    
    def plot_speedup_comparison(self,
                               comparison_df: pd.DataFrame,
                               experiment_name: str = "experiment"):
        """Plot speedup relative to baseline."""
        fig, ax = plt.subplots()
        
        stages = comparison_df['stage_name'][1:]  # Skip baseline
        speedup = comparison_df['throughput_speedup'][1:]
        
        bars = ax.bar(range(len(stages)), speedup, color=['#ff7f0e', '#2ca02c'])
        ax.axhline(y=1, color='r', linestyle='--', label='Baseline')
        
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Speedup vs Baseline', fontsize=12)
        ax.set_title('Throughput Speedup Relative to Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=15, ha='right')
        ax.legend()
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}×',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        filepath = self.output_dir / f"{experiment_name}_speedup.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filepath}")
    
    def plot_memory_usage(self,
                         comparison_df: pd.DataFrame,
                         experiment_name: str = "experiment"):
        """Plot GPU memory usage across stages."""
        fig, ax = plt.subplots()
        
        stages = comparison_df['stage_name']
        memory = comparison_df['avg_gpu_memory_mb'] / 1024  # Convert to GB
        
        bars = ax.bar(range(len(stages)), memory, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('GPU Memory Usage (GB)', fontsize=12)
        ax.set_title('GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=15, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} GB',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        filepath = self.output_dir / f"{experiment_name}_memory.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filepath}")
    
    def create_all_plots(self,
                        comparison_df: pd.DataFrame,
                        experiment_name: str = "experiment"):
        """Create all visualization plots."""
        print("\nGenerating visualizations...")
        
        self.plot_throughput_comparison(comparison_df, experiment_name)
        self.plot_latency_comparison(comparison_df, experiment_name)
        self.plot_speedup_comparison(comparison_df, experiment_name)
        self.plot_memory_usage(comparison_df, experiment_name)
        
        print("All visualizations created!")
