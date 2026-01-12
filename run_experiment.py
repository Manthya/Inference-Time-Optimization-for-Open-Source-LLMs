"""
Main experiment runner.
Orchestrates the entire inference optimization experiment.
"""
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch

from stages import Stage0Baseline, Stage1vLLM, Stage2FusedvLLM, InferenceResult
from evaluation import DatasetLoader, EvalSample
from metrics import MetricsCollector, QualityEvaluator, MetricsVisualizer


class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, config_path: str = "./configs/experiment_config.yaml"):
        """Initialize experiment runner."""
        self.config = self._load_config(config_path)
        self.dataset_loader = DatasetLoader()
        self.metrics_collector = MetricsCollector(
            output_dir=self.config['output']['results_dir']
        )
        self.visualizer = MetricsVisualizer(
            output_dir=self.config['output']['plots_dir']
        )
        
        # Storage for results
        self.all_stage_results = {}
        self.all_stage_outputs = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    
    def run_stage(self, 
                  stage_class,
                  stage_config: Dict[str, Any],
                  eval_samples: List[EvalSample]) -> List[InferenceResult]:
        """
        Run inference for a single stage.
        
        Args:
            stage_class: Stage class to instantiate
            stage_config: Configuration for the stage
            eval_samples: List of evaluation samples
            
        Returns:
            List of InferenceResult objects
        """
        stage_name = stage_config['name']
        print(f"\n{'='*80}")
        print(f"Running {stage_name}")
        print(f"{'='*80}")
        
        # Initialize stage
        stage = stage_class(
            model_name=self.config['model']['name'],
            precision=self.config['model']['precision'],
        )
        
        # Load model
        start_time = time.time()
        stage.load_model()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Run inference on all samples
        results = []
        outputs = []
        
        batch_size = stage_config.get('batch_size', 1)
        num_samples = len(eval_samples)
        
        print(f"Running inference on {num_samples} samples (batch_size={batch_size})...")
        
        for i in range(0, num_samples, batch_size):
            batch = eval_samples[i:i+batch_size]
            prompts = [sample.prompt for sample in batch]
            
            # Generate
            if batch_size == 1:
                result = stage.generate(
                    prompts[0],
                    max_tokens=self.config['decoding']['max_tokens'],
                    temperature=self.config['decoding']['temperature']
                )
                batch_results = [result]
            else:
                batch_results = stage.batch_generate(
                    prompts,
                    max_tokens=self.config['decoding']['max_tokens'],
                    temperature=self.config['decoding']['temperature']
                )
            
            results.extend(batch_results)
            outputs.extend([r.generated_text for r in batch_results])
            
            # Progress
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i+batch_size, num_samples)}/{num_samples} samples...")
        
        print(f"Completed {stage_name}")
        
        # Cleanup
        stage.cleanup()
        
        return results, outputs
    
    def run_all_stages(self) -> None:
        """Run all enabled stages in the experiment."""
        
        # Load evaluation datasets
        print("\n" + "="*80)
        print("Loading Evaluation Datasets")
        print("="*80)
        eval_samples = self.dataset_loader.load_all_datasets(self.config)
        
        # Stage mapping
        stage_mapping = {
            'stage_0': (Stage0Baseline, self.config['stages']['stage_0']),
            'stage_1': (Stage1vLLM, self.config['stages']['stage_1']),
            'stage_2': (Stage2FusedvLLM, self.config['stages']['stage_2']),
        }
        
        # Run each enabled stage
        for stage_id, (stage_class, stage_config) in stage_mapping.items():
            if not stage_config.get('enabled', False):
                print(f"\nSkipping {stage_id} (disabled)")
                continue
            
            try:
                results, outputs = self.run_stage(stage_class, stage_config, eval_samples)
                self.all_stage_results[stage_config['name']] = results
                self.all_stage_outputs[stage_config['name']] = outputs
            except Exception as e:
                print(f"\nERROR in {stage_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n" + "="*80)
        print("All stages completed!")
        print("="*80)
    
    def analyze_results(self) -> None:
        """Analyze and compare results across stages."""
        
        print("\n" + "="*80)
        print("Analyzing Results")
        print("="*80)
        
        # Collect metrics for each stage
        stage_metrics = []
        for stage_name, results in self.all_stage_results.items():
            metrics = self.metrics_collector.aggregate_stage_metrics(
                results, stage_name
            )
            stage_metrics.append(metrics)
            
            print(f"\n{stage_name}:")
            print(f"  Avg TTFT: {metrics.avg_ttft*1000:.2f} ms")
            print(f"  Avg Per-Token Latency: {metrics.avg_per_token_latency*1000:.2f} ms")
            print(f"  Throughput: {metrics.avg_tokens_per_second:.2f} tokens/s")
            print(f"  Total Tokens: {metrics.total_tokens_generated}")
        
        # Compare stages
        comparison_df = self.metrics_collector.compare_stages(stage_metrics)
        
        print("\n" + "="*80)
        print("Stage Comparison")
        print("="*80)
        print(comparison_df[['stage_name', 'avg_tokens_per_second', 'throughput_speedup']].to_string(index=False))
        
        # Quality evaluation (compare against baseline)
        if len(self.all_stage_outputs) > 1:
            print("\n" + "="*80)
            print("Quality Evaluation")
            print("="*80)
            
            baseline_name = list(self.all_stage_outputs.keys())[0]
            baseline_outputs = self.all_stage_outputs[baseline_name]
            
            for stage_name, stage_outputs in self.all_stage_outputs.items():
                if stage_name == baseline_name:
                    continue
                
                quality_metrics = QualityEvaluator.compare_outputs(
                    baseline_outputs, stage_outputs, stage_name
                )
                
                print(f"\n{stage_name} vs {baseline_name}:")
                print(f"  Exact Match Rate: {quality_metrics['exact_match_rate']*100:.1f}%")
                print(f"  Avg Token Agreement: {quality_metrics['avg_token_agreement']*100:.1f}%")
                print(f"  Avg Edit Distance: {quality_metrics['avg_edit_distance']:.1f}")
        
        # Save results
        experiment_name = self.config['experiment_name']
        self.metrics_collector.save_results(stage_metrics, comparison_df, experiment_name)
        
        # Create visualizations
        self.visualizer.create_all_plots(comparison_df, experiment_name)
        
        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)
    
    def run(self) -> None:
        """Run the complete experiment."""
        print("\n" + "="*80)
        print(f"Starting Experiment: {self.config['experiment_name']}")
        print("="*80)
        print(f"Model: {self.config['model']['name']}")
        print(f"Precision: {self.config['model']['precision']}")
        print(f"Decoding: Temperature={self.config['decoding']['temperature']} (Greedy)")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("\nWARNING: No GPU detected. This experiment requires a CUDA GPU.")
            return
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Run experiment
        start_time = time.time()
        
        try:
            self.run_all_stages()
            self.analyze_results()
        except Exception as e:
            print(f"\nExperiment failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print(f"Experiment Complete! Total time: {total_time/60:.1f} minutes")
        print("="*80)
        print(f"\nResults saved to: {self.config['output']['results_dir']}")
        print(f"Plots saved to: {self.config['output']['plots_dir']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference-time optimization experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/experiment_config.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        choices=['stage_0', 'stage_1', 'stage_2'],
        help='Specific stages to run (default: all enabled in config)'
    )
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = ExperimentRunner(config_path=args.config)
    
    # Override stage selection if specified
    if args.stages:
        for stage_id in ['stage_0', 'stage_1', 'stage_2']:
            runner.config['stages'][stage_id]['enabled'] = stage_id in args.stages
    
    runner.run()


if __name__ == "__main__":
    main()
