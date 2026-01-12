"""
Main experiment runner.
Orchestrates the entire inference optimization experiment.
"""
import yaml
import time
import argparse
import random
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
from tqdm import tqdm

from src.stages import Stage0Baseline, Stage1vLLM, Stage2FusedvLLM, InferenceResult
from src.evaluation import DatasetLoader, EvalSample
from src.metrics import MetricsCollector, QualityEvaluator, MetricsVisualizer


def print_system_info():
    """Print system and GPU information."""
    print("\n" + "="*80)
    print("System Information")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    Total memory: {mem_total:.2f} GB")
        print("All stages will use GPU acceleration with vLLM.")
    else:
        print("No CUDA GPU detected.")
        print("  - Stage 0 (Baseline) can run on CPU")
        print("  - Stages 1-2 (vLLM) require GPU and will exit gracefully if enabled")
    print("="*80 + "\n")


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
        
        # Check for existing checkpoint
        checkpoint_file = self._get_checkpoint_path(stage_name)
        start_index, results, outputs = self._load_checkpoint(checkpoint_file)
        
        if start_index > 0:
            print(f"[CHECKPOINT] Resuming from sample {start_index}/{len(eval_samples)}")
        
        # Initialize stage
        stage = stage_class(
            model_name=self.config['model']['name'],
            precision=self.config['model']['precision'],
            device="auto",  # Auto-detect GPU, fallback to CPU
        )
        
        # Load model
        start_time = time.time()
        stage.load_model()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Run inference on all samples
        batch_size = stage_config.get('batch_size', 1)
        num_samples = len(eval_samples)
        
        print(f"Running inference on {num_samples} samples (batch_size={batch_size})...")
        # Use tqdm for progress bar
        with tqdm(total=num_samples, desc=f"  {stage_name}", unit="sample", initial=start_index) as pbar:
            for i in range(start_index, num_samples, batch_size):
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
                
                # Update progress bar
                pbar.update(len(batch))
                
                # Save checkpoint after each batch
                self._save_checkpoint(checkpoint_file, stage_name, i + len(batch), results, outputs)
                
                # Clear GPU cache every 10 batches to prevent memory accumulation
                if torch.cuda.is_available() and (i // batch_size + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        print(f"Completed {stage_name}")
        
        # Delete checkpoint file on successful completion
        self._delete_checkpoint(checkpoint_file)
        
        # Cleanup
        stage.cleanup()
        
        # Force garbage collection and GPU memory release
        del stage
        import gc
        gc.collect()
        gc.collect()  # Call twice for more thorough cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats to get clean slate
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        return results, outputs
    
    def _get_checkpoint_path(self, stage_name: str) -> Path:
        """Get checkpoint file path for a stage."""
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_stage_name = stage_name.replace(' ', '_').replace('/', '_')
        return output_dir / f"{safe_stage_name}_checkpoint.json"
    
    def _load_checkpoint(self, checkpoint_file: Path) -> tuple:
        """Load checkpoint if exists, return (start_index, results, outputs)."""
        if not checkpoint_file.exists():
            return 0, [], []
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Reconstruct InferenceResult objects
            results = []
            for r in checkpoint.get('results', []):
                result = InferenceResult(
                    prompt=r['prompt'],
                    generated_text=r['generated_text'],
                    tokens_generated=r['tokens_generated'],
                    time_to_first_token=r['time_to_first_token'],
                    total_time=r['total_time'],
                    per_token_latency=r['per_token_latency'],
                    memory_used_mb=r.get('memory_used_mb', 0),
                    gpu_memory_mb=r.get('gpu_memory_mb', 0)
                )
                results.append(result)
            
            outputs = checkpoint.get('outputs', [])
            start_index = checkpoint.get('next_index', 0)
            
            return start_index, results, outputs
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            return 0, [], []
    
    def _save_checkpoint(self, checkpoint_file: Path, stage_name: str, 
                        next_index: int, results: List[InferenceResult], outputs: List[str]):
        """Save checkpoint after processing samples."""
        # Convert InferenceResult to dict
        results_dict = []
        for r in results:
            results_dict.append({
                'prompt': r.prompt,
                'generated_text': r.generated_text,
                'tokens_generated': r.tokens_generated,
                'time_to_first_token': r.time_to_first_token,
                'total_time': r.total_time,
                'per_token_latency': r.per_token_latency,
                'memory_used_mb': r.memory_used_mb,
                'gpu_memory_mb': r.gpu_memory_mb
            })
        
        checkpoint = {
            'stage_name': stage_name,
            'next_index': next_index,
            'results': results_dict,
            'outputs': outputs,
            'timestamp': time.time()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _delete_checkpoint(self, checkpoint_file: Path):
        """Delete checkpoint file after successful completion."""
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"[CHECKPOINT] Deleted checkpoint: {checkpoint_file.name}")
    
    def _save_stage_outputs(self, stage_name: str, eval_samples: List[EvalSample], outputs: List[str]) -> None:
        """Save stage outputs to JSON file for quality analysis."""
        output_dir = Path(self.config['output']['results_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create structured output data
        output_data = {
            'stage_name': stage_name,
            'num_samples': len(outputs),
            'samples': []
        }
        
        for i, (sample, output) in enumerate(zip(eval_samples, outputs)):
            output_data['samples'].append({
                'sample_id': i,
                'dataset': sample.dataset,  # Fixed: use 'dataset' not 'dataset_name'
                'prompt': sample.prompt,
                'reference': sample.expected_output if hasattr(sample, 'expected_output') else None,
                'generated_output': output,
                'output_length': len(output)
            })
        
        # Save to JSON file
        safe_stage_name = stage_name.replace(' ', '_').replace('/', '_')
        output_file = output_dir / f"{safe_stage_name}_outputs.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved outputs: {output_file}")
    
    def run_all_stages(self) -> None:
        """Run all enabled stages in the experiment."""
        
        # Print system information
        print_system_info()
        
        # Load evaluation datasets
        print("\n" + "="*80)
        print("Loading Evaluation Datasets")
        print("="*80)
        eval_samples = self.dataset_loader.load_all_datasets(self.config)
        
        # Use random subset for testing (10 samples)
        random.seed(42)  # For reproducibility
        num_test_samples = 10
        if len(eval_samples) > num_test_samples:
            eval_samples = random.sample(eval_samples, num_test_samples)
            print(f"\n*** Using random subset of {num_test_samples} samples for testing ***\n")
        
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
                
                # Save outputs to file for quality analysis
                self._save_stage_outputs(stage_config['name'], eval_samples, outputs)
                
                # Force cleanup between stages to free GPU memory
                import gc
                gc.collect()
                gc.collect()  # Double collection for thorough cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Additional cleanup: reset memory stats
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                    # Print GPU memory status
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
                    print(f"\nGPU Memory after {stage_config['name']}:")
                    print(f"  Allocated: {mem_allocated:.2f} GB")
                    print(f"  Reserved: {mem_reserved:.2f} GB")
                    print(f"  Free: {mem_free:.2f} GB")
                    print()
                
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
