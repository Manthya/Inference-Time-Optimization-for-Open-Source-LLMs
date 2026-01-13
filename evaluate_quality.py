#!/usr/bin/env python3
"""
LLM-as-a-Judge Quality Evaluation Script

Uses Qwen-3B via vLLM to evaluate the quality of outputs from all stages.
Provides dataset-wise and overall quality scores with comprehensive logging.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import torch
from datetime import datetime

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("WARNING: vLLM not available. Install with: pip install vllm")


# Setup logging
def setup_logging(log_file: str = None):
    """Setup logging configuration."""
    log_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("./results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default log file name with timestamp
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"evaluation_{timestamp}.log"
    else:
        log_file = log_dir / log_file
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__), log_file


logger, log_file = setup_logging()


class LLMJudge:
    """LLM-as-a-Judge evaluator using vLLM."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", gpu_memory_utilization: float = 0.5):
        """Initialize the judge model."""
        logger.info(f"Initializing LLM Judge: {model_name}")
        
        if not VLLM_AVAILABLE:
            logger.error("vLLM is not available")
            raise RuntimeError("vLLM is required for LLM-as-a-Judge evaluation")
        
        if not torch.cuda.is_available():
            logger.error("No CUDA GPU detected")
            raise RuntimeError("GPU is required for judge model evaluation")
        
        # Clear GPU memory before loading judge model
        logger.info("Clearing GPU memory before loading judge model...")
        import gc
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        # Print GPU status
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            mem_free = mem_total - mem_reserved
            logger.info(f"GPU Memory Status: Total={mem_total:.2f}GB, Allocated={mem_allocated:.2f}GB, Free={mem_free:.2f}GB")
        
        logger.info(f"Loading model: {model_name}")
        
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
            max_model_len=1376,  # Set for T4 GPU limitations
            enforce_eager=False,  # Use CUDA graphs for efficiency
        )
        
        # Sampling parameters for judge (use greedy for consistency)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,  # Judge's evaluation response length
        )
        
        logger.info(f"✓ Judge model loaded successfully")
    
    def create_judge_prompt(self, prompt: str, response: str, reference: str = None) -> str:
        """Create evaluation prompt for the judge."""
        
        if reference and reference.strip():
            # With reference answer
            judge_prompt = f"""You are an expert evaluator assessing the quality of AI-generated responses.

**Task/Question:**
{prompt}

**Reference Answer (if available):**
{reference}

**AI Response to Evaluate:**
{response}

**Instructions:**
Evaluate the AI response on the following criteria (0-10 scale each):
1. **Correctness**: Is the answer factually correct and accurate?
2. **Completeness**: Does it fully address the question?
3. **Clarity**: Is it well-written and easy to understand?
4. **Relevance**: Does it stay on topic without unnecessary information?

Provide your evaluation in this exact JSON format:
{{
  "correctness": <score 0-10>,
  "completeness": <score 0-10>,
  "clarity": <score 0-10>,
  "relevance": <score 0-10>,
  "overall": <average score>,
  "reasoning": "<brief explanation>"
}}"""
        else:
            # Without reference answer
            judge_prompt = f"""You are an expert evaluator assessing the quality of AI-generated responses.

**Task/Question:**
{prompt}

**AI Response to Evaluate:**
{response}

**Instructions:**
Evaluate the AI response on the following criteria (0-10 scale each):
1. **Correctness**: Is the answer logically sound and reasonable?
2. **Completeness**: Does it fully address the question?
3. **Clarity**: Is it well-written and easy to understand?
4. **Relevance**: Does it stay on topic without unnecessary information?

Provide your evaluation in this exact JSON format:
{{
  "correctness": <score 0-10>,
  "completeness": <score 0-10>,
  "clarity": <score 0-10>,
  "relevance": <score 0-10>,
  "overall": <average score>,
  "reasoning": "<brief explanation>"
}}"""
        
        return judge_prompt
    
    def evaluate_response(self, prompt: str, response: str, reference: str = None) -> Dict[str, Any]:
        """Evaluate a single response using the judge model."""
        
        judge_prompt = self.create_judge_prompt(prompt, response, reference)
        
        # Generate evaluation
        outputs = self.llm.generate([judge_prompt], self.sampling_params)
        evaluation_text = outputs[0].outputs[0].text.strip()
        
        # Try to parse JSON from the response
        try:
            # Find JSON in the response (might have extra text)
            start_idx = evaluation_text.find('{')
            end_idx = evaluation_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = evaluation_text[start_idx:end_idx]
                scores = json.loads(json_str)
                
                # Validate structure
                required_keys = ['correctness', 'completeness', 'clarity', 'relevance', 'overall']
                if all(key in scores for key in required_keys):
                    return scores
            
            # If parsing failed, return default scores
            print(f"Warning: Failed to parse judge response: {evaluation_text[:100]}...")
            return {
                "correctness": 5.0,
                "completeness": 5.0,
                "clarity": 5.0,
                "relevance": 5.0,
                "overall": 5.0,
                "reasoning": "Failed to parse evaluation",
                "raw_response": evaluation_text
            }
            
        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return {
                "correctness": 5.0,
                "completeness": 5.0,
                "clarity": 5.0,
                "relevance": 5.0,
                "overall": 5.0,
                "reasoning": f"Error: {str(e)}",
                "raw_response": evaluation_text
            }


class QualityEvaluator:
    """Main quality evaluation orchestrator."""
    
    def __init__(self, results_dir: str = "./results"):
        """Initialize evaluator."""
        self.results_dir = Path(results_dir)
        self.judge = None
    
    def load_stage_outputs(self, stage_name: str, max_samples: int = 40, max_length: int = 1376) -> Dict[str, Any]:
        """Load outputs from a stage and filter by length."""
        logger.info(f"Loading outputs for stage: {stage_name}")
        safe_name = stage_name.replace(' ', '_').replace('/', '_')
        output_file = self.results_dir / f"{safe_name}_outputs.json"
        
        if not output_file.exists():
            logger.error(f"Output file not found: {output_file}")
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter samples by output_length < max_length and limit to max_samples
        original_count = len(data['samples'])
        filtered_samples = [s for s in data['samples'] if s.get('output_length', 0) < max_length]
        filtered_samples = filtered_samples[:max_samples]
        data['samples'] = filtered_samples
        
        logger.info(f"✓ Loaded {len(data['samples'])} samples from {stage_name} (filtered from {original_count}, length < {max_length})")
        return data
    
    def _get_comparable_samples(self, stage_data: Dict[str, Dict], max_samples: int = 40, max_length: int = 1376) -> Dict[str, Dict]:
        """Get the same samples across all stages for fair comparison."""
        logger.info("Finding comparable samples across all stages...")
        
        # Find sample_ids that exist in all stages and meet length criteria
        common_samples = None
        
        for stage_name, data in stage_data.items():
            # Filter by length and get sample_ids
            valid_samples = [s for s in data['samples'] if s.get('output_length', 0) < max_length]
            valid_ids = set(s['sample_id'] for s in valid_samples)
            
            if common_samples is None:
                common_samples = valid_ids
            else:
                common_samples = common_samples.intersection(valid_ids)
            
            logger.info(f"  {stage_name}: {len(valid_ids)} samples under {max_length} tokens")
        
        # Get first max_samples common sample_ids (sorted for consistency)
        common_sample_ids = sorted(list(common_samples))[:max_samples]
        logger.info(f"✓ Found {len(common_sample_ids)} comparable samples across all stages")
        
        # Filter each stage to only include common sample_ids
        for stage_name in stage_data:
            data = stage_data[stage_name]
            filtered_samples = [s for s in data['samples'] if s['sample_id'] in common_sample_ids]
            # Sort by sample_id to ensure same order across stages
            filtered_samples = sorted(filtered_samples, key=lambda x: x['sample_id'])
            data['samples'] = filtered_samples
            logger.info(f"  {stage_name}: Using {len(filtered_samples)} comparable samples")
        
        return stage_data
    
    def evaluate_all_stages(self, judge_model: str = "Qwen/Qwen2.5-3B-Instruct") -> Dict[str, Any]:
        """Evaluate outputs from all stages."""
        
        logger.info("="*80)
        logger.info("Starting LLM-as-a-Judge Quality Evaluation")
        logger.info("="*80)
        logger.info(f"Judge Model: {judge_model}")
        logger.info(f"Results Directory: {self.results_dir}")
        
        # Load outputs from all stages
        stages = [
            "Vanilla_HF_Transformers",
            "vLLM_Production",
            "vLLM_+_Kernel_Fusion"
        ]
        
        stage_data = {}
        for stage in stages:
            try:
                data = self.load_stage_outputs(stage)
                stage_data[stage] = data
            except FileNotFoundError as e:
                logger.warning(str(e))
                continue
        
        if not stage_data:
            logger.error("No stage outputs found!")
            raise ValueError("No stage outputs found!")
        
        # Find common sample_ids across all stages and filter for comparable evaluation
        stage_data = self._get_comparable_samples(stage_data, max_samples=40, max_length=1376)
        
        # Initialize judge model
        logger.info("Initializing LLM Judge Model")
        self.judge = LLMJudge(model_name=judge_model)
        
        # Evaluate each stage
        all_evaluations = {}
        
        for stage_name, data in stage_data.items():
            logger.info(f"Evaluating: {stage_name}")
            stage_evals = self._evaluate_stage(stage_name, data)
            all_evaluations[stage_name] = stage_evals
            logger.info(f"✓ Completed evaluation for {stage_name} ({len(stage_evals)} samples)")
        
        # Calculate aggregated metrics
        logger.info("Calculating Aggregated Metrics")
        results = self._aggregate_results(all_evaluations, stage_data)
        logger.info("✓ Metrics aggregated successfully")
        
        return results
    
    def _evaluate_stage(self, stage_name: str, data: Dict[str, Any], batch_size: int = 4) -> List[Dict[str, Any]]:
        """Evaluate all samples for a single stage with batching."""
        
        samples = data['samples']
        evaluations = []
        
        # Process samples in batches
        for batch_start in tqdm(range(0, len(samples), batch_size), desc=f"  Evaluating {stage_name}", unit="batch"):
            batch_end = min(batch_start + batch_size, len(samples))
            batch = samples[batch_start:batch_end]
            
            # Prepare batch of judge prompts
            judge_prompts = []
            batch_metadata = []
            
            for sample in batch:
                prompt = sample['prompt']
                response = sample['generated_output']
                reference = sample.get('reference', None)
                dataset = sample.get('dataset', 'unknown')
                
                # Skip empty responses
                if not response or len(response.strip()) == 0:
                    eval_result = {
                        "sample_id": sample['sample_id'],
                        "dataset": dataset,
                        "correctness": 0.0,
                        "completeness": 0.0,
                        "clarity": 0.0,
                        "relevance": 0.0,
                        "overall": 0.0,
                        "reasoning": "Empty response"
                    }
                    evaluations.append(eval_result)
                else:
                    # Create judge prompt
                    judge_prompt = self.judge.create_judge_prompt(prompt, response, reference)
                    judge_prompts.append(judge_prompt)
                    batch_metadata.append({
                        "sample_id": sample['sample_id'],
                        "dataset": dataset,
                        "prompt": prompt,
                        "response": response
                    })
            
            # Evaluate batch of samples in parallel
            if judge_prompts:
                outputs = self.judge.llm.generate(judge_prompts, self.judge.sampling_params)
                
                for i, output in enumerate(outputs):
                    evaluation_text = output.outputs[0].text.strip()
                    metadata = batch_metadata[i]
                    
                    # Parse evaluation
                    try:
                        start_idx = evaluation_text.find('{')
                        end_idx = evaluation_text.rfind('}') + 1
                        
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = evaluation_text[start_idx:end_idx]
                            scores = json.loads(json_str)
                            
                            required_keys = ['correctness', 'completeness', 'clarity', 'relevance', 'overall']
                            if all(key in scores for key in required_keys):
                                eval_result = scores
                            else:
                                raise ValueError("Missing required keys")
                        else:
                            raise ValueError("No JSON found")
                        
                    except Exception as e:
                        eval_result = {
                            "correctness": 5.0,
                            "completeness": 5.0,
                            "clarity": 5.0,
                            "relevance": 5.0,
                            "overall": 5.0,
                            "reasoning": f"Parse error: {str(e)[:50]}"
                        }
                    
                    eval_result['sample_id'] = metadata['sample_id']
                    eval_result['dataset'] = metadata['dataset']
                    evaluations.append(eval_result)
            
            # Clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return evaluations
    
    def _aggregate_results(self, all_evaluations: Dict[str, List[Dict]], 
                          stage_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results across stages and datasets."""
        
        results = {
            "judge_model": self.judge.model_name if self.judge else "unknown",
            "stages": {},
            "dataset_comparison": {},
            "overall_comparison": {}
        }
        
        # Get list of datasets
        datasets = set()
        for stage_name, data in stage_data.items():
            for sample in data['samples']:
                datasets.add(sample.get('dataset', 'unknown'))
        datasets = sorted(list(datasets))
        
        # Calculate per-stage metrics
        for stage_name, evaluations in all_evaluations.items():
            # Overall metrics
            overall_scores = self._calculate_metrics(evaluations)
            
            # Per-dataset metrics
            dataset_scores = {}
            for dataset in datasets:
                dataset_evals = [e for e in evaluations if e['dataset'] == dataset]
                if dataset_evals:
                    dataset_scores[dataset] = self._calculate_metrics(dataset_evals)
            
            results['stages'][stage_name] = {
                "overall": overall_scores,
                "by_dataset": dataset_scores,
                "num_samples": len(evaluations),
                "detailed_evaluations": evaluations
            }
        
        # Cross-stage comparison by dataset
        for dataset in datasets:
            results['dataset_comparison'][dataset] = {}
            for stage_name in all_evaluations.keys():
                stage_evals = [e for e in all_evaluations[stage_name] if e['dataset'] == dataset]
                if stage_evals:
                    metrics = self._calculate_metrics(stage_evals)
                    results['dataset_comparison'][dataset][stage_name] = metrics['overall']
        
        # Overall comparison across stages
        for stage_name, evaluations in all_evaluations.items():
            metrics = self._calculate_metrics(evaluations)
            results['overall_comparison'][stage_name] = metrics
        
        return results
    
    def _calculate_metrics(self, evaluations: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics from evaluations."""
        
        if not evaluations:
            return {
                "correctness": 0.0,
                "completeness": 0.0,
                "clarity": 0.0,
                "relevance": 0.0,
                "overall": 0.0,
                "count": 0
            }
        
        metrics = {
            "correctness": sum(e['correctness'] for e in evaluations) / len(evaluations),
            "completeness": sum(e['completeness'] for e in evaluations) / len(evaluations),
            "clarity": sum(e['clarity'] for e in evaluations) / len(evaluations),
            "relevance": sum(e['relevance'] for e in evaluations) / len(evaluations),
            "overall": sum(e['overall'] for e in evaluations) / len(evaluations),
            "count": len(evaluations)
        }
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], output_file: str = "quality_evaluation.json"):
        """Save evaluation results to JSON file."""
        
        logger.info(f"Saving evaluation results...")
        output_path = self.results_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Evaluation results saved to: {output_path}")
        
        # Also save a summary report
        self._save_summary_report(results)
    
    def _save_summary_report(self, results: Dict[str, Any]):
        """Save a human-readable summary report."""
        
        logger.info("Creating summary report...")
        summary_path = self.results_dir / "quality_evaluation_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LLM-as-a-Judge Quality Evaluation Summary\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Judge Model: {results['judge_model']}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall comparison
            f.write("="*80 + "\n")
            f.write("Overall Scores (All Datasets)\n")
            f.write("="*80 + "\n\n")
            
            for stage_name, metrics in results['overall_comparison'].items():
                f.write(f"{stage_name}:\n")
                f.write(f"  Correctness:  {metrics['correctness']:.2f}/10\n")
                f.write(f"  Completeness: {metrics['completeness']:.2f}/10\n")
                f.write(f"  Clarity:      {metrics['clarity']:.2f}/10\n")
                f.write(f"  Relevance:    {metrics['relevance']:.2f}/10\n")
                f.write(f"  Overall:      {metrics['overall']:.2f}/10\n")
                f.write(f"  Samples:      {metrics['count']}\n\n")
            
            # Dataset-wise comparison
            f.write("="*80 + "\n")
            f.write("Dataset-wise Scores\n")
            f.write("="*80 + "\n\n")
            
            for dataset, stage_scores in results['dataset_comparison'].items():
                f.write(f"\n{dataset}:\n")
                f.write("-" * 60 + "\n")
                for stage_name, score in stage_scores.items():
                    f.write(f"  {stage_name:30s}: {score:.2f}/10\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"✓ Summary report saved to: {summary_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console."""
        
        print("\n" + "="*80)
        print("Overall Quality Scores (All Datasets)")
        print("="*80)
        
        for stage_name, metrics in results['overall_comparison'].items():
            print(f"\n{stage_name}:")
            print(f"  Correctness:  {metrics['correctness']:.2f}/10")
            print(f"  Completeness: {metrics['completeness']:.2f}/10")
            print(f"  Clarity:      {metrics['clarity']:.2f}/10")
            print(f"  Relevance:    {metrics['relevance']:.2f}/10")
            print(f"  Overall:      {metrics['overall']:.2f}/10")
            print(f"  Samples:      {metrics['count']}")
        
        print("\n" + "="*80)
        print("Dataset-wise Comparison")
        print("="*80)
        
        for dataset, stage_scores in results['dataset_comparison'].items():
            print(f"\n{dataset}:")
            print("-" * 60)
            for stage_name, score in stage_scores.items():
                print(f"  {stage_name:30s}: {score:.2f}/10")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate quality of LLM outputs using LLM-as-a-Judge"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory containing stage output JSON files'
    )
    parser.add_argument(
        '--judge-model',
        type=str,
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Judge model to use for evaluation (default: Qwen2.5-3B-Instruct, requires ~6GB VRAM)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='quality_evaluation.json',
        help='Output filename for evaluation results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation (default: 4)'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("="*80)
        logger.info("LLM-as-a-Judge Quality Evaluation Started")
        logger.info("="*80)
        logger.info(f"Results Directory: {args.results_dir}")
        logger.info(f"Judge Model: {args.judge_model}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Log File: {log_file}")
        
        # Create evaluator
        evaluator = QualityEvaluator(results_dir=args.results_dir)
        
        # Run evaluation
        results = evaluator.evaluate_all_stages(judge_model=args.judge_model)
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        evaluator.save_results(results, output_file=args.output)
        
        logger.info("="*80)
        logger.info("✓ Evaluation Complete!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
