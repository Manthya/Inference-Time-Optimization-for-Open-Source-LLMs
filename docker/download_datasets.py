#!/usr/bin/env python3
"""
Download evaluation datasets during Docker build.
Pre-downloads GSM8K, HumanEval, and MMLU for offline use.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def download_gsm8k(output_dir: Path):
    """Download GSM8K dataset."""
    print("\n[GSM8K] Downloading...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("gsm8k", "main", split="test")
        
        # Save to JSON
        output_file = output_dir / "gsm8k_test.json"
        samples = []
        for i, item in enumerate(dataset):
            samples.append({
                "id": i,
                "question": item["question"],
                "answer": item["answer"],
            })
        
        with open(output_file, "w") as f:
            json.dump(samples, f, indent=2)
        
        print(f"[GSM8K] Downloaded {len(samples)} samples -> {output_file}")
        return True
        
    except Exception as e:
        print(f"[GSM8K] Failed: {e}")
        return False


def download_humaneval(output_dir: Path):
    """Download HumanEval dataset."""
    print("\n[HumanEval] Downloading...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("openai_humaneval", split="test")
        
        # Save to JSON
        output_file = output_dir / "humaneval_test.json"
        samples = []
        for item in dataset:
            samples.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            })
        
        with open(output_file, "w") as f:
            json.dump(samples, f, indent=2)
        
        print(f"[HumanEval] Downloaded {len(samples)} samples -> {output_file}")
        return True
        
    except Exception as e:
        print(f"[HumanEval] Failed: {e}")
        return False


def download_mmlu(output_dir: Path):
    """Download MMLU dataset (selected subjects)."""
    print("\n[MMLU] Downloading...")
    
    subjects = [
        "abstract_algebra",
        "anatomy", 
        "astronomy",
        "college_mathematics",
        "computer_security",
    ]
    
    try:
        from datasets import load_dataset
        
        all_samples = []
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                for item in dataset:
                    all_samples.append({
                        "subject": subject,
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"],
                    })
            except Exception as e:
                print(f"  [MMLU/{subject}] Skipped: {e}")
        
        # Save to JSON
        output_file = output_dir / "mmlu_test.json"
        with open(output_file, "w") as f:
            json.dump(all_samples, f, indent=2)
        
        print(f"[MMLU] Downloaded {len(all_samples)} samples -> {output_file}")
        return True
        
    except Exception as e:
        print(f"[MMLU] Failed: {e}")
        return False


def create_synthetic_long(output_dir: Path):
    """Create synthetic long-context prompts."""
    print("\n[Synthetic] Creating long-context prompts...")
    
    samples = []
    context_lengths = [4096, 8192]
    
    for i in range(20):
        # Generate a long context
        context_len = context_lengths[i % 2]
        num_facts = context_len // 100
        
        facts = []
        for j in range(num_facts):
            facts.append(f"Fact {j}: The value of item {j} is {j * 10}.")
        
        context = " ".join(facts)
        target_idx = num_facts // 2
        
        samples.append({
            "id": i,
            "context": context,
            "question": f"What is the value of item {target_idx}?",
            "expected_answer": str(target_idx * 10),
            "context_length": context_len,
        })
    
    # Save to JSON
    output_file = output_dir / "synthetic_long.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"[Synthetic] Created {len(samples)} samples -> {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/data",
        help="Directory to save datasets"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Evaluation Datasets")
    print("=" * 60)
    
    results = []
    results.append(("GSM8K", download_gsm8k(output_dir)))
    results.append(("HumanEval", download_humaneval(output_dir)))
    results.append(("MMLU", download_mmlu(output_dir)))
    results.append(("Synthetic", create_synthetic_long(output_dir)))
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    all_success = all(r[1] for r in results)
    print("\n" + "=" * 60)
    
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
