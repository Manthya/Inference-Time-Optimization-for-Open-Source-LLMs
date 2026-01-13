"""
Dataset loader and prompt generator for evaluation.
Handles GSM8K, HumanEval, MMLU, and synthetic long-context prompts.
Supports both pre-downloaded JSON files and dynamic generation.
"""
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalSample:
    """Single evaluation sample."""
    dataset: str
    prompt: str
    expected_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetLoader:
    """Load and prepare evaluation datasets."""
    
    def __init__(self, data_dir: str = "/app/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    def load_gsm8k(self, num_samples: int = 100, path: Optional[str] = None) -> List[EvalSample]:
        """
        Load GSM8K math problems.
        
        Args:
            num_samples: Number of samples to load
            path: Path to pre-downloaded JSON file (optional)
        """
        print(f"Loading GSM8K dataset ({num_samples} samples)...")
        
        # Try loading from pre-downloaded file
        json_path = Path(path) if path else self.data_dir / "gsm8k_test.json"
        
        if json_path.exists():
            print(f"  Loading from: {json_path}")
            with open(json_path, 'r') as f:
                problems = json.load(f)
            
            samples = []
            for i, prob in enumerate(problems[:num_samples]):
                prompt = f"Question: {prob['question']}\nAnswer: Let's solve this step by step.\n"
                # Extract numeric answer from full answer string
                answer = prob.get('answer', '')
                if '####' in answer:
                    answer = answer.split('####')[-1].strip()
                
                samples.append(EvalSample(
                    dataset="gsm8k",
                    prompt=prompt,
                    expected_output=answer,
                    metadata={"problem_id": i}
                ))
            
            return samples[:num_samples]
        
        # Fallback to built-in samples
        print("  Using built-in sample problems")
        problems = [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much money does she make every day at the farmers' market?",
                "answer": "18"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "3"
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "answer": "70000"
            },
            {
                "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
                "answer": "540"
            },
            {
                "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
                "answer": "20"
            },
        ]
        
        samples = []
        for i, prob in enumerate(problems):
            prompt = f"Question: {prob['question']}\nAnswer: Let's solve this step by step.\n"
            samples.append(EvalSample(
                dataset="gsm8k",
                prompt=prompt,
                expected_output=prob['answer'],
                metadata={"problem_id": i}
            ))
        
        # Repeat if needed
        while len(samples) < num_samples:
            samples.extend(samples[:num_samples - len(samples)])
        
        return samples[:num_samples]
    
    def load_humaneval(self, num_samples: int = 50, path: Optional[str] = None) -> List[EvalSample]:
        """
        Load HumanEval code generation problems.
        
        Args:
            num_samples: Number of samples to load
            path: Path to pre-downloaded JSON file (optional)
        """
        print(f"Loading HumanEval dataset ({num_samples} samples)...")
        
        json_path = Path(path) if path else self.data_dir / "humaneval_test.json"
        
        if json_path.exists():
            print(f"  Loading from: {json_path}")
            with open(json_path, 'r') as f:
                problems = json.load(f)
            
            samples = []
            for i, prob in enumerate(problems[:num_samples]):
                samples.append(EvalSample(
                    dataset="humaneval",
                    prompt=prob['prompt'],
                    expected_output=prob.get('canonical_solution', ''),
                    metadata={
                        "task_id": prob.get('task_id', f'HumanEval/{i}'),
                        "entry_point": prob.get('entry_point', '')
                    }
                ))
            
            return samples[:num_samples]
        
        # Fallback to built-in samples
        print("  Using built-in sample problems")
        problems = [
            {
                "prompt": '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
                "canonical_solution": '''    sorted_numbers = sorted(numbers)
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
            return True
    return False
'''
            },
            {
                "prompt": '''def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
''',
                "canonical_solution": '''    sum_value = 0
    prod_value = 1
    for n in numbers:
        sum_value += n
        prod_value *= n
    return (sum_value, prod_value)
'''
            },
        ]
        
        samples = []
        for i, prob in enumerate(problems):
            samples.append(EvalSample(
                dataset="humaneval",
                prompt=prob['prompt'],
                expected_output=prob['canonical_solution'],
                metadata={"problem_id": i}
            ))
        
        while len(samples) < num_samples:
            samples.extend(samples[:num_samples - len(samples)])
        
        return samples[:num_samples]
    
    def load_mmlu(self, num_samples: int = 50, path: Optional[str] = None) -> List[EvalSample]:
        """
        Load MMLU multiple-choice questions.
        
        Args:
            num_samples: Number of samples to load
            path: Path to pre-downloaded JSON file (optional)
        """
        print(f"Loading MMLU dataset ({num_samples} samples)...")
        
        json_path = Path(path) if path else self.data_dir / "mmlu_test.json"
        
        if json_path.exists():
            print(f"  Loading from: {json_path}")
            with open(json_path, 'r') as f:
                questions = json.load(f)
            
            samples = []
            for i, q in enumerate(questions[:num_samples]):
                choices = q.get('choices', [])
                choice_labels = ['A', 'B', 'C', 'D']
                
                prompt = f"Question: {q['question']}\n"
                for j, choice in enumerate(choices[:4]):
                    prompt += f"{choice_labels[j]}. {choice}\n"
                prompt += "\nAnswer:"
                
                # Convert numeric answer to letter
                answer = q.get('answer', 0)
                if isinstance(answer, int) and 0 <= answer < 4:
                    answer = choice_labels[answer]
                
                samples.append(EvalSample(
                    dataset="mmlu",
                    prompt=prompt,
                    expected_output=answer,
                    metadata={
                        "subject": q.get('subject', 'unknown'),
                        "problem_id": i
                    }
                ))
            
            return samples[:num_samples]
        
        # Fallback to built-in samples
        print("  Using built-in sample problems")
        questions = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "C"
            },
            {
                "question": "Which of the following is a prime number?",
                "choices": ["12", "15", "17", "21"],
                "answer": "C"
            },
            {
                "question": "What is the derivative of x^2?",
                "choices": ["x", "2x", "x^2", "2"],
                "answer": "B"
            },
        ]
        
        samples = []
        for i, q in enumerate(questions):
            prompt = f"Question: {q['question']}\n"
            for j, choice in enumerate(q['choices']):
                prompt += f"{['A', 'B', 'C', 'D'][j]}. {choice}\n"
            prompt += "\nAnswer:"
            
            samples.append(EvalSample(
                dataset="mmlu",
                prompt=prompt,
                expected_output=q['answer'],
                metadata={"problem_id": i}
            ))
        
        while len(samples) < num_samples:
            samples.extend(samples[:num_samples - len(samples)])
        
        return samples[:num_samples]
    
    def load_synthetic_long(self, num_samples: int = 20, path: Optional[str] = None) -> List[EvalSample]:
        """
        Load or generate synthetic long-context prompts.
        
        Args:
            num_samples: Number of samples to generate
            path: Path to pre-generated JSON file (optional)
        """
        print(f"Loading synthetic long-context prompts ({num_samples} samples)...")
        
        json_path = Path(path) if path else self.data_dir / "synthetic_long.json"
        
        if json_path.exists():
            print(f"  Loading from: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            samples = []
            for item in data[:num_samples]:
                prompt = item['context'] + f"\n\nQuestion: {item['question']}"
                samples.append(EvalSample(
                    dataset="synthetic_long",
                    prompt=prompt,
                    expected_output=item['expected_answer'],
                    metadata={
                        "problem_id": item['id'],
                        "context_length": item.get('context_length', len(prompt.split()))
                    }
                ))
            
            return samples[:num_samples]
        
        # Generate dynamically
        print("  Generating synthetic prompts...")
        samples = []
        context_lengths = [4096, 8192]
        
        for i in range(num_samples):
            context_len = random.choice(context_lengths)
            num_facts = context_len // 100
            
            facts = []
            for j in range(num_facts):
                facts.append(f"Fact {j}: The value of item {j} is {j * 10}.")
            
            context = " ".join(facts)
            target_idx = num_facts // 2
            question = f"What is the value of item {target_idx}?"
            
            prompt = context + f"\n\nQuestion: {question}"
            expected_answer = str(target_idx * 10)
            
            samples.append(EvalSample(
                dataset="synthetic_long",
                prompt=prompt,
                expected_output=expected_answer,
                metadata={
                    "problem_id": i,
                    "context_length": len(prompt.split()),
                    "target_fact": target_idx
                }
            ))
        
        return samples
    
    def load_all_datasets(self, config: Dict[str, Any]) -> List[EvalSample]:
        """
        Load all enabled datasets according to configuration.
        """
        all_samples = []
        
        datasets_config = config.get('datasets', {})
        
        if datasets_config.get('gsm8k', {}).get('enabled', True):
            cfg = datasets_config['gsm8k']
            all_samples.extend(self.load_gsm8k(
                num_samples=cfg.get('num_samples', 100),
                path=cfg.get('path')
            ))
        
        if datasets_config.get('humaneval', {}).get('enabled', True):
            cfg = datasets_config['humaneval']
            all_samples.extend(self.load_humaneval(
                num_samples=cfg.get('num_samples', 50),
                path=cfg.get('path')
            ))
        
        if datasets_config.get('mmlu', {}).get('enabled', True):
            cfg = datasets_config['mmlu']
            all_samples.extend(self.load_mmlu(
                num_samples=cfg.get('num_samples', 50),
                path=cfg.get('path')
            ))
        
        if datasets_config.get('synthetic_long', {}).get('enabled', True):
            cfg = datasets_config['synthetic_long']
            all_samples.extend(self.load_synthetic_long(
                num_samples=cfg.get('num_samples', 20),
                path=cfg.get('path')
            ))
        
        print(f"\nTotal evaluation samples: {len(all_samples)}")
        return all_samples
