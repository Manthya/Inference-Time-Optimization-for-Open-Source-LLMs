"""
Quick test script to verify the setup.
Tests basic imports and configuration loading.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from stages import (
            BaseInferenceStage, 
            Stage0Baseline, 
            Stage1vLLM, 
            Stage2FusedvLLM
        )
        print("✓ Stages module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import stages: {e}")
        return False
    
    try:
        from evaluation import DatasetLoader, EvalSample
        print("✓ Evaluation module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import evaluation: {e}")
        return False
    
    try:
        from metrics import (
            MetricsCollector, 
            QualityEvaluator, 
            MetricsVisualizer
        )
        print("✓ Metrics module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import metrics: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        config_path = Path(__file__).parent / 'configs' / 'experiment_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Experiment: {config['experiment_name']}")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Temperature: {config['decoding']['temperature']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠ No GPU detected (experiment requires CUDA)")
            return False
    except Exception as e:
        print(f"✗ Failed to check GPU: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    
    try:
        from evaluation import DatasetLoader
        
        loader = DatasetLoader()
        
        # Test loading a small sample
        gsm8k_samples = loader.load_gsm8k(num_samples=2)
        print(f"✓ Loaded {len(gsm8k_samples)} GSM8K samples")
        
        humaneval_samples = loader.load_humaneval(num_samples=2)
        print(f"✓ Loaded {len(humaneval_samples)} HumanEval samples")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Running Setup Tests")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("GPU", test_gpu()))
    results.append(("Dataset Loading", test_dataset_loading()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou can now run the experiment:")
        print("  python run_experiment.py")
    else:
        print("Some tests failed. ✗")
        print("\nPlease fix the issues before running the experiment.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
