#!/usr/bin/env python3
"""
Download model weights during Docker build.
This script pre-downloads the model so it's cached in the Docker image.
"""
import argparse
import os
import sys
from pathlib import Path


def download_model(model_name: str, cache_dir: str):
    """Download model weights using HuggingFace Hub."""
    print(f"=" * 60)
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"=" * 60)
    
    # Set cache directory
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    # Import after setting env vars
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoConfig
    
    try:
        # Download model files
        print("\n[1/3] Downloading model weights...")
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=None,
            local_dir_use_symlinks=False,
        )
        print(f"Model downloaded to: {model_path}")
        
        # Download tokenizer
        print("\n[2/3] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print("Tokenizer downloaded successfully")
        
        # Download config
        print("\n[3/3] Downloading config...")
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(f"Model config: {config.model_type}")
        
        print("\n" + "=" * 60)
        print("Model download complete!")
        print("=" * 60)
        
        # Print cache size
        cache_path = Path(cache_dir)
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
            print(f"Total cache size: {total_size / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download model for experiment")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/app/models",
        help="Directory to cache model"
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model, args.cache_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
