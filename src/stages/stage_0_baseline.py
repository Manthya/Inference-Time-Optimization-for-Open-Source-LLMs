"""
Stage 0: Vanilla HuggingFace Transformers Baseline
- Pure PyTorch eager mode
- Standard attention (no FlashAttention)
- No batching optimizations
- No CUDA graphs
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from .base_stage import BaseInferenceStage, InferenceResult


class Stage0Baseline(BaseInferenceStage):
    """Vanilla HuggingFace Transformers baseline implementation."""
    
    def __init__(self, 
                 model_name: str,
                 precision: str = "bfloat16",
                 device: str = "cuda"):
        super().__init__(model_name, precision, device, "Stage0-Vanilla-HF")
        
    def load_model(self):
        """Load model using standard HuggingFace Transformers."""
        print(f"[{self.stage_name}] Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Determine dtype
        if self.precision == "bfloat16":
            dtype = torch.bfloat16
        elif self.precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load model in standard mode
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            # Explicitly disable optimizations
            attn_implementation="eager",  # Use standard attention
        )
        
        self.model.eval()
        print(f"[{self.stage_name}] Model loaded successfully")
        
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 512,
                 temperature: float = 0.0) -> InferenceResult:
        """Generate text using standard HF generation."""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Memory before generation
        mem_before = self.get_memory_usage()
        
        # Track timing
        start_time = time.perf_counter()
        per_token_latencies = []
        first_token_time = None
        
        # Generate with greedy decoding (temperature=0)
        with torch.no_grad():
            # Use generate with explicit greedy parameters
            generation_start = time.perf_counter()
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding
                temperature=None,  # Not used with greedy
                top_k=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache (standard)
            )
            
            total_time = time.perf_counter() - generation_start
        
        # Decode output
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)
        
        # Calculate metrics
        # Note: In vanilla HF, we can't easily get per-token timing
        # So we approximate TTFT and per-token latency
        avg_token_time = total_time / max(tokens_generated, 1)
        time_to_first_token = avg_token_time  # Approximation
        per_token_latencies = [avg_token_time] * tokens_generated
        
        # Memory after generation
        mem_after = self.get_memory_usage()
        memory_used = mem_after['cpu_memory_mb'] - mem_before['cpu_memory_mb']
        gpu_memory = mem_after.get('gpu_memory_mb', 0)
        
        return InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            time_to_first_token=time_to_first_token,
            total_time=total_time,
            per_token_latency=per_token_latencies,
            memory_used_mb=memory_used,
            gpu_memory_mb=gpu_memory
        )
