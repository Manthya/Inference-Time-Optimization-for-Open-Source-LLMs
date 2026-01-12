"""
Stage 1: vLLM Production Runtime
- FlashAttention v2
- Paged KV Cache (PagedAttention)
- Continuous batching
- Chunked prefill
- CUDA Graphs
"""
import time
from typing import List, Optional
from vllm import LLM, SamplingParams

from .base_stage import BaseInferenceStage, InferenceResult


class Stage1vLLM(BaseInferenceStage):
    """vLLM production runtime implementation."""
    
    def __init__(self, 
                 model_name: str,
                 precision: str = "bfloat16",
                 device: str = "cuda",
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9):
        super().__init__(model_name, precision, device, "Stage1-vLLM")
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
    def load_model(self):
        """Load model using vLLM."""
        print(f"[{self.stage_name}] Loading model with vLLM: {self.model_name}")
        
        # Determine dtype
        dtype_map = {
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32"
        }
        dtype = dtype_map.get(self.precision, "auto")
        
        # Initialize vLLM engine
        # This automatically enables:
        # - FlashAttention v2
        # - PagedAttention (Paged KV Cache)
        # - Continuous batching
        # - CUDA graphs (for decode)
        self.model = LLM(
            model=self.model_name,
            dtype=dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,  # Enable CUDA graphs
        )
        
        print(f"[{self.stage_name}] Model loaded successfully")
        print(f"  - FlashAttention: Enabled")
        print(f"  - PagedAttention: Enabled")
        print(f"  - CUDA Graphs: Enabled")
        
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 512,
                 temperature: float = 0.0) -> InferenceResult:
        """Generate text using vLLM."""
        
        # Configure sampling parameters for greedy decoding
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            # Greedy decoding when temperature is 0
        )
        
        # Memory before generation
        mem_before = self.get_memory_usage()
        
        # Track timing
        start_time = time.perf_counter()
        
        # Generate
        outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        
        total_time = time.perf_counter() - start_time
        
        # Extract results
        output = outputs[0]
        generated_text = output.outputs[0].text
        tokens_generated = len(output.outputs[0].token_ids)
        
        # vLLM provides detailed metrics
        metrics = output.metrics if hasattr(output, 'metrics') else None
        
        # Calculate timing metrics
        if metrics:
            time_to_first_token = getattr(metrics, 'first_token_time', 0)
            per_token_latencies = getattr(metrics, 'token_times', [])
        else:
            # Fallback approximation
            avg_token_time = total_time / max(tokens_generated, 1)
            time_to_first_token = avg_token_time
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
    
    def batch_generate(self,
                      prompts: List[str],
                      max_tokens: int = 512,
                      temperature: float = 0.0) -> List[InferenceResult]:
        """
        Generate text for multiple prompts using continuous batching.
        This is where vLLM really shines!
        """
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
        )
        
        # Memory before generation
        mem_before = self.get_memory_usage()
        
        # Track timing
        start_time = time.perf_counter()
        
        # Batch generate (vLLM handles continuous batching internally)
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)
        
        total_time = time.perf_counter() - start_time
        
        # Convert to InferenceResult objects
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            
            # Per-request metrics
            metrics = output.metrics if hasattr(output, 'metrics') else None
            
            if metrics:
                time_to_first_token = getattr(metrics, 'first_token_time', 0)
                per_token_latencies = getattr(metrics, 'token_times', [])
            else:
                avg_token_time = total_time / max(tokens_generated, 1)
                time_to_first_token = avg_token_time
                per_token_latencies = [avg_token_time] * tokens_generated
            
            # Memory after generation (shared across batch)
            mem_after = self.get_memory_usage()
            memory_used = (mem_after['cpu_memory_mb'] - mem_before['cpu_memory_mb']) / len(prompts)
            gpu_memory = mem_after.get('gpu_memory_mb', 0) / len(prompts)
            
            results.append(InferenceResult(
                prompt=prompts[i],
                generated_text=generated_text,
                tokens_generated=tokens_generated,
                time_to_first_token=time_to_first_token,
                total_time=total_time / len(prompts),  # Amortized
                per_token_latency=per_token_latencies,
                memory_used_mb=memory_used,
                gpu_memory_mb=gpu_memory
            ))
        
        return results
