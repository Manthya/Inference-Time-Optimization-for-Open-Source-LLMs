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
                 device: str = "auto"):
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
        
        # Apply chat template if available (Qwen models need this)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Format as chat message
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"[DEBUG] Chat template applied, length: {len(formatted_prompt)} chars")
        else:
            formatted_prompt = prompt
            print(f"[DEBUG] No chat template, using raw prompt")
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        print(f"[DEBUG] Input tokens: {input_length}, EOS token ID: {self.tokenizer.eos_token_id}")
        
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
            
            # Qwen models have special stop tokens
            stop_token_ids = [self.tokenizer.eos_token_id]
            # Add Qwen-specific stop tokens if they exist
            if hasattr(self.tokenizer, 'im_end_id'):
                stop_token_ids.append(self.tokenizer.im_end_id)
            elif 151643 in self.tokenizer.get_vocab().values():  # <|im_end|>
                stop_token_ids.append(151643)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding
                temperature=None,  # Not used with greedy
                top_k=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,  # Multiple stop tokens
                use_cache=True,  # Enable KV cache (standard)
            )
            
            total_time = time.perf_counter() - generation_start
        
        # Decode output
        generated_ids = outputs[0][input_length:]
        generated_ids_list = generated_ids.tolist()
        
        # Remove padding tokens (0) from the end
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        non_pad_ids = []
        for token_id in generated_ids_list:
            if token_id == pad_token_id:
                break
            non_pad_ids.append(token_id)
        
        # If we filtered out padding, use the non-padded version
        if len(non_pad_ids) < len(generated_ids_list):
            print(f"[DEBUG] Filtered {len(generated_ids_list) - len(non_pad_ids)} padding tokens")
            generated_ids_list = non_pad_ids
            tokens_generated = len(non_pad_ids)
        else:
            tokens_generated = len(generated_ids)
        
        generated_text = self.tokenizer.decode(non_pad_ids, skip_special_tokens=True)
        
        # Debug: Check for any EOS-like tokens
        if self.tokenizer.eos_token_id in generated_ids_list:
            eos_position = generated_ids_list.index(self.tokenizer.eos_token_id)
            print(f"[DEBUG] Stage 0 hit EOS (151645) at token {eos_position}/{tokens_generated}")
        # Check for other potential stop tokens (Qwen specific)
        elif 151643 in generated_ids_list:  # Qwen's <|im_end|> token
            stop_pos = generated_ids_list.index(151643)
            print(f"[DEBUG] Stage 0 hit <|im_end|> (151643) at token {stop_pos}/{tokens_generated}")
        else:
            print(f"[DEBUG] Stage 0 generated all {tokens_generated} tokens (no EOS)")
            # Show last few token IDs to see what's actually generated
            last_n = min(10, len(generated_ids_list))
            print(f"[DEBUG] Last {last_n} token IDs: {generated_ids_list[-last_n:]}")
        
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
        
        # Clear GPU cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
