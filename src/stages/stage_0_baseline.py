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
from typing import Optional

from .base_stage import BaseInferenceStage, InferenceResult


class Stage0Baseline(BaseInferenceStage):
    """
    Stage 0: Vanilla HuggingFace Transformers (Eager Mode)

    - Pure PyTorch eager execution
    - No FlashAttention
    - No kernel fusion
    - No CUDA graphs
    - Greedy or sampling decoding (vLLM comparable)
    """

    def __init__(
        self,
        model_name: str,
        precision: str = "bfloat16",
        device: str = "auto",
    ):
        super().__init__(model_name, precision, device, "Stage0-Vanilla-HF")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        print(f"[{self.stage_name}] Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Use torch_dtype="auto" to match the validated test approach
        # This lets the model choose the best dtype automatically
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

        # Actual device after device_map
        self.actual_device = next(self.model.parameters()).device
        actual_dtype = next(self.model.parameters()).dtype

        print(f"[{self.stage_name}] Model loaded successfully")
        print(f"  - Device: {self.actual_device}")
        print(f"  - Precision: {actual_dtype}")
        print(f"  - Attention: Default (auto-selected)")
        print(f"  - CUDA graphs: Disabled")

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> InferenceResult:
        """
        Greedy decoding using HuggingFace .generate() with do_sample=False.
        This approach is validated to work correctly with Qwen models.
        """

        # Apply chat template if available
        use_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )

        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.actual_device)

        input_len = inputs.input_ids.shape[1]

        # Track memory and time
        mem_before = self.get_memory_usage()
        start_time = time.perf_counter()

        # GREEDY DECODING with do_sample=False (validated approach)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding
                num_beams=1,      # Ensure greedy (not beam search)
            )

        total_time = time.perf_counter() - start_time

        # Extract only the generated tokens (exclude input)
        output_ids = generated_ids[0][input_len:]
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        tokens_generated = len(output_ids)

        print(f"\n[Stage 0] Generated {tokens_generated} tokens in {total_time:.2f}s")
        print(f"[Stage 0] Text preview: {generated_text[:100]}...")

        avg_token_time = total_time / max(tokens_generated, 1)
        mem_after = self.get_memory_usage()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            time_to_first_token=avg_token_time,
            total_time=total_time,
            per_token_latency=[avg_token_time] * tokens_generated,
            memory_used_mb=mem_after["cpu_memory_mb"] - mem_before["cpu_memory_mb"],
            gpu_memory_mb=mem_after.get("gpu_memory_mb", 0),
        )
