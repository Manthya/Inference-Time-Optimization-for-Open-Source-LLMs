"""
Base class for all inference stages.
Defines the common interface and measurement utilities.
"""
import time
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import psutil
import numpy as np


@dataclass
class InferenceResult:
    """Results from a single inference run."""
    prompt: str
    generated_text: str
    tokens_generated: int
    time_to_first_token: float
    total_time: float
    per_token_latency: List[float]
    memory_used_mb: float
    gpu_memory_mb: Optional[float] = None
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.total_time > 0:
            return self.tokens_generated / self.total_time
        return 0.0
    
    @property
    def avg_per_token_latency(self) -> float:
        """Average per-token latency in milliseconds."""
        if self.per_token_latency:
            return np.mean(self.per_token_latency) * 1000
        return 0.0


class BaseInferenceStage(ABC):
    """Base class for all inference optimization stages."""
    
    def __init__(self, 
                 model_name: str,
                 precision: str = "bfloat16",
                 device: str = "cuda",
                 stage_name: str = "base"):
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self.stage_name = stage_name
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 512,
                 temperature: float = 0.0) -> InferenceResult:
        """Generate text from a prompt."""
        pass
    
    def batch_generate(self,
                      prompts: List[str],
                      max_tokens: int = 512,
                      temperature: float = 0.0) -> List[InferenceResult]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature)
            results.append(result)
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        # CPU memory
        process = psutil.Process()
        memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info
    
    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
