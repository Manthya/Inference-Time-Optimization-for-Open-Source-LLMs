"""Stage initialization."""
from .base_stage import BaseInferenceStage, InferenceResult
from .stage_0_baseline import Stage0Baseline
from .stage_1_vllm import Stage1vLLM
from .stage_2_fused import Stage2FusedvLLM

__all__ = [
    'BaseInferenceStage',
    'InferenceResult',
    'Stage0Baseline',
    'Stage1vLLM',
    'Stage2FusedvLLM',
]
