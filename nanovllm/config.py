import os
from dataclasses import dataclass
from transformers import AutoConfig
from typing import Any

@dataclass
class SpeculativeConfig:
    method: str = "ngram"
    num_speculative_tokens: int = 3
    prompt_lookup_max: int = 2
    prompt_lookup_min: int = 1
    
    def __post_init__(self):
        supported_methods: list[str] = ["ngram"]
        if self.method not in supported_methods:
            raise ValueError(
                f"Unsupported speculative decoding method: {self.method}. "
                f"Supported methods: {supported_methods}"
            )
        if self.prompt_lookup_min > self.prompt_lookup_max:
            raise ValueError(
                "prompt_lookup_min must be less than or equal to prompt_lookup_max, "
                f"got {self.prompt_lookup_min} > {self.prompt_lookup_max}"
            )

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    speculative_config: dict[str, Any] | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.rope_scaling = None          # disable auto rope scaling
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)