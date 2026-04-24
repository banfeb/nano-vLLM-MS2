<p align="center">
<img width="300" src="assets/nano-vllm-ms-logo.png">
</p>


# Nano-vLLM-MS

A lightweight vLLM implementation based on nano-vLLM, enhanced with MoE model support and Speculative Decoding.

## Key Features

* ✅ **MoE model support** - Supports Qwen3-MoE models in the nano-vLLM inference pipeline. Included router-based top-k expert selection and fused expert computation.
* ✅ **Speculative Decoding pipeline** - Implements an end-to-end speculative decoding path with N-gram draft token proposal.
* 🚀 **Efficient N-gram proposer** - Uses prompt lookup with Numba acceleration and adaptive threading control to reduce drafting overhead for larger batches.
* ⚡ **Performance-oriented runtime** - Retains nano-vLLM optimizations such as prefix caching, tensor parallelism, CUDA graph capture, etc.
* 📊 **Reproducible examples and benchmark** - Provides `example_moe.py`, `example_sd.py`, and `bench.py` to quickly validate functionality and compare throughput with vLLM.

## Installation

```bash
pip install git+https://github.com/banfeb/nano-vLLM-MS2.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download yujiepan/qwen3-moe-tiny-random \
  --local-dir ~/huggingface/qwen3-moe-tiny-random/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example_moe.py` or `example_sd.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1,speculative_config={})
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM-MS."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 3090  (24GB)
- Model: models--yujiepan--qwen3-moe-tiny-random
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     |  5.53    |  24242.19                |
| Nano-vLLM-MS      | 133,966     | 6.37    |  23214.07              |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=banfeb/nano-vLLM-MS2&type=Date)](https://www.star-history.com/#banfeb/nano-vLLM-MS2&Date)
