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
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
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

We report two sets of benchmark results below:

1. MoE inference benchmark for validating the Qwen3-MoE execution path.
2. Speculative decoding benchmark on Qwen3-0.6B for measuring the throughput gain of the N-gram drafting pipeline.

### MoE Inference Benchmark

This benchmark is measured on the MoE model path and does **not** include speculative decoding. The tested Qwen3-MoE model is a random-parameter checkpoint used for functionality and throughput validation rather than generation-quality evaluation.

**Test Configuration:**
- Hardware: RTX 3090 (24GB)
- Model: `yujiepan/qwen3-moe-tiny-random`
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100 and 1024 tokens
- Output Length: Randomly sampled between 100 and 1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|------------------|---------------|----------|------------------------|
| vLLM             | 133,966       | 5.53     | 24,242.19              |
| Nano-vLLM-MS     | 133,966       | 6.37     | 23,214.07              |

### Speculative Decoding Benchmark

To evaluate speculative decoding, we additionally benchmarked Nano-vLLM-MS on `Qwen3-0.6B`, which is the base dense model used by the current speculative decoding example. This result compares standard decoding against decoding with the N-gram speculative path enabled.

**Test Configuration:**
- Model: `Qwen3-0.6B`
- Inference Engine: `Nano-vLLM-MS`
- Comparison: standard decoding vs. speculative decoding

**Performance Results:**
| Mode | Throughput (tokens/s) |
|------|------------------------|
| Standard decoding | 1,418.09 |
| Speculative decoding | 1,996.81 |

Speculative decoding improves throughput from `1418.09 tok/s` to `1996.81 tok/s`, which is about a `40.8%` increase in this setup.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=banfeb/nano-vLLM-MS2&type=Date)](https://www.star-history.com/#banfeb/nano-vLLM-MS2&Date)
