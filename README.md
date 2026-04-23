<p align="center">
<img width="300" src="assets/nano-vllm-ms-logo.png">
</p>


# Nano-vLLM-MS

nano-vllm基础上完成:
1. 对moe模型的支持
2. 对Speculative Decoding技术的支持

## Key Features

* ✅ **MoE model support** - End-to-end support for MoE models in the nano-vLLM pipeline.
* ✅ **Speculative Decoding (N-gram draft + rejection sampling)** - Decode path now supports N-gram drafting, verify pass, and scheduler post-processing for accepted tokens.
* ✅ **Optional speculative config** - `speculative_config` is optional; regular decode path still works when it is not provided.
* ✅ **N-gram proposer threading control** - 已加入“决定是否启用多线程”的门限参数（当前策略在 `tp=1` 场景下可用）。
* ⚠️ **Spec Decode TP TODO** - 如果有 TP 需要重新考虑线程参数；TODO: 之后完成 `spec_decode` 的 TP 并行后，重新调整线程参数策略。
* ⚡ **Optimization suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

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
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
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
