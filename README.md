<p align="center">
<img width="300" src="assets/nano-vllm-ms-logo.png">
</p>


# Nano-vLLM-MS

nano-vllm基础上完成:
1. 对moe模型的支持
2. 对Speculative Decoding技术的支持

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/banfeb/nano-vLLM-MS2.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
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
| Nano-vLLM-MS      | 133,966     | 6.37    |  21020.11               |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=banfeb/nano-vLLM-MS2&type=Date)](https://www.star-history.com/#banfeb/nano-vLLM-MS2&Date)
