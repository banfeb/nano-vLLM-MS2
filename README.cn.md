<p align="center">
<img width="300" src="assets/nano-vllm-ms-logo.png">
</p>


# Nano-vLLM-MS

这是一个基于 nano-vLLM 的轻量级 vLLM 实现，增强了对 MoE 模型和 Speculative Decoding（推测解码）的支持。

## 核心特性

* ✅ **MoE 模型支持** - 在 nano-vLLM 推理流水线中支持 Qwen3-MoE 模型，包含基于 router 的 top-k expert 选择与融合专家计算。
* ✅ **Speculative Decoding 流水线** - 实现了端到端的推测解码路径，包含 N-gram 草稿 token 提案。
* 🚀 **高效 N-gram proposer** - 使用基于 prompt 查找的方法，并结合 Numba 加速与自适应线程控制，降低大 batch 场景下的草稿生成开销。
* ⚡ **面向性能的运行时** - 保留了 nano-vLLM 的优化能力，例如 prefix cache、tensor parallel、CUDA graph capture 等。
* 📊 **可复现示例与基准测试** - 提供 `example_moe.py`、`example_sd.py` 和 `bench.py`，可快速验证功能并与 vLLM 对比吞吐。

## 安装

```bash
pip install git+https://github.com/banfeb/nano-vLLM-MS2.git
```

## 模型下载

如果你想手动下载模型权重，可使用以下命令：
```bash
huggingface-cli download --resume-download yujiepan/qwen3-moe-tiny-random \
  --local-dir ~/huggingface/qwen3-moe-tiny-random/ \
  --local-dir-use-symlinks False
```

## 快速开始

使用方式可参考 `example_moe.py` 或 `example_sd.py`。API 基本与 vLLM 接口一致，`LLM.generate` 方法有少量差异：
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1,speculative_config={})
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM-MS."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## 基准测试

基准测试请参考 `bench.py`。

**测试配置：**
- 硬件：RTX 3090（24GB）
- 模型：models--yujiepan--qwen3-moe-tiny-random
- 总请求数：256 条序列
- 输入长度：在 100–1024 token 范围内随机采样
- 输出长度：在 100–1024 token 范围内随机采样

**性能结果：**
| 推理引擎 | 输出 Token 数 | 耗时（s） | 吞吐（tokens/s） |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     |  5.53    |  24242.19                |
| Nano-vLLM-MS      | 133,966     | 6.37    |  23214.07              |

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=banfeb/nano-vLLM-MS2&type=Date)](https://www.star-history.com/#banfeb/nano-vLLM-MS2&Date)
