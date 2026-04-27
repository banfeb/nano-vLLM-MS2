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
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
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

下面给出两组基准测试结果：

1. 用于验证 Qwen3-MoE 执行路径的 MoE 推理基准测试。
2. 基于 Qwen3-0.6B 的 Speculative Decoding 基准测试，用于衡量 N-gram 草稿生成流水线带来的吞吐提升。

### MoE 推理基准测试

这组测试数据来自 MoE 模型路径，**不包含** Speculative Decoding。所使用的 Qwen3-MoE 模型是一个随机参数 checkpoint，因此这里的测试主要用于功能验证与吞吐评估，而不是生成质量评估。

**测试配置：**
- 硬件：RTX 3090（24GB）
- 模型：`yujiepan/qwen3-moe-tiny-random`
- 总请求数：256 条序列
- 输入长度：在 100 到 1024 token 范围内随机采样
- 输出长度：在 100 到 1024 token 范围内随机采样

**性能结果：**
| 推理引擎 | 输出 Token 数 | 耗时（s） | 吞吐（tokens/s） |
|----------|---------------|-----------|------------------|
| vLLM | 133,966 | 5.53 | 24,242.19 |
| Nano-vLLM-MS | 133,966 | 6.37 | 23,214.07 |

### Speculative Decoding 基准测试

为了评估 Speculative Decoding 的效果，我们另外在 `Qwen3-0.6B` 上测试了 Nano-vLLM-MS。该模型也是当前 speculative decoding 示例所使用的基础稠密模型。这里对比的是普通解码与开启 N-gram speculative path 后的吞吐表现。

**测试配置：**
- 模型：`Qwen3-0.6B`
- 推理引擎：`Nano-vLLM-MS`
- 对比方式：普通解码 vs. Speculative Decoding

**性能结果：**
| 模式 | 吞吐（tokens/s） |
|------|------------------|
| 普通解码 | 1,418.09 |
| Speculative Decoding | 1,996.81 |

在该配置下，Speculative Decoding 将吞吐从 `1418.09 tok/s` 提升到 `1996.81 tok/s`，约提升 `40.8%`。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=banfeb/nano-vLLM-MS2&type=Date)](https://www.star-history.com/#banfeb/nano-vLLM-MS2&Date)
