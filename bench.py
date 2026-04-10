import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    qwen3_0_6B = "/data/wxtang/model/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
    qwen3_moe_tiny_random = "/data/wxtang/model/models--yujiepan--qwen3-moe-tiny-random/snapshots/fb6c5ee2a2c19bd9aced6d9afd8a858966a7bb7e"
    path = os.path.expanduser(qwen3_moe_tiny_random)
    # enforce_eager False代表开启图捕获优化，但moe暂时未能支持图捕获优化，
    # 因此这里设置为True以确保正确性。后续moe支持图捕获优化后，可以将enforce_eager设置为False以获得更好的性能。
    llm = LLM(path, enforce_eager=True, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
