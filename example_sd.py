import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/data/wxtang/model/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path, 
        enforce_eager=True, 
        tensor_parallel_size=1,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 2,
            "prompt_lookup_max": 2,
        }
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
