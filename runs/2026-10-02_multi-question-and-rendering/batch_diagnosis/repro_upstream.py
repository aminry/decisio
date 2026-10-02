"""Minimal reproduction on stock vLLM 0.30.0 (no decisio code): a prompt scored alone and the same prompt in a batch of
two give different next-token log-probabilities, and identical batches differ between runs.

    VLLM_USE_DEEP_GEMM=0 python repro_upstream.py [--model Qwen/Qwen3.6-35B-A3B-FP8] [--prompts 20]
"""

import argparse
import random

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=1400)
    a = ap.parse_args()

    llm = LLM(model=a.model, max_model_len=8192, limit_mm_per_prompt={"image": 0, "video": 0}, max_logprobs=20,
              logprobs_mode="processed_logprobs")
    labels = list(range(32, 52))  # 20 token ids; their log-softmax is compared
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(labels), allowed_token_ids=labels)
    rng = random.Random(0)


    def lp(prompts):
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], [sp] * len(prompts), use_tqdm=False)
        return [np.array([o.outputs[0].logprobs[0][t].logprob for t in labels]) for o in outs]


    worst = {"alone_repeat": 0.0, "alone_vs_batch2": 0.0, "batch2_repeat": 0.0}
    for _ in range(a.prompts):
        p = [rng.randrange(1000, 150000) for _ in range(a.tokens)]
        a1, a2 = lp([p])[0], lp([p])[0]
        b1, b2 = lp([p, p]), lp([p, p])
        worst["alone_repeat"] = max(worst["alone_repeat"], float(np.abs(a1 - a2).max()))
        worst["alone_vs_batch2"] = max(worst["alone_vs_batch2"], float(max(np.abs(a1 - x).max() for x in b1)))
        worst["batch2_repeat"] = max(worst["batch2_repeat"], float(max(np.abs(x - y).max() for x, y in zip(b1, b2))))
    print("max |d logprob| over", a.prompts, "prompts:", worst)


if __name__ == "__main__":
    main()
