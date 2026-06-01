#!/usr/bin/env python
# coding: utf-8

import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"                    # CUDA_VISIBLE_DEVICES
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/final_results.csv"
MAX_TOKENS  = 32768
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import re
import sys
import csv
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician."
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def run_inference():
    data = [json.loads(line) for line in open(DATA_PATH)]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["VLLM_FLASHINFER_FORCE_TENSOR_CORES"] = "0"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        attention_backend="TRITON_ATTN",
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.5,
        top_p=0.96,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("Model loaded.")

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_text += "</think>\n"
        prompts.append(prompt_text)

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    responses = [out.outputs[0].text.strip() for out in outputs]

    results = []
    for item, response in tqdm(zip(data, responses), total=len(data), desc="Collecting"):
        results.append({
            "id":       item.get("id"),
            "is_mcq":   bool(item.get("options")),
            "response": response,
        })

    print(f"Done. {len(results)} responses collected.")

    SAVE_EVAL = False
    
    with open(OUTPUT_PATH, "w", newline="") as f:
        if SAVE_EVAL:
            fieldnames = ["id", "is_mcq", "gold", "response", "correct"]
        else:
            fieldnames = ["id", "is_mcq", "response"]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                          "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            writer.writerow(record)
    
    print(f"Saved {len(results)} records to {out_path}")


if __name__ == "__main__":
    run_inference()