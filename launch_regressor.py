import argparse
import re
import time
import os
import numpy as np
import pandas as pd 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from data.processing.math_equivalence import is_equiv
from pathlib import Path
from evaluation.evaluation import evaluate_answer

# L1_p08_eval: L1 model on the eval dataset with the template
# sentences_template_p08_eval.parquet: is qwen3-8b running on the eval dataset with the chat template
# tale_template_p08: checking whether with the template tale actually adheres to the target tokens, math and aime
# sentences_template_math_aime.parquet: checking whether with adding the template the model performs and adheres better.
#L1_p08_new_evaluate.parquet: Using sebs evaluate_answer function

MODEL = "l3lab/L1-Qwen3-8B-Max"  #"/scratch/s3799042/Qwen3-8B/" #"l3lab/L1-Qwen3-8B-Max"  # "/scratch/s3799042/Qwen3-8B/" 
OUTPUT_PATH = Path(__file__).parent / "data" / "processed" / "results"/ "L1_p08_new_evaluate.parquet"
TEST_PROBLEMS = Path(__file__).parent / "data" / "processed" / "dataset_splitting" / "test.parquet" #"eval_L1_bert" / "gsm8k_olympiad_amc.parquet" #"dataset_splitting" / "test.parquet" #"eval_L1_bert" / "gsm8k_olympiad_amc.parquet" # "old" / "dataset_splitting" / "test.parquet"
TEST_TARGET_TOKENS = Path(__file__).parent / "data" / "processed" / "regressor_target_tokens" / "rgs_results_L1_b20_p08.npy"

IS_EVAL = False

def build_prompt(problem: str, target_think_tokens: int, tokenizer, use_chat_template = False) -> str:
    if IS_EVAL:
        problem = f"{problem} Let’s think step by step inside and output the final answer within boxed{{}}."

    prompt = f"{problem} Think for {target_think_tokens} tokens." # "Use less than {target_think_tokens} tokens." 
                                                                               # Think for {target_think_tokens} tokens. <think>" 
                                                                               # Use less than {target_think_tokens // 60} sentences. <think>"


    if use_chat_template == False:
        return prompt
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    prompt= tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return prompt

def extract_boxed(s: str) -> str | None:
    if not s: return None
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    return m.group(1).strip() if m else None

def extract_think_text(full_text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def main():
    df = pd.read_parquet(TEST_PROBLEMS)
    targets = np.load(TEST_TARGET_TOKENS, allow_pickle=True).item()
    print(targets)
    df = df.rename(columns={'id': 'question_id'})

    df =  df.groupby("question_id", as_index=False).first()
    mask = np.isin(df['question_id'], targets['ids'])
    df = df[mask]
    if len(df) != len(targets['ids']):
        raise ValueError(f"Data length mismatch: DataFrame has {len(df)} rows, targets has {len(targets['ids'])} elements.")
    
    targets = targets['target']
    #os.makedirs(args.generated_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    llm = LLM(
        model=MODEL,
        dtype="float16", 
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
    )

    sampling_params = SamplingParams(max_tokens=10_000, temperature=0, skip_special_tokens=False)

    prompts = []
    for i in range(len(df)):
        prompts.append(build_prompt(df.iloc[i]["prompt"], targets[i], tokenizer, use_chat_template=True))

    print(prompts[0])

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    
    full_texts = []
    generated_texts = []
    
    for i, output in enumerate(outputs):
        prompt_text = prompts[i]
        generated_suffix = output.outputs[0].text
        full_text = prompt_text + generated_suffix
        full_texts.append(full_text)
        generated_texts.append(generated_suffix)

    think_encodings = tokenizer(generated_texts, add_special_tokens=False)["input_ids"]
    generated_lengths = [len(ids) for ids in think_encodings]

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        full_text = full_texts[i]
        (is_ok, _, _) = evaluate_answer(row["solution_col"], full_text)

        records.append({
            "question_id": row['question_id'],
            "prompt": prompts[i],
            "solution": row["solution_col"],
            "generated_think_text": generated_texts[i],
            "generated_text": full_text,
            "target_think_tokens": int(targets[i]),
            "generated_think_tokens": generated_lengths[i],
            "latency_sec": (t1 - t0) / len(prompts),
            "is_correct": bool(is_ok),
        })

    final_df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_PATH)
    final_df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()