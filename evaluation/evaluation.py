from vllm import LLM, SamplingParams
import os
from fine_tuning_new import filter_text
from fine_tuning_new import get_train_validation
import pandas as pd
from pathlib import Path
from vllm.lora.request import LoRARequest
import re
import torch
import gc
from math_equivalence import is_equiv
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

def extract_boxed(s: str) -> str | None:
    if not s: return None
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    return m.group(1).strip() if m else None


def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    exp_val = extract_boxed(expected_answer)
    gen_val = extract_boxed(generated_answer)
    if exp_val is None or gen_val is None:
        return False
    return is_equiv(gen_val, exp_val)

def create_prompts(df, tokenizer, is_math_500):
    think_pattern = re.compile(r"Think for\s+\d+\s+tokens\.\s*", flags=re.IGNORECASE)
    prompts = []
    for _, row in df.iterrows():
        # --- Clean prompt ---

        if is_math_500:
            clean_prompt = row["problem"] + "Let’s think step by step inside and output the final answer within boxed{{}}"
        else:
            raw_prompt = row["shortest_prompt"]
            clean_prompt = think_pattern.sub("", raw_prompt).strip()

        messages = [
            {"role": "user", "content": clean_prompt}
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        print(prompt_text)
        prompts.append(prompt_text)
    
    return prompts

def save_results(df, out, tokenizer, model_name):
    texts = []
    is_correct = []
    token_length = []
    question_ids = []
    for i, result in enumerate(out):
        question_ids.append(df["unique_id"].iloc[i])
        text = result.outputs[0].text
        is_correct.append(evaluate_answer(df["solution"].iloc[i], text))
        texts.append(text)
        token_length.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    results = pd.DataFrame({"id": question_ids, "text": texts, "length": token_length, "is_correct": is_correct})
    print(results)
    filename = "MATH-8B.parquet"
    path = Path(__file__).parent / filename
    results.to_parquet(path)
                            
def evaluate_fine_tuned_model(val_df, is_math_500):
    base_model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    lora_dir        = "models/full_epoch_4" 
    sampling = SamplingParams(
        max_tokens=5000,
        temperature=0,
    )

    llm= LLM(
        model=base_model_name,
        dtype="bfloat16",
        max_model_len=5000,
        enable_lora = True,
        max_lora_rank = 8
    )

    tokenizer = llm.get_tokenizer()
    prompts =create_prompts(val_df, tokenizer, is_math_500)

    out = llm.generate(prompts, sampling, lora_request=LoRARequest("lora_adapter", 1, lora_dir))
    save_results(val_df, out, tokenizer, base_model_name)

def evaluate_base_model(val_df, is_math_500):
    base_model_name = "qwen/Qwen3-8B"
    local_path = "/scratch/s3799042/Qwen3-8B"
    #base_model_name = "hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K"
    #local_path = "/scratch/s3799042/TokenSkip-7B"
    if local_path == "":
        local_path = base_model_name
    snapshot_download(
        repo_id=base_model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False
    )

    sampling = SamplingParams(
        max_tokens=8000,
        temperature=0,
    )




    llm = LLM(
        model=local_path,
        dtype="bfloat16",
        max_model_len=8000,
    )
    tokenizer = llm.get_tokenizer()
    prompts =create_prompts(val_df, tokenizer, is_math_500)

    out = llm.generate(prompts, sampling)
    save_results(val_df, out, tokenizer, base_model_name)

def read_aime():
    df = pd.read_parquet(Path(__file__).parent / "test_aime.parquet")
    df = df.rename(columns={
    "prompt": "problem",
    "solution_col": "solution",
    "question_id": "unique_id"
    })
    df = df.drop_duplicates(subset="question_id", keep="first")
    return df


if __name__ == "__main__":
    val = pd.read_json(
        "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl",
        lines=True
    )
    evaluate_base_model(val, True)
    #evalaute_fine_tuned_model(val, True)
