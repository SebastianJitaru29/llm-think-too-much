import os 

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from collections import Counter
import math
import numpy as np
import torch
import re

def extract_generated_text(df):
    splitted = df["generated_text"].str.split("Let’s think step by step inside and output the final answer within boxed{}.", n=1, expand=True)
    out_text = splitted[1].str.replace(r"Think for \d+ tokens\. +", "", n=1, regex=True).str.strip()
    return out_text

def missing_think_end_mask(text):
    end_counts = text.str.count("</think>")
    missing_think_end_mask = end_counts != 0
    return missing_think_end_mask

def zero_think_mask(df):
    mask = df["generated_think_tokens"] != 0
    return mask

def math500_mask(df):
    df_500 = pd.read_json(
        "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl",
        lines=True
    )
    problems = df_500["problem"].tolist()

    mask = df["prompt"].apply(
        lambda p: not any(problem in p for problem in problems)
    )
    return mask

def repetition_score(text):
    words = text.lower().split()
    counts = Counter(words)
    total = len(words)
    k = len(counts)

    if k <= 1:   # fully degenerate case
        return 1.0

    # Shannon entropy
    H = -sum((c/total) * math.log(c/total) for c in counts.values())

    # Normalized entropy
    H_norm = H / math.log(k)

    return H_norm

def repetition_score(text):
    words = text.lower().split()
    counts = Counter(words)
    total = len(words)
    k = len(counts)

    if k <= 1:   # fully degenerate case
        return 1.0

    # Shannon entropy
    H = -sum((c/total) * math.log(c/total) for c in counts.values())

    # Normalized entropy
    H_norm = H / math.log(k)

    return H_norm

def low_entropy_mask(text):
    entropy = text.map(repetition_score)
    low_entropy_mask = entropy > entropy.quantile(0.004)
    return low_entropy_mask

def filter_text(df):
    text = extract_generated_text(df)
    mask0 = missing_think_end_mask(text)
    mask1 = low_entropy_mask(text)
    mask2 = zero_think_mask(df)
    mask3 = math500_mask(df)
    initial_count = len(df)
    df = df[mask0 & mask1 & mask2 & mask3]
    print(f"filtered out: {initial_count-len(df)} from {initial_count}")
    return df


def split_train_validation(df, val_question_ids):

    val_question_ids = set(val_question_ids)

    # Validation set: all rows whose question_id is in the supplied list
    df_val = df[df["question_id"].isin(val_question_ids)].copy()

    # Training set: all remaining rows
    df_train = df[~df["question_id"].isin(val_question_ids)].copy()

    return df_train, df_val

def get_train_validation(df):
    df = df.copy()
    df["is_correct_norm"] = df["is_correct"].astype(str).str.strip().str.lower() == "true"
    df["generated_think_tokens"] = df["generated_think_tokens"].astype(int)

    # Shortest correct per question_id
    shortest_correct = (
        df[df["is_correct_norm"]]
        .sort_values(["question_id", "generated_think_tokens"])
        .groupby("question_id", as_index=False)
        .first()                    # shortest correct
        .drop(columns=["is_correct_norm"])
        .rename(columns=lambda c: f"shortest_{c}" if c != "question_id" else c)
    )

    # Longest any answer per question_id
    longest_any = (
        df.sort_values(["question_id", "generated_think_tokens"], ascending=[True, False])
        .groupby("question_id", as_index=False)
        .first()                    # longest
        .drop(columns=["is_correct_norm"])
        .rename(columns=lambda c: f"longest_{c}" if c != "question_id" else c)
    )
    out = shortest_correct.merge(longest_any, on="question_id", how="inner")

    train_df, val_df = split_train_validation(out, np.arange(0, 10, 1))

    return train_df, val_df

    
def build_dpo_dataset(train_df, tokenizer, eos_token):
    """
    Create DPO-ready dataset entries from a dataframe with:
      - shortest_prompt
      - shortest_generated_text
      - longest_generated_text
    """

    dpo_data = []

    # regex for removing "Think for N tokens."
    think_pattern = re.compile(r"Think for\s+\d+\s+tokens\.\s*", flags=re.IGNORECASE)

    # regex to remove leading text until <think>
    strip_before_think = re.compile(r"(?is)^.*?(?=<think>)")

    for _, row in train_df.iterrows():
        # --- Clean prompt ---
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

        # --- Clean outputs ---
        short_out = row["shortest_generated_text"]
        long_out  = row["longest_generated_text"]

        clean_short = strip_before_think.sub("", short_out).strip() + eos_token
        clean_long  = strip_before_think.sub("", long_out).strip() + eos_token

        # --- Build DPO entry ---
        dpo_data.append({
            "prompt": prompt_text,
            "chosen": clean_short,
            "rejected": clean_long,
        })

    return dpo_data

def print_dpo_dataset(dataset, tokenizer):

    '''
        n_tokens = len(tokenizer(row['rejected'], return_tensors="pt")['input_ids'][0])
        if n_tokens > 2200:
            print(n_tokens)
    '''

    for idx,row in enumerate(dataset):

        print(f"PROMPT: {row['prompt']} \n, CHOSEN {row['chosen']} \n, REJECTED {row['rejected']}\n\n")
        if idx == 5:
            break


def train(train_df, val_df, continue_from: str = None):
    OUTPUT_DIR = Path(__file__).parent / "models"
    model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    model_path = "/scratch/s3799042/DeepScaleR-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

    train_list = build_dpo_dataset(train_df, tokenizer, eos_token=tokenizer.eos_token)
    val_list   = build_dpo_dataset(val_df, tokenizer, eos_token=tokenizer.eos_token)
    dataset     = Dataset.from_list(train_list)
    dataset_val = Dataset.from_list(val_list)

    tokenizer.pad_token = tokenizer.eos_token

    if continue_from is None:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=model_path,
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(base, lora_config)

    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=model_path,
        )

        model = PeftModel.from_pretrained(
            base,
            continue_from,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=1,
        max_length=3_000,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        eval_dataset=dataset_val,
    )

    n_epochs = 20
    for epoch in range(n_epochs):
        trainer.train()
        model.save_pretrained(f"{OUTPUT_DIR}/full_epoch_{epoch+2}")

def prompt():
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Two standard 6-sided dice are tossed. What is the probability that the sum of the numbers shown on the dice is a prime number? Express your answer as a common fraction."
    
    messages = [
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id


    output_ids = model.generate(
        **inputs,
        max_new_tokens=5000,
        temperature=0.001,          # deterministic
        eos_token_id=eos_id, 
        bos_token_id=bos_id  
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(response)


def print_dataset(train_df, val_df):
    print(train_df.columns)
    for idx, row in train_df.iterrows():
        print(f"Question {row['question_id']}: \n SHORTEST PROMPT: {row['shortest_prompt']} \n SHORTEST ANSWER: {row['shortest_generated_text']} \n \n")
        print(f"Question {row['question_id']}: \n LONGEST PROMPT: {row['longest_prompt']} \n LONGEST ANSWER: {row['longest_generated_text']} \n \n")
        if idx == 5:
            break

    print("VALIDATION \n")

    for idx, row in val_df.iterrows():
        print(f"Question {row['question_id']}: \n SHORTEST PROMPT: {row['shortest_prompt']} \n SHORTEST ANSWER: {row['shortest_generated_text']} \n \n")
        print(f"Question {row['question_id']}: \n LONGEST PROMPT: {row['longest_prompt']} \n LONGEST ANSWER: {row['longest_generated_text']} \n \n")
        if idx == 5:
            break


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent.parent.parent / "Data" / "NLP" / "Train" / "data"
    df = pd.read_parquet(data_folder / "math_results.parquet")
    print(df.columns)
    print(len(df['question_id'].unique()))
    half_df = len(df) // 2
    filtered_df = filter_text(df)
    train_df, val_df = get_train_validation(filtered_df)
    train(train_df, val_df, Path(__file__).parent / "models" / "full_epoch_1")
#print_dataset(train_df, val_df)
