import argparse
import os
import numpy as np
import pandas as pd
import tqdm
import torch
import re
from pathlib import Path
import pandas as pd
# CHANGED: Import AutoModel instead of AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel

data = Path(__file__).parent.parent / "dataset_splitting" / "train.parquet"
model_path = "/scratch/s3799042/bert-base-uncased"
output = "/scratch/s3799042/results_nlp/train_hidden_states"
batch_size = 32

def read_data():
    df = pd.read_parquet(data)
    df_unique = df.drop_duplicates(subset="question_id", keep="first")
    print(f"loaded {len(df_unique)} unique prompts")
    return df_unique["prompt"].astype(str).tolist(), df_unique["question_id"]

def main():

    """
    df = pd.read_parquet(data)

    if "problem" in df.columns:
        problems = df["problem"].astype(str).tolist()
    elif "Question" in df.columns:
        problems = df["Question"].astype(str).tolist()
    elif "prompt" in df.columns:
        problems = df["prompt"].astype(str).tolist()
        problems = [
            re.sub(r" Let’s think step by step inside and output the final answer within boxed\{\}\. Think for \d+ tokens\. <think>", "", problem)
            for problem in problems
        ]
        problems = list(set(problems))
        problems = [problem for problem in problems if len(problem) < 400]
    else:
        raise ValueError("No Question or problem column found.")
    """
    problems, ids = read_data()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir="/scratch/s3799042/",
    )

    model = AutoModel.from_pretrained(
        "bert-base-uncased",
        cache_dir="/scratch/s3799042/",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # 1. Filter out sequences longer than 512 tokens (including specials)
    # ------------------------------------------------------------------
    filtered_problems = []
    filtered_ids = []

    for problem, pid in zip(problems, ids):
        n_tokens = len(
            tokenizer(
                problem,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if n_tokens <= 512:
            filtered_problems.append(problem)
            filtered_ids.append(pid)

    problems = filtered_problems
    ids = filtered_ids

    print(f"Kept {len(problems)} samples after length filtering")

    # ------------------------------------------------------------------
    # 2. Compute CLS embeddings
    # ------------------------------------------------------------------
    hidden_states = []

    for i in tqdm.tqdm(range(0, len(problems), batch_size)):
        batch = problems[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )

        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # CLS token embedding
        cls_embeddings = out.last_hidden_state[:, 0, :]

        hidden_states.append(cls_embeddings.cpu().numpy().astype(np.float16))

    hidden_states = np.concatenate(hidden_states, axis=0)

    # ------------------------------------------------------------------
    # 3. Save embeddings together with ids
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output), exist_ok=True)

    np.save(
        output,
        {
            "ids": np.array(ids),
            "embeddings": hidden_states,
        },
    )

    print(f"Saved {len(ids)} BERT embeddings with ids to {output}")

if __name__ == "__main__":
    main()