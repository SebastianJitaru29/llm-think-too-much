import argparse
import os
import numpy as np
import pandas as pd
import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    if "problem" in df.columns:
        problems = df["problem"].astype(str).tolist()
    elif "Question" in df.columns:
        problems = df["Question"].astype(str).tolist()
    else:
        raise ValueError("No Question or problem column found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    batch_size = args.batch_size
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
                output_hidden_states=True,
                use_cache=False,
            )

        batch_last_tokens = out.hidden_states[-1][:, -1, :]

        hidden_states.append(batch_last_tokens.cpu().numpy().astype(np.float16))

    hidden_states = np.concatenate(hidden_states, axis=0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, hidden_states)

    print(f"Saved hidden states to {args.output}")


if __name__ == "__main__":
    main()
