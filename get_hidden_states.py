import argparse
import os
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    if "problem" in df.columns:
        problems = df["problem"].astype(str).tolist()
    elif "Question" in df.columns:
        problems = df["Question"].astype(str).tolist()
    else:
        assert False, "No Question or problem"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    hidden_states = []

    for problem in problems:
        enc = tokenizer(problem, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # final layer hidden states: [1, seq_len, hidden_dim]
        last_layer = out.hidden_states[-1]
        last_token = last_layer[0, -1, :]   # [hidden_dim]

        hidden_states.append(last_token.cpu().numpy().astype(np.float16))

    hidden_states = np.stack(hidden_states, axis=0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, hidden_states)

    print(f"Saved hidden states to {args.output}")


if __name__ == "__main__":
    main()
