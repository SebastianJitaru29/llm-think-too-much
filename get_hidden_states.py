import argparse
import os
import numpy as np
import pandas as pd
import torch

from vllm import LLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    problems = df["problem"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        dtype="float16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=32768,
    )

    model = llm.get_model()
    model.eval()

    hidden_states = []

    for problem in problems:
        enc = tokenizer(problem, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )

        last_layer = out.hidden_states[-1]      # [1, seq_len, hidden_dim]
        last_token = last_layer[0, -1, :]       # [hidden_dim]
        hidden_states.append(last_token.cpu().numpy().astype(np.float16))

    hidden_states = np.stack(hidden_states, axis=0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, hidden_states)

    print(f"Saved hidden states to {args.output}")


if __name__ == "__main__":
    main()
