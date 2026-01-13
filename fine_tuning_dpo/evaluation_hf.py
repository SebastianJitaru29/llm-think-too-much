import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning_new import filter_text
from fine_tuning_new import get_train_validation
from math_equivalence import is_equiv
from huggingface_hub import snapshot_download


def extract_boxed(s: str) -> str | None:
    if not s:
        return None
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
        if is_math_500:
            clean_prompt = (
                row["problem"]
                + "Let’s think step by step inside and output the final answer within boxed{{}}"
            )
        else:
            raw_prompt = row["shortest_prompt"]
            clean_prompt = think_pattern.sub("", raw_prompt).strip()

        messages = [{"role": "user", "content": clean_prompt}]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        print(prompt_text)
        prompts.append(prompt_text)

    return prompts


def run_hf_generation(model, tokenizer, prompts, max_tokens=3000, temperature=0):
    outputs = []
    for idx, prompt in enumerate(prompts):

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        gen = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        # mimic vLLM output structure
        class Result:
            pass
        class Output:
            pass
        r = Result()
        o = Output()
        o.text = text
        r.outputs = [o]
        outputs.append(r)
        print(f"progress: {idx}")
    return outputs


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
        token_length.append(
            len(tokenizer(text, add_special_tokens=False)["input_ids"])
        )

    results = pd.DataFrame(
        {
            "id": question_ids,
            "text": texts,
            "length": token_length,
            "is_correct": is_correct,
        }
    )
    print(results)

    filename = "TokenSkip-7B.parquet"
    path = Path(__file__).parent / filename
    results.to_parquet(path)


def val_base_model(val_df, is_math_500):
    base_model_name = "hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K"
    local_path = "/scratch/s3799042/TokenSkip-7B"

    if local_path == "":
        local_path = base_model_name

    model_dir = snapshot_download(
        repo_id=base_model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    prompts = create_prompts(val_df, tokenizer, is_math_500)

    out = run_hf_generation(model, tokenizer, prompts)

    save_results(val_df, out, tokenizer, base_model_name)

    del model
    gc.collect()
    torch.cuda.empty_cache()


def val_fine_tuned_model(val_df, is_math_500):
    base_model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    lora_dir = "models/full_epoch_4"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, lora_dir)

    prompts = create_prompts(val_df, tokenizer, is_math_500)

    out = run_hf_generation(model, tokenizer, prompts)

    save_results(val_df, out, tokenizer, base_model_name)

    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    val = pd.read_json(
        "hf://datasets/HuggingFaceH4/MATH-500/test.jsonl",
        lines=True
    )

    val_base_model(val, True)
    # val_fine_tuned_model(val, True)
