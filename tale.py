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

MODEL = "/scratch/s3799042/Qwen3-8B/"
OUTPUT_PATH = Path(__file__).parent / "data" / "processed" / "results"/ "TALE_eval_template.parquet"
TEST_PROBLEMS = Path(__file__).parent / "data" / "processed" / "eval_L1_bert" / "gsm8k_olympiad_amc.parquet" #"dataset_splitting" / "test.parquet" #"eval_L1_bert" / "gsm8k_olympiad_amc.parquet" # "old" / "dataset_splitting" / "test.parquet"

DIR_TARGET_TOKENS = Path(__file__).parent / "data" / "processed" / "tale_target_tokens"
IS_EVAL = True

def get_token_buget_prompt(question, tokenizer):
    to_remove = "Let’s think step by step inside and output the final answer within boxed{}."

    cleaned_text = question.replace(to_remove, "")

    prompt = (
            f"Q: {cleaned_text}"
            "Task: Analyze the given question and estimate the minimum number of tokens "
            "required to generate a complete and accurate response. "
            "Please Give the response by strictly following this format: [[budget]]. "
            "for example, Budget: [[12]].\n\n"
            
        )

    messages = [
        {"role": "user", "content": prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    return prompt

def build_prompt(problem: str, target_think_tokens: int, tokenizer, use_chat_template = False) -> str:
    if IS_EVAL:
        problem = f"{problem} Let’s think step by step inside and output the final answer within boxed{{}}."

    prompt = f"{problem} Use less than {target_think_tokens} tokens."


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

def evaluate_answer(expected_answer: str, generated_answer: str) -> bool:
    if IS_EVAL == False:
        exp_val = extract_boxed(expected_answer)
    else:
        exp_val = expected_answer
    gen_val = extract_boxed(generated_answer)

    if exp_val is None:
        exp_val = expected_answer.strip()

    if exp_val is None:
        print("Expected value is none")

    
    if gen_val is None:
        print("Generated value is none")
        print(generated_answer)
        
    return is_equiv(gen_val, exp_val)


def main():
    df = pd.read_parquet(TEST_PROBLEMS)

    df = df.rename(columns={'id': 'question_id'})
    df = df.drop_duplicates(subset='question_id', keep='first')

    #os.makedirs(args.generated_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    llm = LLM(
        model=MODEL,
        dtype="float16", 
        trust_remote_code=True,
        tensor_parallel_size=1, 
        max_model_len=32768,    
    )

    sampling_params = SamplingParams(max_tokens=20_000, temperature=0, skip_special_tokens=False)


    token_budget_prompt = []
    for idx in range(len(df)):
        token_budget_prompt.append(get_token_buget_prompt(df.iloc[idx]['prompt'], tokenizer))
        if idx == 0:
            print(df.iloc[idx]['prompt'])
            print("CLEANED")
            print(token_budget_prompt[0])

    outputs_targets = llm.generate(token_budget_prompt, sampling_params)

    targets = []
    output_texts = []

    for output in outputs_targets:
        if len(output.outputs) > 0:
            output_target = output.outputs[0].text
            output_texts.append(output_target)

            match = re.search(r"\[\[(\d+)\]\]", output_target)
            if match:
                targets.append(int(match.group(1)))
            else:
                targets.append(100)
        else:
            output_texts.append("")
            targets.append(100)

    data = {
        "targets": np.array(targets),
        "outputs": np.array(output_texts)
    }

    np.save(DIR_TARGET_TOKENS / "results_no_thinking_reversed.npy", data)

    prompts = []
    for i in range(len(df)):
        print(targets[i])
        prompts.append(build_prompt(df.iloc[i]["prompt"], targets[i], tokenizer, use_chat_template=True))


    print(prompts[0])

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    
    generated_texts = []
    think_texts = []
    
    for i, output in enumerate(outputs):
        prompt_text = prompts[i]
        generated_suffix = output.outputs[0].text
        full_text = prompt_text + generated_suffix
        generated_texts.append(full_text)
        think_texts.append(extract_think_text(full_text))

    think_encodings = tokenizer(think_texts, add_special_tokens=False)["input_ids"]
    think_lengths = [len(ids) for ids in think_encodings]

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        full_text = generated_texts[i]
        #print(full_text)
        is_ok = evaluate_answer(row["solution_col"], full_text)

        records.append({
            "question_id": row['question_id'],
            "prompt": prompts[i],
            "solution": row["solution_col"],
            "generated_think_text": think_texts[i],
            "generated_text": full_text,
            "target_think_tokens": int(targets[i]),
            "generated_think_tokens": think_lengths[i],
            "latency_sec": (t1 - t0) / len(prompts),
            "is_correct": bool(is_ok),
        })

    final_df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_PATH)
    final_df.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()