import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
import csv
from pathlib import Path
from ..launch_experiments import evaluate_answer
import pandas as pd
from peft import PeftModel
import re 

TEST_DATA_FOLDER = Path(__file__).parent.parent / "test.parquet"

ORIGINAL_MODEL_PATH = "agentica-org/DeepScaleR-1.5B-Preview"
THINK_PRUNE = "Shiyu-Lab/DeepScaleR-1.5B-Preview-thinkprune-iter3k"
EPOCH = "epoch_2"
MODEL_FINETUNED_PATH = Path(__file__).parent.parent / "qwen-1.5B-dpo-lora" / EPOCH 

def clean_special_tokens(text):
    text = text.replace("<｜end▁of▁sentence｜>", "")
    parts = text.split("<｜begin▁of▁sentence｜>", 2) 
    if len(parts) >= 2:
        text ="<｜begin▁of▁sentence｜>" + parts[1] 

    text = text + "<｜end▁of▁sentence｜>"
    
    return text

def remove_think_string(string):
    string = re.sub(r" Think for \d+ tokens\. \<think\>", "", string)
    string = re.sub(r" Think for \d+ tokens\.", "", string)
    return string

def read_data(path):
    data = {}
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        qid = row['question_id']
        prompt = remove_think_string(row['prompt'])
        if row["solution_col"] != None:
            answer = row['solution_col']
        if row["expected_answer"] != None:
            answer = row['expected_answer']
        level = row['level']
        if prompt not in data:
            data[prompt] = (qid, answer, level)
    return data


def evaluate_model_on_test_data(model, tokenizer, test_data, folder, batch_size=50, output_prefix="results"):
    prompts = [k for k, v in test_data.items()]

    batch_count = 100
    for batch_start in range(57, 58, batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            sequences = model.generate(
                **inputs,
                max_new_tokens=6000,
                do_sample=False,
                top_p=1.0,
                temperature=0.0
            )

        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)

        batch_rows = []
        for seq_tensor, answer_str, prompt in zip(sequences, decoded, batch_prompts):
            think_text = clean_special_tokens(answer_str)
            num_tokens = len(tokenizer.encode(think_text))

            qid, correct_answer, level = test_data[prompt]
            is_correct = evaluate_answer(correct_answer, answer_str)

            batch_rows.append({
                'question_id': qid,
                'prompt': prompt,
                'correct_answer': correct_answer,
                'generated': answer_str,
                'number_of_tokens': num_tokens,
                'is_correct': is_correct,
                'level': level
            })

        # sanity check: batch_rows length should equal len(batch_prompts)
        if len(batch_rows) != len(batch_prompts):
            raise RuntimeError(f"batch size mismatch: got {len(batch_rows)} rows for {len(batch_prompts)} prompts")

        # save this batch and clear
        batch_count += 1
        filename = f"{output_prefix}_batch_{batch_count}.csv"
        with open(folder / filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(batch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(batch_rows)

    # done

def evaluate_model(model_path, test_data_folder, output_folder, lora_path ="", epoch = ""):
    if epoch == "": 
        results_folder = Path(__file__).parent / "results" / output_folder
    else:
        results_folder =  Path(__file__).parent / "results" / output_folder / epoch

    os.makedirs(results_folder, exist_ok=True)
    test_data = read_data(test_data_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  
        torch_dtype="auto"  # loads weights in FP16 if available
    )

    if lora_path != "":
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    evaluate_model_on_test_data(model, tokenizer, test_data, results_folder, batch_size = 1)

#evaluate_model(ORIGINAL_MODEL_PATH, TEST_DATA_FOLDER, "dpo", MODEL_FINETUNED_PATH, EPOCH)
evaluate_model(THINK_PRUNE, TEST_DATA_FOLDER, "thinkprune")
#evaluate_model(ORIGINAL_MODEL_PATH, TEST_DATA_FOLDER, "base")



