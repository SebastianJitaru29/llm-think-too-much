import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
import pandas as pd
import re
from pathlib import Path


BASE_MODEL = "agentica-org/DeepScaleR-1.5B-Preview"
TRAIN_PATH = Path(__file__).parent.parent / "train.parquet" 
OUTPUT_DIR = "./qwen-1.5B-dpo-lora"

def read_data(train_path):
    results = {}

    df = pd.read_parquet(train_path)
    for _, row in df.iterrows():
        qid = row['question_id']
        level = row["level"]
        tokens = int(row['generated_think_tokens'])
        correct = row['is_correct'].strip().lower() == 'true'

        if (qid, level) not in results:
            results[(qid, level)] = {'min_true': None, 'max_any': None}

        if correct:
            cur = results[(qid, level)]['min_true']
            if cur is None or tokens < cur['generated_think_tokens']:
                results[(qid, level)]['min_true'] = row
                row['generated_think_tokens'] = tokens

        cur = results[(qid, level)]['max_any']
        if cur is None or tokens > cur['generated_think_tokens']:
            results[(qid, level)]['max_any'] = row
            row['generated_think_tokens'] = tokens

    for (qid, level), vals in list(results.items()):
        a = vals['min_true']
        b = vals['max_any']
        
        if a is None or b is None:
            continue
        if a['generated_think_tokens'] == b['generated_think_tokens']:
            results[(qid, level)] = None
    
    return results

# L1 produces often the same answer copied after each other in order to get to the
# target tokens
def remove_duplicate_answers(text):
    split = re.split(r"<\/think>", text)
    after_last_think = split[-1]
    before_last_think = split[:-1][0] + "</think>"

    result = re.search(r"^(.*?\\boxed\{.*?\})", after_last_think, flags=re.DOTALL)
    if result:
        return before_last_think + result[0]
    
    return ""

# Often the <｜end▁of▁sentence｜> and "<｜begin▁of▁sentence｜>" is copied an absured amount of times
# in order to reach the desired amount of tokens
def clean_special_tokens(text):
    text = text.replace("<｜end▁of▁sentence｜>", "")
    parts = text.split("<｜begin▁of▁sentence｜>", 2) 
    if len(parts) >= 2:
        text ="<｜begin▁of▁sentence｜>" + parts[1] 

    text = text + "<｜end▁of▁sentence｜>"
    
    return text

def clean_generated_text(text):
    #text = remove_duplicate_answers(text)
    text = clean_special_tokens(text)
    return text

def clean_data_rows(data_rows):
    for row in data_rows:
        row['prompt'] = re.sub(r"Think for \d+ tokens\. \<think\>", "", row['prompt'])
        row['prompt'] = re.sub(r"Think for \d+ tokens\.", "", row['prompt'])
        row['chosen'] = clean_generated_text(row['chosen'])
        row['rejected'] = clean_generated_text(row['rejected'])

    return data_rows

def contains_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def remove_chinese_answers(data_rows):
    new_data_rows = []
    count_chinese = 0
    for row in data_rows:
        if contains_chinese(row['chosen']) or contains_chinese(row['rejected']):
            count_chinese += 1
            continue
        new_data_rows.append(row)

    print(f"{count_chinese}  chinese answers removed")
    return new_data_rows

def remove_empty_values(data_rows):
    new_data_rows = []
    count_empty = 0
    for row in data_rows:
        if row['chosen'] == "" or row['rejected']=="":
            count_empty += 1
            continue
        new_data_rows.append(row)

    print(f"{count_empty}  empty answers removed")
    return new_data_rows

def convert_data_to_dataset(data: dict):
    data_rows = []
    for (qid, level), vals in data.items():
        if not vals or vals.get('min_true') is None or vals.get('max_any') is None:
            continue
        if vals['min_true']['generated_think_tokens'] == vals['max_any']['generated_think_tokens']:
            continue

        data_rows.append({
            'prompt': vals['min_true']['prompt'],
            'chosen': vals['min_true']['generated_text'],
            'rejected': vals['max_any']['generated_text']
        })

    data_rows = clean_data_rows(data_rows)
    data_rows = remove_empty_values(data_rows)
    data_rows = remove_chinese_answers(data_rows)
    dataset = Dataset.from_list(data_rows)
    return dataset


def create_dataset():
    data = read_data(TRAIN_PATH)
    dataset = convert_data_to_dataset(data)
    return dataset

def fine_tune_model():
    dataset = create_dataset()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=1,
        max_length=10_000,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # if None, uses a frozen copy of base model
        args=dpo_args,
        train_dataset=dataset,
    )

    n_epochs = 20
    for epoch in range(n_epochs):
        trainer.train()
        model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch+1}")




fine_tune_model()


