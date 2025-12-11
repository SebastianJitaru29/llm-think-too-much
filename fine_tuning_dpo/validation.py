from vllm import LLM, SamplingParams
from fine_tuning_new import filter_text
from fine_tuning_new import get_train_validation
import pandas as pd
from pathlib import Path
from vllm.lora.request import LoRARequest
import re
import torch
import gc

def create_prompts(df, tokenizer):
    think_pattern = re.compile(r"Think for\s+\d+\s+tokens\.\s*", flags=re.IGNORECASE)
    prompts = []
    for _, row in df.iterrows():
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
        print(prompt_text)
        prompts.append(prompt_text)
    
    return prompts

def validation(val_df):
    # Base model and LoRA adapter paths
    base_model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    lora_dir        = "models/full_epoch_1"   # change if needed


    # Sampling
    sampling = SamplingParams(
        max_tokens=2000,
        temperature=0.001,
    )

    # ---- Fine-tuned model (base + LoRA) ----
    print("=== Fine-tuned Output (Base + LoRA) ===")

    llm_fine = LLM(
        model=base_model_name,
        dtype="bfloat16",
        max_model_len=3000,
        enable_lora = True,
        max_lora_rank = 8
    )
    tokenizer = llm_fine.get_tokenizer()
    prompts =create_prompts(val_df, tokenizer)
    out_fine = llm_fine.generate(prompts, sampling, lora_request=LoRARequest("lora_adapter", 1, lora_dir))
    for i, result in enumerate(out_fine):
        text = result.outputs[0].text
        print(f"[FINE] Prompt {i} length: {len(text.split())}")
    #print(text_fine)

    del llm_fine            # remove the reference
    gc.collect()            # run Python garbage collection
    torch.cuda.empty_cache() # free GPU memory
    # ---- Base model only ----
    print("\n=== Original Model Output (Base Only) ===")

    llm_base = LLM(
        model=base_model_name,
        dtype="bfloat16",
        max_model_len=3000,
    )

    out_base = llm_base.generate(prompts, sampling)
    for i, result in enumerate(out_base):
        text = result.outputs[0].text
        print(f"[BASE] Prompt {i} length: {len(text.split())}")
    #print(text_base)


if __name__ == "__main__":
    data_folder = Path(__file__).parent.parent.parent.parent / "Data" / "NLP" / "Train" / "data"
    df = pd.read_parquet(data_folder / "math_results.parquet")
    filtered = filter_text(df)
    train, val = get_train_validation(filtered)
    validation(val)
