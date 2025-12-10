from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def validation():
    # Paths
    base_model_name = "agentica-org/DeepScaleR-1.5B-Preview"
    finetuned_path = "models/full_epoch_1"   # adjust if needed
    cache_dir = "/scratch/s3799042/DeepScaleR-1.5B"

    def load(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

    def generate(model, tokenizer, prompt, max_new_tokens=2000):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.001,
                eos_token_id = tokenizer.eos_token_id
            )
        output = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n\n LEN {len(output.split(' '))} \n\n ")
        return output

    # Load models
    tok_fine, model_fine = load(finetuned_path)
    tok_base, model_base = load(base_model_name)

    # Input prompt
    prompt = "<｜begin▁of▁sentence｜><｜User｜>Find the center of the circle with equation $x^2 - 6x + y^2 + 2y = 9$. Let’s think step by step inside and output the final answer within boxed{}. <think><｜Assistant｜>"

    # Outputs
    print("=== Fine-tuned Output ===")
    print(generate(model_fine, tok_fine, prompt))

    print("\n=== Original Model Output ===")
    print(generate(model_base, tok_base, prompt))

validation()

if __name__ == ("__main___"):
    validation()