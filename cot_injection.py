import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./models/L1-Qwen-1.5B-Exact"
def run_inference(model_LCPO, tokenizer, device, prompt, limit):
    print(f"=======================================================\n{prompt}\n=====================================================")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Innitialize cache 
    with torch.no_grad():
        outputs = model_LCPO(**inputs)
    past_key_values = outputs.past_key_values
    generated = inputs.input_ids
    # Generate until EOS token
    while limit!=0:
        with torch.no_grad():
            out = model_LCPO(
                input_ids=generated[:, -1:],  # Start from laft logit of already processed prompt
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values
        next_token = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        # EOS check
        if next_token.item() == tokenizer.eos_token_id:
            print("Stopped: EOS emitted")
            break
        limit -= 1
    return tokenizer.decode(generated[0], skip_special_tokens=True), generated.shape[1] - inputs.input_ids.shape[1]

def inject_target(prompt:str, target:int) -> str:
    pattern = r"Think for\s+\d+\s+tokens"
    replacement = f"Think for {target} tokens"
    updated_prompt = re.sub(pattern, replacement, prompt)
    return updated_prompt

def main():
    model_LCPO = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    target_tokens = 2500
    prompt = f"Solve: What is 23 * 19 * 30?, Think for {target_tokens} tokens."
    
    limit = 500
    full_text, tokens = run_inference(model_LCPO, tokenizer, device, prompt, limit)
    print(f"Actual | Expected number of tokens: {tokens} | {target_tokens}")
    print("====================================")
    print(f"All generated text:\n{full_text}")
    
    full_text1, tokens1 = run_inference(model_LCPO, tokenizer, device, inject_target(full_text, 800), 1000)
    print(f"Actual | Expected number of tokens: {tokens1} | 300")
    print("====================================")
    print(f"All generated text:\n{full_text1}")
    print(tokens1+tokens)
if __name__ == "__main__":
    main()

