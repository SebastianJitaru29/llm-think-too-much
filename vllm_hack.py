import re
from pathlib import Path

FILE_PATH = Path("/scratch/s6019595/llm-think-too-much/.venv/lib/python3.11/site-packages/vllm/platforms/cuda.py")
GPU_UUID = "GPU-5a700359-a3ea-8409-2b6c-1b49daf556ae"

def patch_cuda_py():
    if not FILE_PATH.exists():
        print(f"Error: {FILE_PATH} not found.")
        return

    content = FILE_PATH.read_text()

    # Matches any nvmlDeviceGetHandleByIndex call, with or without pynvml prefix
    pattern = r"(?:pynvml\.)?nvmlDeviceGetHandleByIndex\s*\([^)]*\)"
    replacement = f"pynvml.nvmlDeviceGetHandleByUUID(b'{GPU_UUID}')"

    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        FILE_PATH.write_text(new_content)
        print(f"Successfully patched {FILE_PATH} with GPU UUID.")
    else:
        print("No changes made. Pattern not found.")

if __name__ == "__main__":
    patch_cuda_py()