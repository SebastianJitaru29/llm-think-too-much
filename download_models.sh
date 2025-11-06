mkdir -p models && \
huggingface-cli download l3lab/L1-Qwen-1.5B-Exact --local-dir models/L1-Qwen-1.5B-Exact && \
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir models/DeepSeek-R1-Distill-Qwen-1.5B