#!/bin/bash
#SBATCH --mem=50GB
#SBATCH --time=04:00:00
#SBATCH --job-name=nlp_experiments
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home4/s6019595/logs/nlp_experiments_%j.out
#SBATCH --error=/home4/s6019595/logs/nlp_experiments_%j.err

echo "Job started on $(hostname) at $(date)"

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# ------------------------------------------------------------
# 1. GO TO PROJECT FOLDER (Fixes "File not found" errors)
# ------------------------------------------------------------
cd /projects/s6019595/llm-think-too-much

# Verify we are in the right place
echo "Current directory: $(pwd)"

# 2. Activate Environment
source .lmenv/bin/activate

# 3. Run Script
# We use 'python3' because the venv is active.
# We use relative paths (./data) because we cd'd into the folder above.
srun python3 launch_experiments_final.py \
  --data ./data/math.parquet \
  --model-path ./models/L1-Qwen3-8B-Max \
  --generated-dir ./generated_math_final