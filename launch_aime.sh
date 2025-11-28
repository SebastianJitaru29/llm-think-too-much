#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name=nlp_experiments
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/nlp_experiments_first.out
#SBATCH --error=logs/nlp_experiments_first.err

echo "Job started on $(hostname) at $(date)"
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source .lmenv/bin/activate
srun python3 launch_aime_experiments.py \
  --data ./data/math.parquet \
  --model-path ./models/L1-Qwen3-8B-Max \
  --generated-dir ./generated_math \

