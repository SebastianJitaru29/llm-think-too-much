#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=aime_job_long
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/aime_job_long.out
#SBATCH --error=logs/aime_job_long.err

echo "Job started on $(hostname) at $(date)"
module purge
#module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source .llmvenv/bin/activate
srun python3 launch_experiments.py \
  --data ./data/aime.parquet \
  --model-path ./models/L1-Qwen3-8B-Max \
  --generated-dir ./data/generated \
  --file-name aime_results_long
echo "Job ended at $(date)"
