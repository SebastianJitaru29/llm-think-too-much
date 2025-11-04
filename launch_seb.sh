#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --job-name=nlp_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/nlp_job.out
#SBATCH --error=logs/nlp_job.err

echo "Job started on $(hostname) at $(date)"
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source ../.venv/bin/activate
srun python3 launch_experiments.py   --data ./data/dataset.parquet   --model-path ./models/L1-Qwen-1.5B-Exact   --generated-dir ./generated_math   --hidden-dir ./hidden_math --batch-size 64