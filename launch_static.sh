#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --job-name=static_regressor
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/static_regressor.out
#SBATCH --error=logs/static_regressor.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

srun python launch_regressor.py --model-path ./models/L1-Qwen3-8B-Max/ --generated-dir ./data