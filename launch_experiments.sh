#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=20:00:00
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
echo "Job started on $(hostname) at $(date)"
module purge
#module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home4/s6019595/.llmvenv/bin/activate
srun python3 launch_experiments.py \
  --data ./data/raw/eval_data.parquet \
  --model-path /scratch/s6019595/models/L1-Qwen3-8B-Max/ \
  --generated-dir ./data/processed/eval \
  --file-name eval
echo "Job ended at $(date)"
