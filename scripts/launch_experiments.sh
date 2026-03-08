#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=60:00:00
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4 
#SBATCH --output=logs/train_long_%j.out
#SBATCH --error=logs/train_long_%j.err
echo "Job started on $(hostname) at $(date)"
module purge
#module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home4/s6019595/.llmvenv/bin/activate
cd
srun python3 /home4/s6019595/llm-think-too-much/launch_experiments.py \
  --data /home4/s6019595/llm-think-too-much/data/raw/train_data.parquet \
  --model-path /scratch/s6019595/models/L1-Qwen3-8B-Max/ \
  --generated-dir /home4/s6019595/llm-think-too-much/data/processed \
  --file-name train
echo "Job ended at $(date)"
