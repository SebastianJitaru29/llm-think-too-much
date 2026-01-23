#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval_gsm8k%j.out
#SBATCH --error=logs/eval_gsm8k%j.err
echo "Job started on $(hostname) at $(date)"
module purge
#module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home4/s6019595/.llmvenv/bin/activate
srun python3 evaluation/eval_pipeline.py \
  --model-path /scratch/s6019595/models/Qwen3-8B/ \
  --datasets gsm8k \
  --enable-thinking 
echo "Job ended at $(date)"