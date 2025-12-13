#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=34:00:00
#SBATCH --job-name=math1  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/math1.out
#SBATCH --error=logs/math1.err

echo "Job started on $(hostname) at $(date)"
module purge
#module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source /home4/s6019595/.llmvenv/bin/activate
srun python3 launch_experiments.py \
  --data ./data/math_part1.parquet \
  --model-path /scratch/s6019595/models/L1-Qwen3-8B-Max/ \
  --generated-dir ./data/generated \
  --file-name math1_results_long
echo "Job ended at $(date)"
