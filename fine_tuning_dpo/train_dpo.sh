#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=dpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
module purge
module load CUDA/12.4.0
source /home3/s3799042/venvs/venv_think/bin/activate
srun python3 /home3/s3799042/llm-think-too-much/fine_tuning_dpo/fine_tuning_new.py