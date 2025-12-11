#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=0:25:00
#SBATCH --job-name=hidden_states
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/hidden_states.out
#SBATCH --error=logs/hidden_states.err

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

source .venv/bin/activate
srun python get_hidden_states.py --data ./data/math.parquet --model ./models/L1-Qwen3-8B-Max/ --output ./regressor/innit_hidden_states/hidden_states_math.npy --batch-size 32
