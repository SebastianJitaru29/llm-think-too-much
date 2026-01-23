#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=0:45:00
#SBATCH --job-name=hidden_states
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/hidden_states.out
#SBATCH --error=logs/hidden_states.err

# module purge
# module load Python/3.11.5-GCCcore-13.2.0
# module load CUDA/12.1.1

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=MIG-76de4d8d-dd90-510e-bc47-b5feb5c965b3
# python get_hidden_states.py --data ./data/actual_train.parquet --model ./models/L1-Qwen3-8B-Max/ --output ./regressor/innit_hidden_states/hidden_states_train.npy --batch-size 16
python get_hidden_states.py --data ./data/actual_test.parquet --model ./models/L1-Qwen3-8B-Max/ --output ./regressor/innit_hidden_states/hidden_states_test.npy --batch-size 16
# python get_hidden_states.py --data ./data/aime.parquet --model ./models/L1-Qwen3-8B-Max/ --output ./regressor/innit_hidden_states/hidden_states_aime.npy --batch-size 8
# python get_hidden_states.py --data ./data/test_all.parquet --model ./models/L1-Qwen3-8B-Max/ --output ./data/hidden_states_test.npy --batch-size 8