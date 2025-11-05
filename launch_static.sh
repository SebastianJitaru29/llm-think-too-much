#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name=static_regressor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/static_regressor.out
#SBATCH --error=logs/static_regressor.err

module purge
module load meson-python/0.18.0-GCCcore-14.2.0
source ./venv/bin/activate
srun python static_regressor.py