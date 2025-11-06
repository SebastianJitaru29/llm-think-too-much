#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=static_regressor
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/static_regressor.out
#SBATCH --error=logs/static_regressor.err

module purge
module load meson-python/0.18.0-GCCcore-14.2.0
source .venv/bin/activate
srun python launch_regressor.py --type static --batch 20