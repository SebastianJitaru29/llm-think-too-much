deactivate
rm -r .venv/

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

pip install uv

python -m uv venv --seed
source .venv/bin/activate
pip install uv

# python -m uv pip install --upgrade pip setuptools packaging wheel
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
