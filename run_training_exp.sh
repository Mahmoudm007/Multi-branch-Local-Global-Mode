#!/bin/bash
# Sequential benchmark launcher for SDRE.
#SBATCH --job-name=TrainingAllModels
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:8
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --time=6-23:00:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.txt

set -euo pipefail

mkdir -p outputs errors

module reset
module load StdEnv/2023
module load gcc/12.3
module load python/3.10
module load opencv/4.10.0


VENV_PATH="${VENV_PATH:-${VENV_DIR:-$HOME/projects/p65425/mmabdela/mma_venv}}"
source "$VENV_PATH/bin/activate"

python -m pip install --no-index -r requirements.txt


export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CLB_PRETRAINED_CACHE_DIR="${CLB_PRETRAINED_CACHE_DIR:-$PWD/weights/pretrained_cache}"
export HF_HOME="$CLB_PRETRAINED_CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$CLB_PRETRAINED_CACHE_DIR/torch"
export TIMM_HOME="$CLB_PRETRAINED_CACHE_DIR/timm"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

CONFIG_PATH="${CONFIG_PATH:-configs/multibranch_default.yaml}"

mkdir -p weights "$HF_HUB_CACHE" "$TORCH_HOME" "$TIMM_HOME" "$TRANSFORMERS_CACHE" Output/_global_comparison_per_combination

echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Config: $CONFIG_PATH"
echo "Python: $(which python)"
nvidia-smi -L || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

python -u run_training_exp.py --config configs/multibranch_default.yaml --skip-completed true
 
