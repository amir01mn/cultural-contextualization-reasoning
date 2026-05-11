#!/bin/bash
#SBATCH --account=ACCOUNT_NAME          # replace with your Alliance account
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=cultural_kg
#SBATCH --output=logs/job_%j.log

# ── Environment ───────────────────────────────────────────────────────────────
module load StdEnv/2023
module load gcc
module load arrow
module load cuda
module load python/3.11

cd ~/cultural_contextualization
source venv/bin/activate

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_DISABLE_XET=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── Phase 1-3: Mine + cluster + induce schemas ────────────────────────────────
echo "=== Starting Phase 1-3: Mining and schema induction ==="
python -u src/induction_main.py \
  --datasets arabculture diwali culturebank blend candle \
  --candle-path datasets/candle \
  --sample 500 \
  --backend hf

# ── Phase 4: Final KG construction ───────────────────────────────────────────
echo "=== Starting Phase 4: Final KG construction ==="
python -u src/final_main.py \
  --datasets arabculture diwali culturebank blend candle \
  --candle-path datasets/candle \
  --sample 500 \
  --eps 0.65 \
  --backend hf \
  --no-viz

echo "=== Done ==="
