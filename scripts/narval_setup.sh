#!/bin/bash
# Run this ONCE on the Narval LOGIN NODE to set up the environment and
# pre-download everything. Do NOT submit this as a job.

module load StdEnv/2023
module load gcc
module load arrow
module load cuda
module load python/3.11

cd ~/cultural_contextualization

# Create virtual environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install dependencies
pip install -r narval_requirements.txt

# Set HuggingFace cache
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_DISABLE_XET=1
mkdir -p $HF_HOME

# Pre-download Qwen model (requires internet — login node only)
echo "Downloading Qwen2.5-7B-Instruct ..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='auto')
print('Qwen model downloaded.')
"

# Pre-download HuggingFace datasets (requires internet — login node only)
echo "Pre-downloading datasets ..."
python -c "
from datasets import load_dataset, get_dataset_config_names

print('Downloading ArabCulture ...')
for cfg in ['Algeria','Egypt','Jordan','KSA','Lebanon','Libya','Morocco','Palestine','Sudan','Syria','Tunisia','UAE','Yemen']:
    load_dataset('MBZUAI/ArabCulture', cfg, split='test')

print('Downloading DIWALI ...')
load_dataset('nlip/DIWALI', split='train')

print('Downloading CultureBank ...')
for split in ['tiktok', 'reddit']:
    load_dataset('SALT-NLP/CultureBank', split=split)

print('Downloading BLEnD ...')
for split in ['DZ','AS','AZ','CN','ET','GR','ID','IR','MX','KP','NG','KR','ES','GB','US','JB']:
    load_dataset('nayeon212/BLEnD', 'short-answer-questions', split=split)

print('All datasets downloaded.')
"

echo "Setup complete. Submit the job with: sbatch scripts/narval_run.sh"
