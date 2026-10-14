#!/bin/bash
#SBATCH --job-name=hw2-finalA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/slurm-%j.out
# Tuning (Q3.4) was already done locally; configs/final_A.yaml is shipped with the code.
# Submit from inside the A2 directory: sbatch hpc/train_final_A.sbatch
set -e
export WANDB_MODE=${WANDB_MODE:-offline}
PY=.venv/bin/python
nvidia-smi

# Q3.5: train the best configuration with ~1e17 FLOPs
$PY hpc/run_train.py configs/final_A.yaml

# Q4.2: generations from our own model at several temperatures
for t in 0.1 0.5 1.0; do
  echo "=== generate T=$t"
  $PY src/lm/generate.py --config configs/final_A.yaml --prefixes hpc/prefixes3.jsonl \
    --max_new_tokens 48 --temperature $t | tee outputs/final_A/generate_T$t.log
  cp outputs/final_A/generation.jsonl outputs/final_A/generation_T$t.jsonl
done
echo "ALL DONE"
