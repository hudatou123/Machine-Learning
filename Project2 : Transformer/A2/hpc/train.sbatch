#!/bin/bash
#SBATCH --job-name=hw2-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/slurm-%j.out
# Submit from inside the A2 directory: sbatch hpc/train.sbatch
set -e
export WANDB_MODE=${WANDB_MODE:-offline}
PY=.venv/bin/python
nvidia-smi

# Q3.4: hyperparameter search, each config <= 1e15 FLOPs
for cfg in configs/tuning/*.yaml; do
  echo "=== tuning $cfg"
  $PY hpc/run_train.py "$cfg"
done

# Q3.5: train the best configuration with ~1e17 FLOPs
$PY hpc/make_final_config.py
$PY hpc/run_train.py configs/final.yaml

# Q4.2: generations from our own model at several temperatures
for t in 0.1 0.5 1.0; do
  echo "=== generate T=$t"
  $PY src/lm/generate.py --config configs/final.yaml --prefixes hpc/prefixes3.jsonl \
    --max_new_tokens 48 --temperature $t | tee outputs/final/generate_T$t.log
  cp outputs/final/generation.jsonl outputs/final/generation_T$t.jsonl
done
echo "ALL DONE"
