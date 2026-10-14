#!/bin/bash
#SBATCH --job-name=hw2-cfg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/slurm-%j.out
# Train + generate for an arbitrary config: sbatch -J <name> hpc/train_cfg.sbatch configs/<name>.yaml
set -e
export WANDB_MODE=${WANDB_MODE:-offline}
PY=.venv/bin/python
CFG=$1
OUT=$($PY -c "from omegaconf import OmegaConf; print(OmegaConf.load('$CFG').output_dir)")
nvidia-smi

$PY hpc/run_train.py $CFG

for t in 0.1 0.5 1.0; do
  echo "=== generate T=$t"
  $PY src/lm/generate.py --config $CFG --prefixes hpc/prefixes3.jsonl \
    --max_new_tokens 48 --temperature $t | tee $OUT/generate_T$t.log
  cp $OUT/generation.jsonl $OUT/generation_T$t.jsonl
done
echo "ALL DONE"
