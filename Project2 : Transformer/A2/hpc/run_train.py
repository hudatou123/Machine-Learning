"""Runs lm.train.main() while also dumping every wandb.log call to <output_dir>/log.jsonl."""
import json, os, sys
import wandb
from omegaconf import OmegaConf
import lm.train as T

cfg = OmegaConf.load(sys.argv[1])
os.makedirs(cfg.output_dir, exist_ok=True)
log_f = open(os.path.join(cfg.output_dir, "log.jsonl"), "w")
_orig_init = wandb.init
def _init(*a, **kw):
    run = _orig_init(*a, **kw)
    orig_log = wandb.log
    def _log(d, step=None, **k):
        log_f.write(json.dumps({"step": step, **d}) + "\n"); log_f.flush()
        return orig_log(d, step=step, **k)
    wandb.log = _log
    return run
wandb.init = _init
T.main()
