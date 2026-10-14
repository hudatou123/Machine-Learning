"""Pick the tuning config with the lowest val perplexity and scale its training to ~1e17 FLOPs."""
import glob, json, os
from omegaconf import OmegaConf
from lm.model import DecoderLM

results = []
for cfg_path in sorted(glob.glob("configs/tuning/*.yaml")):
    cfg = OmegaConf.load(cfg_path)
    eval_path = os.path.join(cfg.output_dir, "eval.json")
    if os.path.exists(eval_path):
        ppl = json.load(open(eval_path))["val-perplexity"]
        results.append((ppl, cfg_path, cfg))
        print(f"{cfg_path}: val ppl = {ppl:.2f}")

ppl, best_path, best = min(results, key=lambda r: r[0])
print(f"best: {best_path} (ppl {ppl:.2f})")

# keep the model hyperparameters fixed; only scale the training budget
final = OmegaConf.create(OmegaConf.to_container(best))
final.output_dir = "outputs/final"
final.batch_size = 64
final.seq_len = 128
final.grad_accumulation_steps = 1
final.num_warmup_steps = 1000
final.min_lr = float(best.max_lr) / 10
model = DecoderLM(50257, **final.model_config)
tokens_per_step = final.batch_size * final.seq_len * final.grad_accumulation_steps
final.num_training_steps = int(0.98e17 / (model.flops_per_token * tokens_per_step)) // 100 * 100
print(f"final FLOPs = {model.flops_per_token * tokens_per_step * final.num_training_steps:.3e}")
OmegaConf.save(final, "configs/final.yaml")
print(OmegaConf.to_yaml(final))
