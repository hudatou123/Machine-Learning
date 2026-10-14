# Assignment 2 — Written Answers (draft)

## Q1.1

**A. Test examples** (tokenizer: `google-bert/bert-base-cased`, WordPiece)

- *Not injective:* BERT's pre-tokenizer splits on whitespace and discards it, so `"Hello world"` and `"Hello   \t world"` are different strings that map to the identical token sequence. Any amount or kind of whitespace between words is lost.
- *Not invertible:* Characters outside BERT's vocabulary are mapped to `[UNK]`, so `"I love NLP 🤖"` decodes to `"I love NLP [UNK]"`. The original emoji can never be recovered.
- *Not preserving concatenation:* `"hello"` and `"world"` tokenize to `[hello]` and `[world]`, but `"helloworld"` tokenizes to `[hello, ##world]`. WordPiece uses a separate continuation token `##world` inside a word, so tokenize(a+b) ≠ tokenize(a)+tokenize(b).

**B. Why concatenation-preserving tokenizers must be trivial**

Any string s = c₁c₂…cₙ is the concatenation of its single characters, so if a tokenizer preserves concatenation then tokenize(s) = tokenize(c₁)+…+tokenize(cₙ). The output is therefore fully determined by how each single character is tokenized, and a multi-character token (e.g. "th") can never be produced. The vocabulary effectively reduces to the alphabet, which is exactly the trivial tokenizer. Any useful merge of characters into a longer token necessarily breaks concatenation at that merge boundary.

## Q1.3

**A.** The longest learned token is `"\n\nReferences"` (12 bytes, with the fragment `"ferences"` also in the vocabulary), followed by common English words such as `" which"`, `" with"`, `" from"`, `" that"`. This suggests the training data is Wikipedia-style articles, where nearly every page ends with a "References" section.

**B.** BPE merges are learned directly from frequent strings in the training corpus, so the vocabulary and the merge order reveal what text appeared in it and how often. If private strings such as names, email addresses, phone numbers or API keys occur often enough, they can become tokens themselves and be read off the published tokenizer.

## Q1.4

**A.** Yes. The English text uses 119 tokens while its Thai translation uses 636 tokens, about 5.3× more. The tokenizer was trained only on English, so it learned no Thai merges, and each Thai character (3 bytes in UTF-8) falls back to roughly byte-level tokens.

**B.** Speakers of low-resource languages pay more per sentence under per-token pricing, fit less content into the model's fixed context window, and get slower generation. Their text is also split into less meaningful units, which tends to lower model quality for those languages.

## Q2.1

Weight tying uses the same matrix for the input token embedding and for the output projection that maps the final hidden state to vocabulary logits. This removes a whole v×d parameter matrix, which is a large fraction of a small language model's parameters, and acts as a regularizer. Press and Wolf show that the output embedding is itself a good word embedding, and that tying the two improves perplexity while making the model much smaller.

## Q2.2

```python
logits = torch.matmul(h, E.T)   # (b, s, d) @ (d, v) -> (b, s, v)
```
(equivalently `torch.einsum("bsd,vd->bsv", h, E)`)

## Q3.2

**A.**
- Warmup, t ∈ [0, a): lr(t) = lr_max · t / a
- Annealing, t ∈ [a, b): lr(t) = lr_min + ½ (lr_max − lr_min) · (1 + cos(π (t − a) / (b − a)))
- (For t ≥ b: lr(t) = lr_min.)

**B.** Warmup avoids unstable, very large updates at the start, when the weights are random and Adam's moment estimates are still unreliable. The large learning rate early on then makes fast progress, and the smooth decay to a small learning rate at the end lets the model settle into a lower-loss minimum, which usually gives a better final loss than any single constant learning rate.

## Q3.3

**A.** Validation loss of GPT-tiny: **7.062** (validation perplexity ≈ 1167).

**B.** Training loss saturates after roughly **1,500 steps**. It drops from 10.83 to about 7.35 by step 1,000, then decreases by only about 0.2 more over the remaining 1,000 steps (7.13 at step 2,000).

**C.** *(insert the W&B screenshot of `train-loss` for run `offline-run-20261013_145043-1sykqx63`)*

## Q3.4

| Config | n_embd | n_head | n_layer | max_lr | steps | train FLOPs | val loss | val PPL |
|---|---|---|---|---|---|---|---|---|
| A | 256 | 4 | 4 | 5e-4   | 2400 | 9.63e14 | 5.763 | 318.34 |
| B | 256 | 4 | 4 | 1.5e-3 | 2400 | 9.63e14 | 5.797 | 329.36 |
| C | 128 | 4 | 4 | 1.5e-3 | 5500 | 9.97e14 | 5.440 | **230.50** |
| D | 256 | 4 | 8 | 1.5e-3 | 2000 | 9.70e14 | 5.900 | 365.14 |
| E | 320 | 5 | 6 | 1.5e-3 | 1600 | 9.44e14 | 5.965 | 389.48 |

All runs used batch_size 32, seq_len 128, n_positions 128, 100 warmup steps, min_lr 1e-4 and no gradient accumulation.

**A. Procedure.** I kept batch_size = 32 and seq_len = n_positions = 128 fixed and varied model width (n_embd 128–320), depth (n_layer 4–8) and peak learning rate (5e-4 vs 1.5e-3). For every configuration I chose num_training_steps so that the total training compute, computed with the FLOP counter in `train.py`, was just under the 1e15 budget (9.4e14–1e15). Smaller models therefore trained for more steps. Each model was trained with cosine annealing (100 warmup steps, min_lr = 1e-4) and evaluated by perplexity on the held-out validation set, and the configuration with the lowest validation perplexity was selected.

**B. Results.** The smallest model, C (n_embd 128, 4 layers, lr 1.5e-3, 5,500 steps), was clearly best with a validation PPL of 230.5, compared with 318.3 for the d256/L4 model (A). Under such a small budget, seeing more tokens matters more than model capacity. Making the model wider or deeper (D, E) reduced the number of affordable steps and gave the worst results (PPL 365 and 389). For the d256/L4 model, raising the learning rate from 5e-4 to 1.5e-3 (A vs B) slightly hurt the final PPL (318 vs 329), even though training loss dropped faster early on.

**C. Final configuration** (`configs/final.yaml`):
```yaml
output_dir: outputs/final
tokenizer_encoding: gpt2
model_config:
  n_embd: 128
  n_head: 4
  n_positions: 128
  n_layer: 4
device: auto
batch_size: 128
seq_len: 128
num_warmup_steps: 1000
num_training_steps: 135000
grad_accumulation_steps: 1
min_lr: 0.00015000000000000001
max_lr: 0.0015
```
(The 1e15 tuning version of this model is `configs/tuning/C-d128-L4-lr1.5e-3.yaml`; the final run keeps the model hyperparameters and only scales batch size / steps to ≈9.79e16 FLOPs.)

## Q3.5

Final model (config C, ≈9.79e16 training FLOPs): validation loss **4.268**, validation perplexity **71.35**.
*(Optional: the same budget with the d256/L4 model from config A gave validation loss 4.091 / PPL 59.78, which suggests that for a 100× larger budget a larger model would have been preferable.)*

## Q4.2

**A.** Prefixes: the three prefixes in `hpc/prefixes3.jsonl` (maple syrup / Charles H. Meisner / night-shift workers' genetics). Settings: max_new_tokens = 48, temperature ∈ {0.1, 0.5, 1.0}, for both my model and Pythia-1.4B. Full generations: see `outputs/q4/q4_2_summary.md` (paste into report).

| Temperature | My model PPL | Pythia-1.4B PPL |
|---|---|---|
| 0.1 | 3.45 | 5.76 |
| 0.5 | 4.62 | 9.17 |
| 1.0 | 461.45 | 19.49 |

**B.** Near zero, sampling becomes almost greedy: generations are deterministic and highly repetitive. For example, my model loops on "the first time of the year, the first time of the year, …". Near one, the model samples from its full distribution, so text is more diverse but also much less coherent, with off-topic or ungrammatical tokens.

**C.** Pythia-1.4B produces fluent, on-topic continuations at every temperature, while my model only reproduces generic high-frequency phrases at low temperature and becomes incoherent at T = 1.0 (PPL 461 vs 19.5). At low temperature my model's PPL is actually lower, but only because it is confidently stuck in repetitive loops, so low PPL here does not mean higher quality. The gap comes from scale: Pythia has about 200× more parameters (1.4B vs 7M), was trained on about 300B tokens versus about 2B, and uses a 2,048-token context versus 128.

## Q4.3

**A.** Pythia-1.4B gives the Northeastern University Wikipedia paragraphs a perplexity of **7.08**. Removing every "the" (8 occurrences) raises it to **9.00**. "the" is a very predictable, low-loss token, and deleting it makes the text ungrammatical, so the model is repeatedly surprised where it expects an article before a noun.

**B.** Original Jabberwocky: PPL **1.58**. My mimic version: PPL **16.99**. The original poem is famous and appears many times in Pythia's training data (the Pile), so the model has essentially memorized it, including the nonsense words. My invented words never appeared in training, so the model cannot predict them even though the poem's structure is the same.

## Q5

- **5.1** *(fill in: e.g. Claude Code, Anthropic)*
- **5.2** *(fill in: how you used the chatbot + a sample prompt)*
- **5.3** *(fill in)*
