# Does LR re-warming alone close the gap with rehearsal?

A one-day continual-pretraining experiment, in the spirit of Ibrahim et al.
("Simple and Scalable Strategies to Continually Pre-train Large Language
Models", 2024). The claim under test: at small scale, **a re-warmed cosine
schedule does most of the work attributed to rehearsal**, and the marginal
benefit of mixing in 5% old-domain data on top of a good schedule is small.

## Setup

| | |
|---|---|
| Base model | `gpt2` from HuggingFace, 124M params, ctx 1024 |
| Old domain (forgetting eval) | FineWeb val shard (`fineweb_val_000000.bin`, GPT-2 BPE) |
| New domain (CPT target) | `codeparrot/codeparrot-clean` (Python), 50M training tokens, 5M held-out val |
| Rehearsal source | `fineweb_train_000001.bin` (GPT-2 BPE) |
| Hardware | 8 × H200, bf16, plain DDP |
| Optimizer | AdamW(β=(0.9, 0.95), wd=0.01 on 2-D params), grad clip 1.0 |
| Steps | 1000 |
| Tokens/step | 8 (per-device bs) × 1024 (ctx) × 8 (GPUs) = 65,536 |
| Total CPT tokens | ~65M |
| Eval cadence | every 50 steps; 32 batches × 8 windows × 1024 tokens ≈ 262K tokens per eval, deterministic |
| Seeds | 3 per condition |

The 2 × 2 condition matrix:

| | replay = 0 | replay = 5% |
|---|---|---|
| **constant LR = 1e-5** | (a) baseline-low | (c) constant + replay |
| **re-warmed cosine, peak 3e-4 → 1e-5, 50-step warmup** | (b) re-warm only | (d) re-warm + replay |

Replay is implemented per-sequence: each item in the batch is independently
drawn from the FineWeb train shard with probability `replay_frac`, otherwise
from the Python shard.

The headline metrics are:

- **Forgetting** Δweb = web_nll(step 1000) − web_nll(step 0)  (lower = better, 0 = no forgetting)
- **Adaptation** Δcode = code_nll(step 1000) − code_nll(step 0)  (more negative = better)

A useful continual-learning method should sit toward the bottom-left in
(Δweb, Δcode) space.

## Results

![main figure](figure_main.png)

### Numerical summary (mean ± std over 3 seeds)

```
[summary table fills in here from cpt/summary.txt]
```

### Findings

1. **Re-warmed cosine forgets far more than constant low LR.** Going from
   constant 1e-5 to peak 3e-4 cosine roughly XXXs the forgetting rate; the
   model gives up a meaningful chunk of FineWeb performance to chase code.

2. **Replay has different leverage in the two regimes.** At constant low LR,
   5% replay has [marginal | noticeable] effect on Δweb because forgetting
   was [already small | already substantial]. At cosine LR, replay [does | does
   not] close the forgetting gap to the constant-LR baseline.

3. **Adaptation is dominated by the LR schedule, not the data mixture.**
   Conditions (a) and (c) (low LR) reach roughly the same Δcode regardless of
   replay, and conditions (b) and (d) (cosine LR) reach a similar Δcode.
   5% replay costs ~5% effective code tokens and the adaptation curves bear
   that out.

4. **Re-warm + replay is on or near the Pareto frontier** between forgetting
   and adaptation. Whether it strictly dominates the other three depends on
   how you weight the two axes.

(The above text is a placeholder reflecting expected directional findings; the
final REPORT.md commits the actual results.)

## What this does and does not show

- This is **124M scale on 65M CPT tokens with one new domain**. Ibrahim et al.
  run 405M and 10B models on much larger token budgets. The relative
  ordering of conditions is likely informative, but the exact numbers aren't.
- We tokenize both domains with the GPT-2 BPE — fine for the old domain (GPT-2
  was trained with this tokenizer) and acceptable for code, though Python
  tokenizes inefficiently in BPE. A native code tokenizer would shift the
  *absolute* code NLL but should leave the *relative* ordering intact.
- "Forgetting" here is val NLL on a single FineWeb shard. We do not measure
  downstream task accuracy — recent work argues some of what looks like
  forgetting in val NLL is actually format collapse on instruction-following
  models. Since we start from a base (un-instruction-tuned) checkpoint, that
  particular concern doesn't apply, but a richer eval is left as future work.
- n=3 per condition is enough to resolve the qualitative ordering when the
  effect sizes are large (>0.1 nats) but is **not** enough to draw conclusions
  about small effects (≤0.02 nats).
- Peak LR 3e-4 was chosen as a middle-of-the-road CPT setting and not
  separately tuned; the picture could shift if peak LR is co-varied with
  cosine length.

## How to reproduce

```bash
# 1) tokenize ~55M code tokens into cpt/data/{code_train,code_val}.bin (~2 min)
.venv/bin/python cpt/prepare_code_shard.py

# 2) run all 12 conditions (4 conditions x 3 seeds, ~40 min on 8xH200)
bash cpt/run_sweep.sh

# 3) plot + summary table
.venv/bin/python cpt/plot_results.py
```

Per-run results land in `cpt/results/*.json`, training logs in `cpt/logs/`.
