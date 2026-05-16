# gradient-moment

Unofficial, vibe-coded, third-party, PyTorch implementation of the **Gradient Moment** metric from
*Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD*
([arXiv:2603.20155](https://arxiv.org/abs/2603.20155), Section 5).
Third-party, not an official release.

For models without a tractable likelihood, generative PPL is gameable by
low-entropy sampling. GM uses the *gradient* of a reference LM's log-likelihood
instead: ≈0 on the training distribution, non-zero on mismatched samples.

## What it computes

For reference loss L(θ, x) = −log p_θ(x):

```
centered             = || E_samples[∇L] − E_data[∇L] ||²    ← Gradient Moment
uncentered_samples   = || E_samples[∇L] ||²
uncentered_data      = || E_data[∇L] ||²
```

Unbiased per-pair estimators (Eq. 14) averaged over independent pairs. Each
batch is split in half along axis 0 internally. Gradients via
`torch.autograd.grad`; `.grad` buffers are untouched.

## Empirical behavior

Reference: gpt2-large.

### `scripts/sources_bench.py` — degenerate-source sanity (n=64)

OWT held-out blocks (seq_len=1024) as `data`.

| source | centered_gm | ppl_samples | tok_entropy_samples |
|---|---|---|---|
| `real` (same-distribution) | **−0.001 ± 0.06** | 14.6 ± 0.7 | 5.47 ± 0.02 |
| `repeated_real` (1 row tiled) | **+7.00 ± 0.09** | 14.0 | 5.46 |
| `shuffle` (within-row permutation) | **+8.63 ± 0.40** | 2575 ± 101 | 5.47 ± 0.02 |
| `data` (reference for both) | — | 14.5 ± 0.6 | — |

`repeated_real` is the headline case: PPL matches data (14.0 vs 14.5), GM
catches the collapse (+7.0).

### Top-p sweep on a small AR LM (n=512)

6L/8H/512d GPT-2-style on OWT, top-p nucleus, scored vs held-out OWT.

| top_p | centered_gm | ppl_samples | ppl_data | tok_entropy_samples |
|---|---|---|---|---|
| 1.00 | **0.674 ± 0.011** | 96.0 ± 1.0 | 14.3 | 5.62 |
| 0.95 | 0.440 ± 0.011 | 40.3 ± 0.6 | 14.5 | 5.30 |
| 0.90 | **0.365 ± 0.014** | 23.5 ± 0.4 | 14.4 | 5.06 |
| 0.85 | 0.382 ± 0.016 | 14.9 ± 0.3 | 14.6 | 4.81 |
| 0.80 | **0.415 ± 0.017** | 9.5 ± 0.2 | 14.3 | 4.53 |

`centered_gm` is U-shaped with a min near `top_p=0.90`. `ppl_samples` keeps
dropping monotonically below `ppl_data` (the gameability finding); GM reverses
direction, flagging mode-collapse.

### Failure mode: reference == generator (n=64)

When the model under evaluation IS the reference, GM degenerates by
construction: `E_{x~p_θ}[∇_θ log p_θ(x)] = 0` (score-function identity), so any
distribution close to `p_θ` produces a small μ_g and hence small GM. The
following sweep used **gpt2-large** sampling at temperature T as the
"samples" side, with **gpt2-large** also as the reference.

| T | centered_gm | ppl_samples | tok_entropy_samples |
|---|---|---|---|
| 0.30 | **0.472 ± 0.071** | **1.15 ± 0.01** | **3.02 ± 0.08** |
| 0.50 | 0.540 ± 0.079 | 1.32 ± 0.04 | 3.29 ± 0.10 |
| 0.70 | 0.668 ± 0.072 | 2.06 ± 0.07 | 4.06 ± 0.10 |
| 1.00 | 0.596 ± 0.072 | 7.52 ± 0.34 | 5.15 ± 0.04 |

As T drops (sharper, more mode-collapsed), `ppl_samples` collapses to 1.15
(far below `ppl_data` ≈ 14.5 — the canonical PPL gameability), `tok_entropy`
drops to 3.0 (vs data ≈ 5.45), and `centered_gm` **decreases** rather than
rises. The mode-collapsed regime literally produces the smallest GM in this
sweep.

Takeaway: GM can still be fooled — just harder than PPL. The attacker has
to concentrate on critical points of `p_θ` (where the score vanishes), not
merely on high-probability tokens.

## Install

```bash
pip install git+https://github.com/ehoogeboom/gradient-moment.git                 # core
pip install "gradient-moment[huggingface] @ git+https://github.com/ehoogeboom/gradient-moment.git"  # + gpt2-large helper
```

CUDA required for real reference LMs; HF loaders raise on CPU unless
`allow_cpu=True` (test stubs only).

## API

```python
from gradient_moment import evaluate, make_generative_llm_metrics, make_sample_entropy_metric

def forward_fn(model, batch):
    # return (mean_nll, mean_predictive_entropy); NLL differentiable w.r.t. model.parameters()
    ...

def batches():
    for ... in ...:
        yield batch_samples, batch_data  # even leading axis; split in half internally

results = evaluate(
    [make_generative_llm_metrics(forward_fn, model), make_sample_entropy_metric()],
    batches(),
)
```

Batches may be tensors, dicts, or lists/tuples. For the bare per-pair
estimator: `gradient_moment_estimate(loss_fn, model, batch_samples, batch_data)`.

`make_generative_llm_metrics` bundles `centered_gm`, `uncentered_gm_{samples,data}`,
`generative_ce_{samples,data}`, `generative_ppl_{samples,data}`, and
`generative_entropy_{samples,data}` in one forward+backward per sub-batch.

## HuggingFace helper (gpt2-large)

Pinned to gpt2-large for cross-run comparability. For other references use
`causal_lm_forward(model)` with your own model.

```python
from gradient_moment.huggingface import gpt2_large_evaluate, gpt2_large_evaluate_texts

results = gpt2_large_evaluate(batches())               # token-id tensors
results = gpt2_large_evaluate_texts(text_batches(), max_length=1024)  # raw strings
```

For caching across runs, build metrics yourself via `make_gpt2_large_text_metrics(...)`.

## Sign

Per-pair GM estimates are inner products and can be negative; their means
converge to non-negative squared norms.

## Based on

```bibtex
@article{hoogeboom2026dmmd,
  title  = {Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD},
  author = {Hoogeboom, Emiel and Ruhe, David and Heek, Jonathan and Mensink, Thomas and Salimans, Tim},
  journal= {arXiv preprint arXiv:2603.20155},
  year   = {2026}
}
```
