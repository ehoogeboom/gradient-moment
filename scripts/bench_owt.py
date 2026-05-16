"""Same-distribution sanity bench on OWT with a GPT-2 reference.

Treats two disjoint halves of OWT as ``samples`` and ``data``. Centered GM
should be ~0; ``samples``/``data`` entropies/CE/PPL should match.
``--batch-size`` is the per-side sub-batch size (each evaluation batch feeds
``2*batch_size`` rows per side; GM splits in half internally).

Requires CUDA: the metric backprops through gpt2-large per pair, which is
impractical on CPU.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterator

import numpy as np
import torch

from gradient_moment import (
    make_generative_llm_metrics,
    make_sample_entropy_metric,
)
from gradient_moment.huggingface import GPT2_LARGE_MODEL_ID, causal_lm_forward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Skylion007/openwebtext", help="HF dataset id")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument(
        "--fallback-dataset",
        default="wikitext",
        help="Used if the primary dataset is gated/missing",
    )
    p.add_argument("--fallback-config", default="wikitext-103-raw-v1")
    return p.parse_args()


def stream_token_blocks(
    dataset_iter,
    tokenizer,
    seq_len: int,
    text_key: str,
) -> Iterator[np.ndarray]:
    """Tokenize streamed records, concatenate, and yield fixed-length blocks."""
    buf: list[int] = []
    for record in dataset_iter:
        text = record.get(text_key) or ""
        if not text:
            continue
        ids = tokenizer.encode(text)
        buf.extend(ids)
        while len(buf) >= seq_len:
            block, buf = buf[:seq_len], buf[seq_len:]
            yield np.asarray(block, dtype=np.int64)


def collect_batches(block_iter, batch_size: int, num_batches: int, device: torch.device) -> list[torch.Tensor]:
    """Pull ``num_batches`` batches of shape (B, T) from a block iterator."""
    out = []
    for _ in range(num_batches):
        rows = [next(block_iter) for _ in range(batch_size)]
        out.append(torch.from_numpy(np.stack(rows, axis=0)).to(device))
    return out


def load_dataset_streaming(args: argparse.Namespace):
    """Try the primary dataset; fall back to ``args.fallback_*`` if it errors."""
    from datasets import load_dataset

    try:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
        text_key = "text"
        print(f"[data] Streaming '{args.dataset}' (split={args.split})")
    except Exception as exc:
        print(f"[data] Primary dataset failed ({exc!r}); falling back to "
              f"'{args.fallback_dataset}/{args.fallback_config}'")
        ds = load_dataset(args.fallback_dataset, args.fallback_config, split=args.split, streaming=True)
        text_key = "text"
    return ds, text_key


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This bench requires CUDA (backprops through gpt2-large per pair).")
    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"[model] Loading {GPT2_LARGE_MODEL_ID} (PyTorch, {args.dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(GPT2_LARGE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(GPT2_LARGE_MODEL_ID, torch_dtype=dtype).to(device).eval()
    print(f"[model] Loaded in {time.time() - t0:.1f}s")

    forward_fn = causal_lm_forward(model)

    # Disjoint blocks from one stream → independent sub-batches.
    ds, text_key = load_dataset_streaming(args)
    blocks = stream_token_blocks(iter(ds), tokenizer, args.seq_len, text_key)

    print(f"[bench] B={args.batch_size}  T={args.seq_len}  num_batches={args.num_batches}")

    total_batches = 2 * args.num_batches  # one samples + one data per step
    rows_per_batch = 2 * args.batch_size
    print(f"[bench] Pre-fetching {total_batches} batches of {rows_per_batch} rows "
          f"(~{total_batches * rows_per_batch * args.seq_len:,} tokens)...")
    t0 = time.time()
    all_batches = collect_batches(blocks, rows_per_batch, total_batches, device)
    rng.shuffle(all_batches)
    print(f"[bench] Tokenized in {time.time() - t0:.1f}s")

    metrics = [
        make_generative_llm_metrics(forward_fn, model),
        make_sample_entropy_metric(),
    ]
    states = [m.init() for m in metrics]
    step_times: list[float] = []
    for i in range(args.num_batches):
        batch_samples, batch_data = all_batches[2 * i: 2 * i + 2]
        t0 = time.time()
        mapped = [m.map_fn(batch_samples, batch_data) for m in metrics]
        states = [m.reduce_fn(s, x) for s, m, x in zip(states, metrics, mapped)]
        elapsed = time.time() - t0
        step_times.append(elapsed)
        gllm, ent_rows = mapped
        h_samples = float(np.mean(np.asarray(ent_rows[0])))
        h_data = float(np.mean(np.asarray(ent_rows[1])))
        print(f"[batch {i+1}/{args.num_batches}] "
              f"GM={float(gllm.centered):+.4e}  "
              f"CE_samples={float(gllm.nll_samples):.3f}  CE_data={float(gllm.nll_data):.3f}  "
              f"H_samples(ref)={float(gllm.ent_samples):.3f}  "
              f"H_data(ref)={float(gllm.ent_data):.3f}  "
              f"H_samples(tok)={h_samples:.3f}  H_data(tok)={h_data:.3f}  "
              f"time={elapsed:.1f}s")

    results = {m.name: m.finalize(s) for m, s in zip(metrics, states)}
    gllm = results["generative_llm_metrics"]
    ent = results["sample_entropy"]

    def fmt_sci(d):
        return f"mean={d['mean']:+.4e}  stderr={d['stderr']:.4e}"

    def fmt_nat(d):
        return f"mean={d['mean']:.4f}  stderr={d['stderr']:.4e}  (n={d['n']})"

    print()
    print("=" * 78)
    print("generative_llm_metrics:")
    print(f"  centered_gm                  ||E_samples - E_data||^2 : {fmt_sci(gllm['centered_gm'])}")
    print(f"  uncentered_gm_samples        ||E_samples||^2          : {fmt_sci(gllm['uncentered_gm_samples'])}")
    print(f"  uncentered_gm_data           ||E_data||^2             : {fmt_sci(gllm['uncentered_gm_data'])}")
    print(f"  generative_ce_samples        (nats/token)             : {fmt_nat(gllm['generative_ce_samples'])}")
    print(f"  generative_ce_data           (nats/token)             : {fmt_nat(gllm['generative_ce_data'])}")
    print(f"  generative_ppl_samples                                : {fmt_nat(gllm['generative_ppl_samples'])}")
    print(f"  generative_ppl_data                                   : {fmt_nat(gllm['generative_ppl_data'])}")
    print(f"  generative_entropy_samples   (ref predictive H, nats) : {fmt_nat(gllm['generative_entropy_samples'])}")
    print(f"  generative_entropy_data      (ref predictive H, nats) : {fmt_nat(gllm['generative_entropy_data'])}")
    print(f"  n_batches                    = {gllm['centered_gm']['n']}")
    print(f"sample_entropy (per-row token diversity, nats; max log T = {float(np.log(args.seq_len)):.3f}):")
    print(f"  samples  : {fmt_nat(ent['samples'])}")
    print(f"  data     : {fmt_nat(ent['data'])}")
    if step_times:
        print(f"timing: per-batch mean {np.mean(step_times):.1f}s")
    print("=" * 78)
    print()
    print("Same-distribution sanity: centered_gm should be statistically "
          "indistinguishable from 0. All samples/data quantities (uncentered_gm, "
          "CE, PPL, ref entropy, sample entropy) should be roughly equal since "
          "samples and data are drawn from the same distribution.")


if __name__ == "__main__":
    main()
