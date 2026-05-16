"""Source comparison: pure-degenerate vs. real held-out OWT, scored by GM.

For each "degenerate" source, evaluates the GM/CE/PPL/entropy of pure samples
from that source against held-out real OWT blocks. Writes one JSONL row per
source.

Supported sources:

- ``real``           : second disjoint sample of real held-out OWT blocks.
  The same-distribution sanity baseline — centered_gm should be ~0.
- ``repeated_real``  : tile ``--repeated-real-k`` real rows to fill the
  batch. Per-token PPL ~ real, per-row token entropy ~ real, but the
  sample distribution is mode-collapsed.
- ``shuffle``        : within-row token permutation of real rows. Local
  structure destroyed; PPL high.

Requires CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
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
    p.add_argument("--fallback-dataset", default="wikitext")
    p.add_argument("--fallback-config", default="wikitext-103-raw-v1")
    p.add_argument("--batch-size", type=int, default=2,
                   help="per-side sub-batch size; each batch feeds 2*B rows per side")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--num-batches", type=int, default=5)
    p.add_argument("--sources", nargs="+",
                   default=["real", "repeated_real", "shuffle"],
                   choices=["real", "repeated_real", "shuffle"])
    p.add_argument("--repeated-real-k", type=int, default=1,
                   help="number of unique real rows to tile for repeated_real")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--output", default="results/sources_bench.jsonl")
    return p.parse_args()


def stream_token_blocks(
    dataset_iter, tokenizer, seq_len: int, text_key: str
) -> Iterator[np.ndarray]:
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


def load_streaming_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    try:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
        return ds, "text", args.dataset
    except Exception as exc:
        print(f"[data] Primary dataset failed ({exc!r}); falling back to "
              f"'{args.fallback_dataset}/{args.fallback_config}'")
        ds = load_dataset(args.fallback_dataset, args.fallback_config, split=args.split, streaming=True)
        return ds, "text", args.fallback_dataset


def repeated_real_pool(real_pool, k: int, n_rows: int, rng) -> list[np.ndarray]:
    k = max(1, min(k, len(real_pool)))
    idx = rng.choice(len(real_pool), size=k, replace=False)
    unique = [np.asarray(real_pool[int(i)]).copy() for i in idx]
    out = [unique[i % k].copy() for i in range(n_rows)]
    rng.shuffle(out)
    return out


def shuffle_pool(real_pool, n_rows: int, rng) -> list[np.ndarray]:
    out = []
    for i in range(n_rows):
        row = np.asarray(real_pool[i % len(real_pool)])
        perm = rng.permutation(row.shape[0])
        out.append(row[perm].astype(np.int64))
    return out


def build_batches(
    sample_pool, real_q_pool, num_batches: int, batch_size: int, device: torch.device,
):
    """Pair pure-source ``sample_pool`` rows with disjoint real-q rows."""
    rows_per_batch = 2 * batch_size
    batches = []
    si = qi = 0
    for _ in range(num_batches):
        s_rows = [sample_pool[(si + j) % len(sample_pool)] for j in range(rows_per_batch)]
        q_rows = [real_q_pool[(qi + j) % len(real_q_pool)] for j in range(rows_per_batch)]
        si += rows_per_batch
        qi += rows_per_batch
        batches.append((
            torch.from_numpy(np.stack(s_rows, axis=0)).to(device),
            torch.from_numpy(np.stack(q_rows, axis=0)).to(device),
        ))
    return batches


def run_one_cell(metrics, batches):
    states = [m.init() for m in metrics]
    for bs, bd in batches:
        for i, m in enumerate(metrics):
            states[i] = m.reduce_fn(states[i], m.map_fn(bs, bd))
    return {m.name: m.finalize(s) for m, s in zip(metrics, states)}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows_per_batch = 2 * args.batch_size
    n_pool_rows = args.num_batches * rows_per_batch

    if not torch.cuda.is_available():
        raise RuntimeError("This bench requires CUDA (backprops through gpt2-large per pair).")
    device = torch.device("cuda")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"[model] Loading {GPT2_LARGE_MODEL_ID} (PyTorch, {args.dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(GPT2_LARGE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(GPT2_LARGE_MODEL_ID, torch_dtype=dtype).to(device).eval()
    print(f"[model] Loaded in {time.time()-t0:.1f}s")

    forward_fn = causal_lm_forward(model)

    # Disjoint pools: real_q (data side) + real_src (used to build shuffle / repeated_real / real-baseline pools).
    ds, text_key, dataset_id = load_streaming_dataset(args)
    block_iter = stream_token_blocks(iter(ds), tokenizer, args.seq_len, text_key)
    total_real = 2 * n_pool_rows
    print(f"[data] Pre-fetching {total_real} real blocks of {args.seq_len} tokens "
          f"(~{total_real * args.seq_len:,} tokens)...")
    t0 = time.time()
    real_blocks = [next(block_iter) for _ in range(total_real)]
    rng.shuffle(real_blocks)
    print(f"[data] Tokenized in {time.time()-t0:.1f}s")
    real_q = real_blocks[:n_pool_rows]
    real_src = real_blocks[n_pool_rows:]

    metric_factories = [
        lambda: make_generative_llm_metrics(forward_fn, model),
        lambda: make_sample_entropy_metric(),
    ]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fout = open(args.output, "w")

    def emit(source, results, elapsed):
        gllm = results["generative_llm_metrics"]
        ent = results["sample_entropy"]
        row = {
            "source": source,
            "rows_per_side": rows_per_batch,
            "num_batches": args.num_batches,
            "seq_len": args.seq_len,
            "dataset": dataset_id,
            "centered_gm": gllm["centered_gm"],
            "uncentered_gm_samples": gllm["uncentered_gm_samples"],
            "uncentered_gm_data": gllm["uncentered_gm_data"],
            "generative_ce_samples": gllm["generative_ce_samples"],
            "generative_ce_data": gllm["generative_ce_data"],
            "generative_ppl_samples": gllm["generative_ppl_samples"],
            "generative_ppl_data": gllm["generative_ppl_data"],
            "generative_entropy_samples": gllm["generative_entropy_samples"],
            "generative_entropy_data": gllm["generative_entropy_data"],
            "sample_entropy_samples": ent["samples"],
            "sample_entropy_data": ent["data"],
            "elapsed_s": elapsed,
        }
        fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"  [{source}] "
              f"GM={gllm['centered_gm']['mean']:+.4e}  "
              f"PPL_s={gllm['generative_ppl_samples']['mean']:.2f}  "
              f"H_tok_s={ent['samples']['mean']:.3f}  "
              f"({elapsed:.1f}s)")

    def run_source(source, pool):
        batches = build_batches(pool, real_q, args.num_batches, args.batch_size, device)
        metrics = [f() for f in metric_factories]
        t0 = time.time()
        results = run_one_cell(metrics, batches)
        emit(source, results, time.time() - t0)

    for source in args.sources:
        if source == "real":
            pool = [np.asarray(b) for b in real_src[:n_pool_rows]]
            run_source(source, pool)
        elif source == "repeated_real":
            pool = repeated_real_pool(real_src, args.repeated_real_k, n_pool_rows, rng)
            run_source(source, pool)
        elif source == "shuffle":
            pool = shuffle_pool(real_src, n_pool_rows, rng)
            run_source(source, pool)

    fout.close()
    print(f"[done] Wrote {args.output}")


if __name__ == "__main__":
    main()
