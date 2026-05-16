"""Tests for the metrics framework and concrete metrics.

Uses small synthetic stubs on CPU. The real gpt2-large metric path is gated
to CUDA in ``gradient_moment.huggingface``.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
import pytest
import torch

from gradient_moment import (
    Metric,
    evaluate,
    make_generative_llm_metrics,
    make_sample_entropy_metric,
)


# --- Synthetic model + forward stub -----------------------------------------

class QuadraticModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(dim))


def quadratic_loss(model: QuadraticModel, batch: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((batch - model.w) ** 2).sum(dim=-1).mean()


def quadratic_forward(model: QuadraticModel, batch: torch.Tensor):
    return quadratic_loss(model, batch), torch.tensor(0.0)


def _stream_with_means(
    rng: np.random.Generator,
    g_mean: np.ndarray,
    q_mean: np.ndarray,
    batch_size: int,
    num_pairs: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Each emitted batch has leading axis ``2 * batch_size`` (GM splits in half)."""
    dim = g_mean.shape[-1]
    for _ in range(num_pairs):
        bg = g_mean + rng.standard_normal((2 * batch_size, dim)).astype(np.float32)
        bq = q_mean + rng.standard_normal((2 * batch_size, dim)).astype(np.float32)
        yield torch.from_numpy(bg), torch.from_numpy(bq)


def _close(summary, expected: float, abs_tol: float = 0.1):
    mean, se = summary["mean"], summary["stderr"]
    tol = max(5 * se, abs_tol)
    return abs(mean - expected) < tol, f"mean={mean}, expected={expected}, stderr={se}"


# --- generative_llm_metrics via the framework -------------------------------

def test_gllm_metric_zero_when_g_equals_q():
    dim = 4
    model = QuadraticModel(dim)
    rng = np.random.default_rng(0)
    stream = _stream_with_means(rng, np.zeros(dim, dtype=np.float32),
                                np.zeros(dim, dtype=np.float32),
                                batch_size=32, num_pairs=200)
    metric = make_generative_llm_metrics(quadratic_forward, model)
    results = evaluate([metric], stream)
    gm = results["generative_llm_metrics"]
    for name in ("centered_gm", "uncentered_gm_samples", "uncentered_gm_data"):
        ok, msg = _close(gm[name], 0.0)
        assert ok, f"{name}: {msg}"


def test_gllm_metric_recovers_squared_gap():
    dim = 4
    delta = 0.5
    expected = dim * delta ** 2
    model = QuadraticModel(dim)
    rng = np.random.default_rng(1)
    stream = _stream_with_means(rng, np.full((dim,), delta, dtype=np.float32),
                                np.zeros(dim, dtype=np.float32),
                                batch_size=32, num_pairs=400)
    metric = make_generative_llm_metrics(quadratic_forward, model)
    results = evaluate([metric], stream)
    gm = results["generative_llm_metrics"]
    ok, msg = _close(gm["centered_gm"], expected); assert ok, f"centered_gm: {msg}"
    ok, msg = _close(gm["uncentered_gm_samples"], expected); assert ok, f"uncentered_gm_samples: {msg}"
    ok, msg = _close(gm["uncentered_gm_data"], 0.0); assert ok, f"uncentered_gm_data: {msg}"


def test_gllm_metric_reports_ce_ppl_and_entropy():
    """CE = mean L(0,x) with x~N(0,I_d), so E[CE] = d/2; entropy stub returns 0."""
    dim = 4
    model = QuadraticModel(dim)
    rng = np.random.default_rng(3)
    stream = _stream_with_means(rng, np.zeros(dim, dtype=np.float32),
                                np.zeros(dim, dtype=np.float32),
                                batch_size=64, num_pairs=200)
    metric = make_generative_llm_metrics(quadratic_forward, model)
    results = evaluate([metric], stream)
    gm = results["generative_llm_metrics"]
    expected_ce = dim / 2.0
    expected_ppl = math.exp(expected_ce)
    for side in ("samples", "data"):
        ok, msg = _close(gm[f"generative_ce_{side}"], expected_ce)
        assert ok, f"generative_ce_{side}: {msg}"
        assert gm[f"generative_ppl_{side}"]["mean"] == pytest.approx(expected_ppl, rel=0.1)
        assert gm[f"generative_entropy_{side}"]["mean"] == pytest.approx(0.0, abs=1e-6)


# --- Sample entropy: closed-form ground truth -------------------------------

def test_sample_entropy_all_same_row_is_zero():
    metric = make_sample_entropy_metric()
    T = 8
    batch_g = torch.zeros((3, T), dtype=torch.int64)
    batch_q = torch.zeros((3, T), dtype=torch.int64)
    results = evaluate([metric], [(batch_g, batch_q)])
    h = results["sample_entropy"]
    assert h["samples"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert h["data"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_sample_entropy_all_distinct_row_is_log_T():
    metric = make_sample_entropy_metric()
    T = 7
    distinct = torch.arange(T, dtype=torch.int64)[None, :]
    same = torch.zeros((1, T), dtype=torch.int64)
    results = evaluate([metric], [(distinct, same)])
    h = results["sample_entropy"]
    assert h["samples"]["mean"] == pytest.approx(float(np.log(T)), abs=1e-5)
    assert h["data"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_sample_entropy_matches_naive_numpy():
    rng = np.random.default_rng(7)
    T = 16
    batch_np = rng.integers(0, 5, size=(4, T), dtype=np.int64)

    def numpy_entropy(row):
        _, counts = np.unique(row, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log(p)))

    expected = float(np.mean([numpy_entropy(r) for r in batch_np]))

    metric = make_sample_entropy_metric()
    batch = torch.from_numpy(batch_np)
    results = evaluate([metric], [(batch, batch)])
    h = results["sample_entropy"]
    assert h["samples"]["mean"] == pytest.approx(expected, abs=1e-5)
    assert h["data"]["mean"] == pytest.approx(expected, abs=1e-5)


def test_sample_entropy_n_counts_rows_not_batches():
    metric = make_sample_entropy_metric()
    T, B = 4, 5
    batch = torch.zeros((B, T), dtype=torch.int64)
    n_batches = 3
    results = evaluate([metric], [(batch, batch)] * n_batches)
    assert results["sample_entropy"]["samples"]["n"] == B * n_batches
    assert results["sample_entropy"]["data"]["n"] == B * n_batches


# --- Evaluator drives multiple metrics in one pass --------------------------

def test_evaluate_runs_multiple_metrics_in_one_pass():
    dim = 3
    model = QuadraticModel(dim)
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(2):
        bg = torch.from_numpy(rng.integers(0, 4, size=(4, dim)).astype(np.float32))
        bq = torch.from_numpy(rng.integers(0, 4, size=(4, dim)).astype(np.float32))
        pairs.append((bg, bq))

    metrics = [
        make_generative_llm_metrics(quadratic_forward, model),
        make_sample_entropy_metric(),
    ]
    results = evaluate(metrics, pairs)
    assert set(results.keys()) == {"generative_llm_metrics", "sample_entropy"}
    assert set(results["generative_llm_metrics"].keys()) == {
        "centered_gm",
        "uncentered_gm_samples", "uncentered_gm_data",
        "generative_ce_samples", "generative_ce_data",
        "generative_ppl_samples", "generative_ppl_data",
        "generative_entropy_samples", "generative_entropy_data",
    }
    assert set(results["sample_entropy"].keys()) == {"samples", "data"}


def test_metric_is_named_tuple_with_callable_fields():
    metric = make_sample_entropy_metric()
    assert isinstance(metric, Metric)
    assert metric.name == "sample_entropy"
    for field in ("init", "map_fn", "reduce_fn", "finalize"):
        assert callable(getattr(metric, field))


# --- CPU gating on the huggingface helper -----------------------------------

def test_huggingface_loader_refuses_cpu_without_opt_in():
    """The real gpt2-large path errors on CPU unless the caller opts in.

    We don't actually download the model; the check happens before that. Use
    ``device="cpu"`` to force the path even when CUDA is available."""
    from gradient_moment.huggingface import _resolve_device  # noqa: PLC0415

    with pytest.raises(RuntimeError, match="too heavy for CPU"):
        _resolve_device("cpu", allow_cpu=False)
    # opt-in works:
    assert _resolve_device("cpu", allow_cpu=True).type == "cpu"
