"""Gradient Moment metric (Hoogeboom et al., 2026, arXiv:2603.20155).

PyTorch implementation. All public symbols are accessible at the top level.
"""

from __future__ import annotations

from gradient_moment.core import (
    BatchPair,
    ForwardFn,
    GMEstimate,
    LossFn,
    diff_grad,
    gradient_moment_estimate,
    tree_inner_product,
)
from gradient_moment.metrics.base import Metric, evaluate, running_summary
from gradient_moment.metrics.generative_llm import make_generative_llm_metrics
from gradient_moment.metrics.sample_entropy import make_sample_entropy_metric
from gradient_moment.stats import (
    RunningStats,
    init_stats,
    stats_mean,
    stats_stderr,
    update_stats,
    update_stats_many,
)

__version__ = "0.2.0"

__all__ = [
    "BatchPair",
    "ForwardFn",
    "GMEstimate",
    "LossFn",
    "Metric",
    "RunningStats",
    "diff_grad",
    "evaluate",
    "gradient_moment_estimate",
    "init_stats",
    "make_generative_llm_metrics",
    "make_sample_entropy_metric",
    "running_summary",
    "stats_mean",
    "stats_stderr",
    "tree_inner_product",
    "update_stats",
    "update_stats_many",
]
