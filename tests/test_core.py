"""Tests for the gradient_moment.core primitives.

Synthetic reference: L(w, x) = 0.5 * mean_b ||x_b - w||²  ⇒  grad_w L = w - mean_b(x_b).
With w = 0: E[grad] = -mean(distribution), giving closed-form ground truth for
centered/uncentered moments under different (g_mean, q_mean) settings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradient_moment import (
    GMEstimate,
    diff_grad,
    gradient_moment_estimate,
    tree_inner_product,
)


# --- Synthetic model wrapping a single parameter vector ---------------------

class QuadraticModel(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(dim))


def quadratic_loss(model: QuadraticModel, batch: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((batch - model.w) ** 2).sum(dim=-1).mean()


# --- Building-block correctness --------------------------------------------

def test_tree_inner_product_matches_flat_dot():
    a = [torch.arange(6.0).reshape(2, 3), torch.tensor([1.0, 2.0])]
    b = [torch.full((2, 3), 2.0), torch.tensor([3.0, 4.0])]
    expected = float((a[0] * b[0]).sum() + (a[1] * b[1]).sum())
    assert float(tree_inner_product(a, b)) == pytest.approx(expected)


def test_tree_inner_product_empty():
    assert float(tree_inner_product([], [])) == 0.0


def test_diff_grad_equals_separate_gradients():
    model = QuadraticModel(3)
    bg = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bq = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    grad_g = torch.autograd.grad(quadratic_loss(model, bg), [model.w])[0]
    grad_q = torch.autograd.grad(quadratic_loss(model, bq), [model.w])[0]
    diff = diff_grad(quadratic_loss, model, bg, bq)
    assert len(diff) == 1
    np.testing.assert_allclose(diff[0].numpy(), (grad_g - grad_q).numpy(), atol=1e-6)


def test_multi_parameter_model_supported():
    class TwoParam(torch.nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(dim))
            self.b = torch.nn.Parameter(torch.zeros(()))

    def loss(m, x):
        return 0.5 * ((x - m.w) ** 2).sum(dim=-1).mean() + m.b

    model = TwoParam(3)
    bg = torch.ones((4, 3))
    bq = torch.zeros((4, 3))
    estimate = gradient_moment_estimate(loss, model, bg, bq)
    # Both halves of bg are identical and so are both halves of bq, so each
    # inner product is a squared norm and therefore non-negative.
    assert isinstance(estimate, GMEstimate)
    assert float(estimate.centered) >= 0.0
    assert float(estimate.uncentered_samples) >= 0.0
    assert float(estimate.uncentered_data) >= 0.0


def test_split_halves_raises_on_odd_batch():
    model = QuadraticModel(2)
    bg = torch.zeros((3, 2))
    bq = torch.zeros((4, 2))
    with pytest.raises(ValueError):
        gradient_moment_estimate(quadratic_loss, model, bg, bq)


def test_dict_batch_supported():
    """Batches as dicts (e.g. {input_ids, attention_mask}) get split per leaf."""
    model = QuadraticModel(2)

    def dict_loss(m, batch):
        return quadratic_loss(m, batch["x"])

    bg = {"x": torch.ones((4, 2))}
    bq = {"x": torch.zeros((4, 2))}
    est = gradient_moment_estimate(dict_loss, model, bg, bq)
    assert isinstance(est, GMEstimate)
    assert float(est.centered) >= 0.0
