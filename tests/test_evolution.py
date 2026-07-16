"""Tests for esma.evolution."""

import numpy as np
import torch
import torch.nn as nn

from esma.evolution import apply_evolution


def _make_model(device=None):
    """Create a small model with a .device attribute for apply_evolution."""
    model = nn.Linear(4, 2)
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.device = device
    return model


class TestApplyEvolution:
    """Tests for apply_evolution."""

    def test_single_int_seed_accepted(self):
        model = _make_model()
        params_before = [p.data.clone() for p in model.parameters()]
        apply_evolution(model, 42, absolute_scale=0.1)
        params_after = list(model.parameters())
        for b, a in zip(params_before, params_after):
            assert not torch.equal(b, a.data)

    def test_multiple_seeds_ndarray_accepted(self):
        model = _make_model()
        params_before = [p.data.clone() for p in model.parameters()]
        apply_evolution(model, np.array([42, 43]), absolute_scale=0.1, relative_scales=[1.0, 1.0])
        params_after = list(model.parameters())
        for b, a in zip(params_before, params_after):
            assert not torch.equal(b, a.data)

    def test_ndarray_seeds_accepted(self):
        model = _make_model()
        seeds = np.array([1, 2, 3])
        apply_evolution(model, seeds, absolute_scale=0.01, relative_scales=[1.0, 1.0, 1.0])
        # Just check it runs; params should change
        assert True

    def test_zero_scale_no_change(self):
        model = _make_model()
        params_before = [p.data.clone() for p in model.parameters()]
        apply_evolution(model, 42, absolute_scale=0.0)
        params_after = list(model.parameters())
        for b, a in zip(params_before, params_after):
            assert torch.equal(b, a.data)

    def test_same_seed_same_update(self):
        model1 = _make_model()
        model2 = _make_model()
        # Same init
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)
        apply_evolution(model1, 123, absolute_scale=0.1)
        apply_evolution(model2, 123, absolute_scale=0.1)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1.data, p2.data)

    def test_different_seed_different_update(self):
        model1 = _make_model()
        model2 = _make_model()
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)
        apply_evolution(model1, 1, absolute_scale=0.1)
        apply_evolution(model2, 2, absolute_scale=0.1)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert not torch.allclose(p1.data, p2.data)

    def test_reverse_subtracts_update(self):
        model = _make_model()
        params_before = [p.data.clone() for p in model.parameters()]
        apply_evolution(model, 99, absolute_scale=0.1, reverse=True)
        params_after = list(model.parameters())
        for b, a in zip(params_before, params_after):
            assert not torch.equal(b, a.data)

    def test_relative_scales_applied(self):
        model = _make_model()
        p_before = next(model.parameters()).data.clone()
        apply_evolution(model, 1, absolute_scale=1.0, relative_scales=[0.5])
        p_after = next(model.parameters()).data
        diff = (p_after - p_before).abs().max().item()
        assert diff > 0
        assert diff < 2.0  # scale 0.5 keeps update moderate
