import pytest
import torch

from research.fedgcdr_akf.model import FusionHead, make_gat
from research.fedgcdr_akf.upstream import GAT


def inputs():
    torch.manual_seed(12)
    return {
        "local_user": torch.randn(4), "graph_user": torch.randn(4),
        "neighbors": torch.randn(3, 4), "item_vectors": torch.randn(5, 4),
        "knowledge": torch.randn(2, 4, requires_grad=True),
        "available": torch.tensor([True, True]), "train_count": 3,
    }


def test_original_gat_is_reused():
    assert type(make_gat(4, "cpu")) is GAT


@pytest.mark.parametrize("mode", ["fixed", "global", "user", "item"])
def test_gamma_bounds_gradients_and_frozen_source_knowledge(mode):
    values = inputs()
    head = FusionHead(4, 2, mode)
    score, gamma, regularizer = head(**values)
    assert score.shape == gamma.shape == (5,)
    assert torch.isfinite(score).all()
    assert ((gamma >= 0) & (gamma <= 1)).all()
    (score.square().mean() + regularizer).backward()
    assert values["knowledge"].grad is None
    assert head.mappers[0].L1.weight.grad.abs().sum() > 0
    if mode == "global":
        assert head.gamma_logit.grad.abs() > 0
    if mode in {"user", "item"}:
        assert head.gate[0].weight.grad.abs().sum() > 0


def test_user_gate_is_item_invariant_but_user_conditioned():
    values = inputs()
    head = FusionHead(4, 2, "user")
    _, gamma, _ = head(**values)
    assert torch.allclose(gamma, gamma[:1].expand_as(gamma))
    _, other, _ = head(**{**values, "local_user": values["local_user"] + 1})
    assert not torch.allclose(gamma, other)


def test_item_gate_changes_with_candidate():
    values = inputs()
    head = FusionHead(4, 2, "item")
    _, gamma, _ = head(**values)
    assert gamma.std() > 1e-5


@pytest.mark.parametrize("mode", ["fixed", "fixed_capacity", "global", "user", "item"])
def test_missing_knowledge_cannot_leak_mapper_biases(mode):
    values = inputs()
    values["available"] = torch.tensor([False, False])
    head = FusionHead(4, 2, mode)
    for mapper in head.mappers:
        torch.nn.init.constant_(mapper.L3.bias, 10)
    logits, gamma, regularizer = head(**values)
    expected = (values["local_user"] * values["item_vectors"]).sum(-1)
    assert torch.equal(logits, expected)
    assert torch.equal(gamma, torch.zeros(5))
    assert regularizer == 0


def test_zero_fixed_gamma_disables_transfer_score():
    values = inputs()
    head = FusionHead(4, 2, "fixed", fixed_gamma=0)
    logits, gamma, _ = head(**values)
    local = (values["local_user"] * values["item_vectors"]).sum(-1)
    assert torch.allclose(logits, head.f2(local[:, None]).squeeze(-1))
    assert gamma.count_nonzero() == 0


def test_fixed_capacity_is_close_to_user_gate_budget():
    user = FusionHead(8, 2, "user")
    fixed = FusionHead(8, 2, "fixed_capacity")
    user_params = sum(p.numel() for p in user.parameters())
    fixed_params = sum(p.numel() for p in fixed.parameters())
    assert abs(user_params - fixed_params) / user_params < 0.03
