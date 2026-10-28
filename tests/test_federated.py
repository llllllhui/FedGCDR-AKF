import math
import random

import pytest
import torch

from research.fedgcdr_akf.federated import (
    average_shared, extract_knowledge, initialize_domain, ranking_metrics, sample_negatives, train_round,
)


def test_negative_sampler_excludes_train_positives_and_terminates():
    rng = random.Random(4)
    assert not set(sample_negatives([1, 3], 10, 100, rng)) & {1, 3}
    assert sample_negatives(range(5), 5, 20, rng) == []
    assert sample_negatives(range(4), 5, 20, rng) == [4] * 20


def test_shared_aggregation_is_weighted_not_sequential():
    module = torch.nn.Linear(1, 1, bias=False)
    average_shared(module, [(1, {"weight": torch.tensor([[2.]])}), (3, {"weight": torch.tensor([[6.]])})])
    assert module.weight.item() == 5


def test_multi_positive_recall_and_ndcg():
    recall, ndcg = ranking_metrics([7, 2, 8, 3], {2, 3}, k=3)
    assert recall == .5
    assert ndcg == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_sparse_local_updates_and_missing_source_mask():
    torch.manual_seed(3)
    state = initialize_domain(3, 5, 4, "cpu")
    before_users, before_items = state.users.clone(), state.items.clone()
    histories = [[0, 1], [], [1, 2]]
    cfg = {"client_batch_size": 2, "max_positives": 2, "max_history": 2,
           "negatives_per_positive": 0, "local_epochs": 1, "learning_rate": .01}
    result = train_round(state.gat, state.users, None, state.items, histories, cfg, random.Random(2))
    assert result["clients"] == 2 and math.isfinite(result["loss"])
    assert result["payload_bytes_estimate"] > 0
    assert not torch.equal(state.users[0], before_users[0])
    assert torch.equal(state.users[1], before_users[1])
    assert torch.equal(state.items[3:], before_items[3:])
    knowledge, available = extract_knowledge(state, histories, 2)
    assert available.tolist() == [True, False, True]
    assert not knowledge.requires_grad
    assert torch.equal(knowledge[1], torch.zeros(4))
