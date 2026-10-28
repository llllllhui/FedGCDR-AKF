"""In-process user clients with sparse item updates and shared-module FedAvg."""

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import make_gat


@dataclass
class DomainState:
    users: torch.Tensor
    items: torch.Tensor
    gat: nn.Module


def initialize_domain(num_users, num_items, dim, device):
    users = torch.empty(num_users, dim, device=device).uniform_(-0.1, 0.1)
    items = torch.empty(num_items, dim, device=device).uniform_(-0.1, 0.1)
    return DomainState(users, items, make_gat(dim, device))


def sample_negatives(positives, num_items, count, rng):
    positives = set(positives)
    if not count or len(positives) >= num_items:
        return []
    if len(positives) > num_items // 2:
        candidates = [i for i in range(num_items) if i not in positives]
        return rng.choices(candidates, k=count)
    result = []
    while len(result) < count:
        candidate = rng.randrange(num_items)
        if candidate not in positives:
            result.append(candidate)
    return result


@torch.no_grad()
def average_shared(module, updates):
    """Updates are (train weight, local state); each client used one shared snapshot."""
    total = sum(weight for weight, _ in updates)
    if not total:
        return
    destination = module.state_dict()
    for name, parameter in destination.items():
        value = torch.zeros_like(parameter)
        for weight, state in updates:
            value.add_(state[name], alpha=weight / total)
        parameter.copy_(value)


def train_round(shared, users, graph_users, items, histories, cfg, rng, knowledge=None, available=None):
    """Only touched item rows are copied; private user rows are never averaged."""
    client_ids = [u for u, history in enumerate(histories) if history]
    rng.shuffle(client_ids)
    losses, payload = [], 0
    pretraining = knowledge is None
    batch_size = cfg["client_batch_size"]
    for start in range(0, len(client_ids), batch_size):
        updates, item_updates = [], []
        for uid in client_ids[start:start + batch_size]:
            history = histories[uid]
            positives = rng.sample(history, min(len(history), cfg["max_positives"]))
            negatives = sample_negatives(history, len(items), len(positives) * cfg["negatives_per_positive"], rng)
            examples = positives + negatives
            neighbors = history[-cfg["max_history"]:]
            touched = sorted(set(examples + neighbors))
            index = torch.tensor(touched, device=items.device, dtype=torch.long)
            example_index = torch.searchsorted(index, index.new_tensor(examples))
            neighbor_index = torch.searchsorted(index, index.new_tensor(neighbors))
            local_items = nn.Parameter(items[index].detach().clone())
            local_user = nn.Parameter(users[uid].detach().clone())
            local_graph_user = None if pretraining else nn.Parameter(graph_users[uid].detach().clone())
            local_model = copy.deepcopy(shared)
            local_model.train()
            parameters = [local_user, local_items, *local_model.parameters()]
            if local_graph_user is not None:
                parameters.append(local_graph_user)
            optimizer = torch.optim.Adam(parameters, lr=cfg["learning_rate"])
            labels = items.new_tensor([1.] * len(positives) + [0.] * len(negatives))
            for _ in range(cfg["local_epochs"]):
                optimizer.zero_grad()
                if pretraining:
                    nodes = torch.cat((local_user.unsqueeze(0), local_items[neighbor_index]))
                    graph, _, _, _ = local_model(nodes)
                    logits = (graph[0] * local_items[example_index]).sum(dim=-1)
                    regularizer = 0
                else:
                    logits, _, regularizer = local_model(
                        local_user, local_graph_user, local_items[neighbor_index],
                        local_items[example_index], knowledge[uid], available[uid], len(history),
                    )
                loss = F.binary_cross_entropy_with_logits(logits, labels) + regularizer
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite local loss for client index {uid}")
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            weight = len(history)
            updates.append((weight, {k: v.detach().clone() for k, v in local_model.state_dict().items()}))
            item_updates.append((index, local_items.detach() - items[index]))
            with torch.no_grad():
                users[uid].copy_(local_user)
                if not pretraining:
                    graph_users[uid].copy_(local_graph_user)
            # Conceptual downlink + uplink float tensors and item-row IDs; no real network traffic.
            active = shared.active_parameters() if hasattr(shared, "active_parameters") else shared.parameters()
            model_bytes = sum(p.numel() * p.element_size() for p in active)
            payload += 2 * (model_bytes + local_items.numel() * local_items.element_size() + index.numel() * 8)
        average_shared(shared, updates)
        # Aggregate only the item rows touched by clients in this batch.
        touched_batch = torch.unique(torch.cat([idx for idx, _ in item_updates]))
        delta_sum = torch.zeros(len(touched_batch), items.shape[1], device=items.device)
        count = torch.zeros(len(touched_batch), device=items.device)
        for index, delta in item_updates:
            positions = torch.searchsorted(touched_batch, index)
            delta_sum.index_add_(0, positions, delta)
            count.index_add_(0, positions, torch.ones(len(index), device=items.device))
        with torch.no_grad():
            items[touched_batch] += delta_sum / count.unsqueeze(-1)
    return {"loss": sum(losses) / len(losses) if losses else None, "clients": len(losses), "payload_bytes_estimate": payload}


@torch.no_grad()
def extract_knowledge(state, histories, max_history):
    knowledge = torch.zeros_like(state.users)
    available = torch.zeros(len(histories), dtype=torch.bool, device=state.users.device)
    state.gat.eval()
    for uid, history in enumerate(histories):
        if not history:
            continue
        nodes = torch.cat((state.users[uid:uid + 1], state.items[history[-max_history:]]))
        _, intermediate, _, _ = state.gat(nodes)
        knowledge[uid] = intermediate[0]
        available[uid] = True
    return knowledge.detach(), available


def ranking_metrics(ranked_items, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Ranking metrics require at least one relevant item.")
    ranked = ranked_items[:k]
    hits = [i for i, item in enumerate(ranked) if item in relevant]
    recall = len(hits) / len(relevant)
    dcg = sum(1 / math.log2(i + 2) for i in hits)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return recall, dcg / ideal


@torch.no_grad()
def evaluate(head, users, graph_users, items, train, heldout, knowledge, available, max_history, k=10):
    head.eval()
    rows = []
    for uid, relevant in enumerate(heldout):
        if not relevant:
            continue
        context, _ = head.context(users[uid], graph_users[uid], items[train[uid][-max_history:]], knowledge[uid], available[uid])
        score_parts, gamma_parts = [], []
        for start in range(0, len(items), 4096):
            score, gamma = head.score(users[uid], items[start:start + 4096], context, len(train[uid]))
            score_parts.append(score)
            gamma_parts.append(gamma)
        scores = torch.cat(score_parts)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite ranking scores")
        scores[train[uid]] = -torch.inf
        candidate_count = len(items) - len(train[uid])
        ranked = scores.topk(min(k, candidate_count)).indices.tolist()
        recall, ndcg = ranking_metrics(ranked, relevant, k)
        eligible = torch.isfinite(scores)
        gamma_values = torch.cat(gamma_parts)[eligible]
        rows.append({"user_index": uid, "train_count": len(train[uid]), "has_source_knowledge": bool(available[uid].any()),
                     "recall@10": recall, "ndcg@10": ndcg,
                     "gamma_mean": gamma_values.mean().item(), "gamma_std_across_items": gamma_values.std(unbiased=False).item()})
    if not rows:
        raise ValueError("No users available for this evaluation split.")
    summary = {key: sum(row[key] for row in rows) / len(rows) for key in ("recall@10", "ndcg@10", "gamma_mean")}
    means = torch.tensor([row["gamma_mean"] for row in rows])
    summary["gamma_std_across_users"] = means.std(unbiased=False).item()
    active_means = torch.tensor([row["gamma_mean"] for row in rows if row["has_source_knowledge"]])
    summary["gamma_std_across_users_with_source"] = active_means.std(unbiased=False).item() if len(active_means) else None
    summary["users"] = len(rows)
    summary["by_train_count"] = {}
    for name, lower, upper in (("1_to_4", 1, 4), ("5_to_9", 5, 9), ("10_plus", 10, math.inf)):
        group = [row for row in rows if lower <= row["train_count"] <= upper]
        summary["by_train_count"][name] = {
            "users": len(group),
            **{key: sum(row[key] for row in group) / len(group) if group else None
               for key in ("recall@10", "ndcg@10", "gamma_mean")},
        }
    return summary, rows


@torch.no_grad()
def gate_diagnostics(head, users, graph_users, items, train, knowledge, available, max_history):
    """Training-only probes; missing-source zeros cannot masquerade as dynamic gates."""
    head.eval()
    means, item_stds = [], []
    for uid, history in enumerate(train):
        if not available[uid].any():
            continue
        context, _ = head.context(users[uid], graph_users[uid], items[history[-max_history:]], knowledge[uid], available[uid])
        _, gamma = head.score(users[uid], items[:128], context, len(history))
        means.append(gamma.mean().item())
        item_stds.append(gamma.std(unbiased=False).item())
    values = torch.tensor(means)
    return {
        "scope": "Users with training source knowledge; first up to 128 training-catalog items per user.",
        "users": len(means), "mean": values.mean().item() if len(values) else None,
        "std_across_user_means": values.std(unbiased=False).item() if len(values) else None,
        "mean_std_across_candidate_items": sum(item_stds) / len(item_stds) if item_stds else None,
        "minimum_user_mean": min(means) if means else None,
        "maximum_user_mean": max(means) if means else None,
    }
