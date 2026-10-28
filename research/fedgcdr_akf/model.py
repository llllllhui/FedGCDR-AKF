"""A local branch plus upstream graph transfer and conditional gamma."""

import math
from types import SimpleNamespace

import torch
from torch import nn

from .upstream import GAT, MLP

MODES = ("local", "graph_transfer", "fixed", "fixed_capacity", "global", "user", "item")


def make_gat(dim, device):
    return GAT(SimpleNamespace(device=device), dim, dim, dim)


class FusionHead(nn.Module):
    def __init__(self, dim, num_sources, mode="user", fixed_gamma=0.5, device="cpu"):
        super().__init__()
        if mode not in MODES or not 0 <= fixed_gamma <= 1 or dim < 2:
            raise ValueError("Invalid fusion mode, gamma range or embedding dimension.")
        self.mode, self.fixed_gamma = mode, fixed_gamma
        self.gat = make_gat(dim, device)
        self.mappers = nn.ModuleList([MLP(dim) for _ in range(num_sources)])
        # The enlarged fixed head has approximately the user gate's parameter budget.
        hidden = 2 * dim + 1 if mode == "fixed_capacity" else dim
        self.f1 = nn.Sequential(nn.Linear(2 * dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.f2 = nn.Linear(1, 1)
        nn.init.ones_(self.f2.weight)
        nn.init.zeros_(self.f2.bias)
        if mode == "global":
            initial = min(max(fixed_gamma, 1e-6), 1 - 1e-6)
            self.gamma_logit = nn.Parameter(torch.tensor(math.log(initial / (1 - initial))))
        elif mode in {"user", "item"}:
            width = (3 if mode == "item" else 2) * dim + 1
            self.gate = nn.Sequential(nn.Linear(width, dim), nn.Tanh(), nn.Linear(dim, 1), nn.Sigmoid())
        self.to(device)

    def active_parameters(self):
        if self.mode == "local":
            return []
        if self.mode == "graph_transfer":
            return [*self.gat.parameters(), *self.mappers.parameters()]
        return list(self.parameters())

    def context(self, local_user, graph_user, neighbors, knowledge, available):
        if self.mode == "local":
            return None, local_user.new_zeros(())
        mapped = [mapper(vector.detach()) for mapper, vector, present in
                  zip(self.mappers, knowledge, available) if bool(present)]
        if not mapped and self.mode != "graph_transfer":
            return None, local_user.new_zeros(())
        nodes = torch.cat((graph_user.unsqueeze(0), neighbors), dim=0)
        graph, _, ls, lm = self.gat(nodes, bool(mapped), transfer_vec=mapped or None)
        return graph[0], ls + lm

    def score(self, local_user, item_vectors, transferred_user, train_count):
        local_score = (local_user * item_vectors).sum(dim=-1)
        if self.mode == "local" or transferred_user is None:
            # Missing source knowledge contributes exactly zero, including mapper biases.
            return local_score, torch.zeros_like(local_score)
        if self.mode == "graph_transfer":
            return (transferred_user * item_vectors).sum(dim=-1), torch.ones_like(local_score)
        transferred = transferred_user.expand(len(item_vectors), -1)
        transfer_score = self.f1(torch.cat((transferred, item_vectors), dim=-1)).squeeze(-1)
        if self.mode in {"fixed", "fixed_capacity"}:
            gamma = torch.full_like(local_score, self.fixed_gamma)
        elif self.mode == "global":
            gamma = self.gamma_logit.sigmoid().expand_as(local_score)
        else:
            count = item_vectors.new_full((len(item_vectors), 1), math.log1p(train_count))
            features = [local_user.expand(len(item_vectors), -1), transferred, count]
            if self.mode == "item":
                features.append(item_vectors)
            gamma = self.gate(torch.cat(features, dim=-1)).squeeze(-1)
        logits = self.f2((local_score + gamma * transfer_score).unsqueeze(-1)).squeeze(-1)
        return logits, gamma

    def forward(self, local_user, graph_user, neighbors, item_vectors, knowledge, available, train_count):
        transferred_user, regularizer = self.context(local_user, graph_user, neighbors, knowledge, available)
        logits, gamma = self.score(local_user, item_vectors, transferred_user, train_count)
        return logits, gamma, regularizer
