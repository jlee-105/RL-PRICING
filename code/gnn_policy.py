"""
GNN scheduling policy for RL-based CG pricing.

Outputs (task, mode, delay) logits: for each eligible task, scores each
(mode, delay) combination.  Mode selection determines resource/material
consumption; delay lets the agent push tasks past lead-time windows where
dual prices penalise early consumption.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(adj @ x))


class SchedulingGNN(nn.Module):
    def __init__(self, node_dim: int, global_dim: int,
                 num_modes: int, num_delays: int,
                 hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_modes = num_modes
        self.num_delays = num_delays
        self.num_mode_delays = num_modes * num_delays

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.gcn_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gcn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )

        # Per-node mode-delay logits: (N, num_modes * num_delays)
        self.mode_delay_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.num_mode_delays),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, adj, global_features, action_mask):
        """
        Returns
        -------
        logits : (N * num_modes * num_delays,)  flat, masked
        value  : scalar
        """
        h = self.node_encoder(node_features)
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h = norm(gcn(h, adj) + h)

        g = self.global_encoder(global_features)
        g_exp = g.unsqueeze(0).expand(h.size(0), -1)
        hg = torch.cat([h, g_exp], dim=-1)  # (N, 2H)

        md_logits = self.mode_delay_head(hg)  # (N, Q*D)
        flat_logits = md_logits.reshape(-1)   # (N*Q*D,)
        flat_logits = flat_logits.masked_fill(~action_mask, float("-inf"))

        h_mean = h.mean(dim=0)
        value = self.value_head(torch.cat([h_mean, g])).squeeze(-1)

        return flat_logits, value
