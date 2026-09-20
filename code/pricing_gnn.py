"""
Pricing policy: heterogeneous graph attention network adapted from the lithography DRC
scheduling model (Lithography/gnn_policy.py, DRCSchedulingGNN).

Same structure: type-specific MLP projectors (task / resource / material), residual GAT layers
with layer norm, a global MLP, and an action head that scores every candidate action from
    [h_task || mean h_resources(mode) || mean h_materials(mode) || h_global || f_action]
through a residual MLP. The GAT is dense (masked attention over the adjacency) because a
single project's graph has only ~20 nodes; this avoids a torch_geometric dependency.
Graphs of different sizes are padded into one batch, so all rollouts of all pricing problems
are scored in a single forward pass.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mp_pricing_env import MPPricingEnv

NODE_DIM, ACTION_DIM, GLOBAL_DIM = MPPricingEnv.NODE_DIM, MPPricingEnv.ACTION_DIM, MPPricingEnv.GLOBAL_DIM


def _mlp(i, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))


class DenseGATLayer(nn.Module):
    """h' = h + ReLU(LN(concat_k sum_u alpha_vu^k W^k h_u)), alpha from LeakyReLU(a^T [W h_v || W h_u])."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        assert hidden % heads == 0
        self.heads, self.dh = heads, hidden // heads
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, adj):                          # h [B,N,H], adj [B,N,N] bool
        B, N, _ = h.shape
        wh = self.W(h).view(B, N, self.heads, self.dh)  # [B,N,K,d]
        e_src = (wh * self.a_src).sum(-1)               # [B,N,K] target term
        e_dst = (wh * self.a_dst).sum(-1)               # [B,N,K] neighbour term
        e = F.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1), 0.2)   # [B,N(v),N(u),K]
        e = e.masked_fill(~adj.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha)                 # padded rows have no neighbours
        msg = torch.einsum("bvuk,bukd->bvkd", alpha, wh).reshape(B, N, -1)
        return h + F.relu(self.norm(msg))


class PricingGNNBatch(nn.Module):
    """Same network as PricingGNN, driven by the tensor observation of BatchPricingEnv.

    No Python loop over candidates: resource and material embeddings are mean-pooled per
    (task, mode) with the membership matrices, so scoring all B x n x Q x D actions is a few
    matmuls. Weights are laid out exactly like PricingGNN so checkpoints are interchangeable.
    """

    def __init__(self, hidden: int = 64, layers: int = 3, heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.ModuleList([_mlp(NODE_DIM, hidden) for _ in range(3)])
        self.gat = nn.ModuleList([DenseGATLayer(hidden, heads) for _ in range(layers)])
        self.glob = _mlp(GLOBAL_DIM, hidden)
        self.act_proj = nn.Linear(4 * hidden + ACTION_DIM, hidden)
        self.act_res = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden))
        self.act_out = nn.Linear(hidden, 1)

    def forward(self, obs, n_tasks: int, n_res: int):
        x, adj, g = obs["x"], obs["adj"], obs["g"]
        B, N, _ = x.shape
        h = torch.zeros(B, N, self.hidden, device=x.device)
        h[:, :n_tasks] = self.proj[0](x[:, :n_tasks])
        h[:, n_tasks:n_tasks + n_res] = self.proj[1](x[:, n_tasks:n_tasks + n_res])
        h[:, n_tasks + n_res:] = self.proj[2](x[:, n_tasks + n_res:])
        for layer in self.gat:
            h = layer(h, adj)
        hg = self.glob(g)                                                  # [B,H]

        h_task = h[:, :n_tasks]                                            # [B,n,H]
        h_res = h[:, n_tasks:n_tasks + n_res]
        h_mat = h[:, n_tasks + n_res:]
        res_pool = torch.einsum("bnqr,brh->bnqh", obs["memb_r"], h_res)    # [B,n,Q,H]
        mat_pool = torch.einsum("bnqm,bmh->bnqh", obs["memb_m"], h_mat)

        feats = obs["feats"]                                               # [B,n,Q,D,F]
        D = feats.shape[3]
        parts = [h_task[:, :, None, None].expand(-1, -1, res_pool.shape[2], D, -1),
                 res_pool[:, :, :, None].expand(-1, -1, -1, D, -1),
                 mat_pool[:, :, :, None].expand(-1, -1, -1, D, -1),
                 hg[:, None, None, None].expand(-1, h_task.shape[1], res_pool.shape[2], D, -1),
                 feats]
        u = torch.cat(parts, dim=-1)
        z = self.act_proj(u)
        z = z + self.act_res(z)
        logits = self.act_out(F.relu(z)).squeeze(-1)                       # [B,n,Q,D]
        return logits.masked_fill(~obs["mask"], float("-inf")).reshape(B, -1)


class PricingGNN(nn.Module):

    def __init__(self, hidden: int = 64, layers: int = 3, heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.ModuleList([_mlp(NODE_DIM, hidden) for _ in range(3)])   # task, resource, material
        self.gat = nn.ModuleList([DenseGATLayer(hidden, heads) for _ in range(layers)])
        self.glob = _mlp(GLOBAL_DIM, hidden)
        self.act_proj = nn.Linear(4 * hidden + ACTION_DIM, hidden)
        self.act_res = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden))
        self.act_out = nn.Linear(hidden, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, obs_list: List[dict]) -> List[torch.Tensor]:
        """Logits over each observation's candidate actions (same order as obs['actions'])."""
        dev, B = self.device, len(obs_list)
        N = max(o["x"].shape[0] for o in obs_list)
        x = torch.zeros(B, N, NODE_DIM, device=dev)
        types = torch.full((B, N), -1, dtype=torch.long, device=dev)
        adj = torch.zeros(B, N, N, dtype=torch.bool, device=dev)
        for b, o in enumerate(obs_list):
            k = o["x"].shape[0]
            x[b, :k] = torch.as_tensor(o["x"], device=dev)
            types[b, :k] = torch.as_tensor(o["node_type"], device=dev)
            adj[b, :k, :k] = torch.as_tensor(o["adj"], device=dev)

        h = torch.zeros(B, N, self.hidden, device=dev)
        for t in range(3):
            sel = types == t
            if sel.any():
                h[sel] = self.proj[t](x[sel])
        for layer in self.gat:
            h = layer(h, adj)
        g = self.glob(torch.as_tensor(np.stack([o["global"] for o in obs_list]), device=dev))

        # gather per-candidate inputs for all graphs in one MLP call
        b_idx, t_idx, feats, res_mean, mat_mean, counts = [], [], [], [], [], []
        zero = torch.zeros(self.hidden, device=dev)
        for b, o in enumerate(obs_list):
            counts.append(len(o["actions"]))
            b_idx += [b] * len(o["actions"])
            t_idx += o["task_node"]
            feats.append(torch.as_tensor(o["action_feats"], device=dev))
            for rn, mn in zip(o["res_nodes"], o["mat_nodes"]):
                res_mean.append(h[b, rn].mean(0) if rn else zero)
                mat_mean.append(h[b, mn].mean(0) if mn else zero)
        if not b_idx:
            return [torch.zeros(0, device=dev) for _ in obs_list]
        bi = torch.as_tensor(b_idx, device=dev)
        ti = torch.as_tensor(t_idx, device=dev)
        u = torch.cat([h[bi, ti], torch.stack(res_mean), torch.stack(mat_mean), g[bi], torch.cat(feats)], dim=1)
        z = self.act_proj(u)
        z = z + self.act_res(z)
        logits = self.act_out(F.relu(z)).squeeze(-1)
        return list(torch.split(logits, counts))
