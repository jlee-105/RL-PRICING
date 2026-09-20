"""
MDP for the multi-project pricing problem: build one schedule of project p that
minimizes its reduced cost under the master's duals.

Action = (task, mode, delay). The task must be eligible (all predecessors placed);
it starts at the first period >= (predecessor bound + delay) at which the project's
own capacity (shared resources and the dedicated tool) allows the chosen mode.

Reward: a task started at tau in mode q earns -c[j,q,tau] with
    c[j,q,tau] = gamma_jq - sum_r k_jqr * sum_{t in [tau, tau+p_jq)} pi_rt - sum_m a_jqm * beta_m,tau
and the last step also earns -(w_p * tardiness) + mu_p. The undiscounted return of an
episode is therefore exactly -(reduced cost) of the column it builds.

Observation matches gnn_policy.SchedulingGNN: (node_features, adj_norm, global_features, mask).
Feature sizes depend only on (#shared resources, #parts, max_modes, max_delay), so one policy
serves every project and every number of projects.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from multiproject_instance import MultiProjectInstance
from mp_cg import make_column, reduced_cost

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFEASIBLE_RC = 1e6


class MPPricingEnv:

    def __init__(self, inst: MultiProjectInstance, p: int, pi: Dict, beta: List[Dict], mu: List[float],
                 max_delay: int = 10, max_modes: int = 3, device=DEVICE):
        self.inst, self.p, self.device = inst, p, device
        self.pi, self.beta, self.mu = pi, beta, mu
        pr = inst.projects[p]
        self.pr = pr
        self.rel, self.dl = pr.release, pr.deadline
        self.W = self.dl - self.rel                            # window length

        self.task_ids = list(pr.activities)
        self.n = len(self.task_ids)
        self.idx = {j: i for i, j in enumerate(self.task_ids)}
        self.acts = [pr.activities[j] for j in self.task_ids]
        self.Q, self.D = max_modes, max_delay + 1
        self.num_modes, self.num_delays = self.Q, self.D
        self.num_actions = self.n * self.Q * self.D

        # resources this project can touch: shared ones (canonical order) + its own tool
        self.shared = sorted(r for r in inst.capacity if not r.startswith("TOOL_"))
        self.tool = f"TOOL_{pr.id}" if f"TOOL_{pr.id}" in inst.capacity else None
        self.res = self.shared + ([self.tool] if self.tool else [])
        self.cap = {r: np.array(inst.capacity[r][self.rel:self.dl]) for r in self.res}

        self.preds = [[] for _ in range(self.n)]
        self.succs = [[] for _ in range(self.n)]
        for i, j in pr.precedence:
            self.preds[self.idx[j]].append(self.idx[i])
            self.succs[self.idx[i]].append(self.idx[j])
        adj = torch.eye(self.n)
        for i, j in pr.precedence:
            adj[self.idx[i], self.idx[j]] = adj[self.idx[j], self.idx[i]] = 1.0
        dinv = adj.sum(1).pow(-0.5)
        self.adj_norm = (dinv[:, None] * adj * dinv[None, :]).to(device)

        self._build_costs()
        self._build_graph()
        # feature sizes
        self.per_mode_dim = 2 + len(self.shared) + 1 + len(inst.parts) + self.D
        self.node_feature_dim = 4 + self.Q * self.per_mode_dim
        self.global_feature_dim = 5
        self.reset()

    # ------------------------------------------------------------------ costs
    def _build_costs(self):
        """coef[i][q][s] for start s = rel + offset; +inf where the mode does not fit the window."""
        bp = self.beta[self.p]
        pi_rows = {r: np.array([self.pi.get((r, t), 0.0) for t in range(self.rel, self.dl)]) for r in self.res}
        prefix = {r: np.concatenate([[0.0], np.cumsum(v)]) for r, v in pi_rows.items()}
        self.coef, self.coef_res, self.coef_mat, self.sufmin = [], [], [], []
        for act in self.acts:
            per_mode, per_res, per_mat, per_suf = [], [], [], []
            for md in act.modes:
                c = np.full(self.W, np.inf)
                cr = np.zeros(self.W)
                cm = np.zeros(self.W)
                for s in range(self.W - md.duration + 1):
                    for r, k in md.demand_renewable.items():
                        if r in prefix:
                            cr[s] -= k * (prefix[r][s + md.duration] - prefix[r][s])
                    for m, a in md.demand_parts.items():
                        cm[s] -= a * bp.get((m, self.rel + s), 0.0)
                    c[s] = md.cost + cr[s] + cm[s]
                # suffix minimum: cheapest start at or after s (for the "later is cheaper" feature)
                suf = np.minimum.accumulate(c[::-1])[::-1]
                per_mode.append(c)
                per_res.append(cr)
                per_mat.append(cm)
                per_suf.append(suf)
            self.coef.append(per_mode)
            self.coef_res.append(per_res)
            self.coef_mat.append(per_mat)
            self.sufmin.append(per_suf)
        finite = [abs(x) for pm in self.coef for c in pm for x in c if np.isfinite(x)]
        self.scale = max(1.0, max(finite, default=1.0), self.pr.tardiness_weight)

    # ------------------------------------------------------------------ dynamics
    def reset(self, anchor_task_idx: Optional[int] = None):
        self.placed = np.zeros(self.n, bool)
        self.start = np.zeros(self.n, int)                     # offset from release
        self.mode = np.zeros(self.n, int)
        self.use = {r: np.zeros(self.W, int) for r in self.res}
        self.cmax = 0
        self.carry = 0.0                                        # reward of an anchor placed in reset
        self.dead = False
        self._refresh()
        if anchor_task_idx is not None and self.eligible[anchor_task_idx]:
            a = self._first_valid_action(anchor_task_idx)
            if a is not None:
                _, r, _, _ = self._apply(a)
                self.carry = r
        return self.get_observation()

    def _prec_lb(self, i):
        return max([self.start[k] + self.acts[k].modes[self.mode[k]].duration for k in self.preds[i]], default=0)

    def _fits(self, i, q, s):
        md = self.acts[i].modes[q]
        if s + md.duration > self.W:
            return False
        for r, k in md.demand_renewable.items():
            if r in self.use and np.any(self.use[r][s:s + md.duration] + k > self.cap[r][s:s + md.duration]):
                return False
        return True

    def _start_for(self, i, q, d):
        """First feasible start >= prec bound + d, or -1."""
        md = self.acts[i].modes[q]
        s = self._prec_lb(i) + d
        while s + md.duration <= self.W:
            if self._fits(i, q, s):
                return s
            s += 1
        return -1

    def _refresh(self):
        self.eligible = np.array([not self.placed[i] and all(self.placed[k] for k in self.preds[i])
                                  for i in range(self.n)])
        self.cand = np.full((self.n, self.Q, self.D), -1, int)  # start offset per action, -1 = invalid
        for i in np.flatnonzero(self.eligible):
            for q in range(min(self.Q, len(self.acts[i].modes))):
                prev = None
                for d in range(self.D):
                    s = self._start_for(i, q, d) if prev is None or prev < self._prec_lb(i) + d else prev
                    self.cand[i, q, d] = s
                    prev = s
        # dead end: an unplaced task can never be started (checked for eligible ones)
        if any(self.eligible[i] and (self.cand[i] < 0).all() for i in range(self.n)):
            self.dead = True

    def _first_valid_action(self, i):
        for q in range(self.Q):
            for d in range(self.D):
                if self.cand[i, q, d] >= 0:
                    return (i * self.Q + q) * self.D + d
        return None

    def step(self, action: int):
        obs, r, done, info = self._apply(action)
        r += self.carry
        self.carry = 0.0
        return obs, r, done, info

    def _apply(self, action: int):
        i, rem = divmod(action, self.Q * self.D)
        q, d = divmod(rem, self.D)
        s = self.cand[i, q, d]
        if s < 0:
            raise ValueError("invalid action")
        md = self.acts[i].modes[q]
        self.placed[i], self.start[i], self.mode[i] = True, s, q
        for rr, k in md.demand_renewable.items():
            if rr in self.use:
                self.use[rr][s:s + md.duration] += k
        self.cmax = max(self.cmax, s + md.duration)
        reward = -float(self.coef[i][q][s])
        self._refresh()
        done = bool(self.placed.all())
        info = {}
        if done:
            tard = max(0, self.rel + self.cmax - self.pr.due)
            reward += -self.pr.tardiness_weight * tard + self.mu[self.p]
            col = make_column(self.inst, self.p, self.get_starts_dict(), self.get_modes_dict())
            info = {"column": col, "cost": col.cost,
                    "reduced_cost": reduced_cost(col, self.pi, self.beta, self.mu),
                    "starts": col.starts, "modes": col.modes}
        elif self.dead:
            done = True
            reward += -INFEASIBLE_RC / self.scale
            info = {"column": None, "reduced_cost": INFEASIBLE_RC}
        return self.get_observation(), reward, done, info

    def get_starts_dict(self):
        return {self.task_ids[i]: self.rel + int(self.start[i]) for i in range(self.n) if self.placed[i]}

    def get_modes_dict(self):
        return {self.task_ids[i]: int(self.mode[i]) for i in range(self.n) if self.placed[i]}

    # ------------------------------------------------------------------ graph observation
    # Heterogeneous graph in the style of the lithography DRC model: task, resource, and material
    # nodes with type-specific features, plus per-action "time price" features.
    NODE_DIM = 8
    ACTION_DIM = 12
    GLOBAL_DIM = 7

    def _build_graph(self):
        """Static structure: node list, adjacency, tool flags, remaining critical-path tails."""
        n, S, M = self.n, len(self.res), len(self.inst.parts)
        self.node_type = np.array([0] * n + [1] * S + [2] * M)
        self.res_node = {r: n + k for k, r in enumerate(self.res)}
        self.mat_node = {m: n + S + k for k, m in enumerate(self.inst.parts)}
        N = n + S + M
        adj = np.eye(N, dtype=bool)
        for i, j in self.pr.precedence:
            a, b = self.idx[i], self.idx[j]
            adj[a, b] = adj[b, a] = True
        for i, act in enumerate(self.acts):
            for md in act.modes:
                for r in md.demand_renewable:
                    if r in self.res_node:
                        adj[i, self.res_node[r]] = adj[self.res_node[r], i] = True
                for m in md.demand_parts:
                    adj[i, self.mat_node[m]] = adj[self.mat_node[m], i] = True
        self.adj = adj
        self.uses_tool = np.array([bool(self.tool) and any(self.tool in md.demand_renewable for md in a.modes)
                                   for a in self.acts])
        self.min_dur = np.array([min(md.duration for md in a.modes) for a in self.acts])
        tail = self.min_dur.astype(float).copy()                # longest min-duration path from i to a sink
        for i in reversed(range(n)):                            # ids are topologically ordered
            if self.succs[i]:
                tail[i] = self.min_dur[i] + max(tail[k] for k in self.succs[i])
        self.tail = tail
        # dual summaries per resource / material over the window
        self.pi_abs = {r: np.array([abs(self.pi.get((r, t), 0.0)) for t in range(self.rel, self.dl)])
                       for r in self.res}
        bp = self.beta[self.p]
        self.beta_abs = {m: np.array([abs(bp.get((m, t), 0.0)) for t in range(self.rel, self.dl)])
                         for m in self.inst.parts}

    def candidates(self):
        """Valid actions as (action_id, task, mode, delay, start offset)."""
        out = []
        for i, q, d in zip(*np.nonzero(self.cand >= 0)):
            out.append(((i * self.Q + q) * self.D + d, int(i), int(q), int(d), int(self.cand[i, q, d])))
        return out

    def graph_obs(self):
        W, sc = max(self.W, 1), self.scale
        n, S, M = self.n, len(self.res), len(self.inst.parts)
        x = np.zeros((n + S + M, self.NODE_DIM), np.float32)
        unplaced_tool = self.uses_tool & ~self.placed
        tool_left = float(self.min_dur[unplaced_tool].sum())
        for i, act in enumerate(self.acts):
            x[i] = [self.placed[i], self.eligible[i],
                    (self.start[i] if self.placed[i] else self._prec_lb(i)) / W,
                    sum(not self.placed[k] for k in self.succs[i]) / max(n - 1, 1),
                    self.min_dur[i] / W, max(md.duration for md in act.modes) / W,
                    self.uses_tool[i], self.tail[i] / W]
        for r, v in self.res_node.items():
            load = self.use[r].sum() / max(self.cap[r].sum(), 1)
            x[v] = [self.pi_abs[r].mean() / sc, self.pi_abs[r].max() / sc, load,
                    float(r == self.tool), tool_left / W if r == self.tool else 0.0,
                    self.cap[r].mean() / max(self.cap[r].max(), 1), 0.0, 0.0]
        for m, v in self.mat_node.items():
            left = sum(min(md.demand_parts.get(m, 0) for md in self.acts[i].modes)
                       for i in range(n) if not self.placed[i])
            x[v] = [self.beta_abs[m].mean() / sc, self.beta_abs[m].max() / sc, left / 10.0,
                    self.inst.lead_time[m] / W, self.inst.init_inventory.get(m, 0) / 10.0, 0.0, 0.0, 0.0]

        cands = self.candidates()
        feats = np.zeros((len(cands), self.ACTION_DIM), np.float32)
        res_nodes, mat_nodes = [], []
        tard_now = max(0, self.rel + self.cmax - self.pr.due)
        for k, (_, i, q, d, s) in enumerate(cands):
            md = self.acts[i].modes[q]
            end = s + md.duration
            c = self.coef[i][q][s]
            later = self.sufmin[i][q][s + 1] if s + 1 < W else np.inf
            tard_new = max(0, self.rel + max(self.cmax, end) - self.pr.due)
            left_after = tool_left - (self.min_dur[i] if self.uses_tool[i] else 0.0)
            feats[k] = [c / sc, d / max(self.D - 1, 1), (s - self._prec_lb(i)) / W,
                        self.coef_res[i][q][s] / sc, self.coef_mat[i][q][s] / sc, md.cost / sc,
                        md.duration / W, self.pr.tardiness_weight * (tard_new - tard_now) / sc,
                        (W - end) / W, left_after / max(W - end, 1),
                        (c - later) / sc if np.isfinite(later) else 0.0,
                        (self.tail[i] - self.min_dur[i] + end) / W]
            res_nodes.append([self.res_node[r] for r in md.demand_renewable if r in self.res_node])
            mat_nodes.append([self.mat_node[m] for m in md.demand_parts])
        g = np.array([self.mu[self.p] / sc, self.placed.mean(), self.cmax / W,
                      (self.rel + self.cmax - self.pr.due) / W, self.pr.tardiness_weight / sc,
                      1.0 - self.cmax / W, self.eligible.mean()], np.float32)
        return {"x": x, "node_type": self.node_type, "adj": self.adj, "global": g,
                "actions": [a for a, *_ in cands], "task_node": [i for _, i, *_ in cands],
                "res_nodes": res_nodes, "mat_nodes": mat_nodes, "action_feats": feats}

    # ------------------------------------------------------------------ observation
    def get_action_mask(self):
        return torch.as_tensor((self.cand >= 0).reshape(-1), device=self.device)

    def get_observation(self):
        W, sc = max(self.W, 1), self.scale
        feats = np.zeros((self.n, self.node_feature_dim), np.float32)
        for i, act in enumerate(self.acts):
            f = feats[i]
            f[0] = self.placed[i]
            f[1] = self.eligible[i]
            f[2] = (self.start[i] if self.placed[i] else self._prec_lb(i) if self.eligible[i] else W) / W
            f[3] = sum(not self.placed[k] for k in self.succs[i]) / max(self.n - 1, 1)
            for q in range(min(self.Q, len(act.modes))):
                md = act.modes[q]
                o = 4 + q * self.per_mode_dim
                f[o] = md.duration / W
                for ri, r in enumerate(self.shared):
                    f[o + 1 + ri] = md.demand_renewable.get(r, 0) / max(self.cap[r].max(), 1)
                f[o + 1 + len(self.shared)] = 1.0 if self.tool and md.demand_renewable.get(self.tool, 0) else 0.0
                for mi, m in enumerate(self.inst.parts):
                    f[o + 2 + len(self.shared) + mi] = md.demand_parts.get(m, 0) / 3.0
                f[o + 2 + len(self.shared) + len(self.inst.parts)] = md.cost / sc
                base = o + 3 + len(self.shared) + len(self.inst.parts)
                for d in range(self.D):
                    s = self.cand[i, q, d]
                    # cost of this (mode, delay) at its actual start; 0 when invalid
                    f[base + d] = -self.coef[i][q][s] / sc if s >= 0 else 0.0
        g = np.array([self.mu[self.p] / sc, self.placed.mean(), self.cmax / W,
                      (self.rel + self.cmax - self.pr.due) / W, self.pr.tardiness_weight / sc], np.float32)
        return (torch.as_tensor(feats, device=self.device), self.adj_norm,
                torch.as_tensor(g, device=self.device), self.get_action_mask())
