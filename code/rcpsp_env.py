"""
MDP environment for MRCPSP + Procurement pricing subproblem.

Action space: (task, mode, delay) triples.  At each step the agent picks an
eligible task, a processing mode, AND a delay offset from its earliest feasible
start.  Mode selection determines resource/material consumption and duration.
The delay lets the agent explicitly push tasks past lead-time windows where
dual prices penalise early consumption.

Node features include per-mode dual rewards at EACH candidate delay, so the
policy sees exactly how each (mode, delay) combination costs in dual space.
"""
from __future__ import annotations

import torch
from typing import Dict, List, Tuple, Optional

from ColumnGenerationHeuristic import (
    Instance,
    schedule_cost,
    PENALTY_SLACK,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RCPSPEnv:

    def __init__(
        self,
        inst: Instance,
        alpha_duals: Optional[Dict[Tuple[str, int], float]] = None,
        mu_dual: float = 0.0,
        max_delay: int = 10,
        max_modes: int = 3,
    ):
        self.inst = inst
        self.alpha_duals = alpha_duals or {}
        self.mu_dual = mu_dual

        self.task_ids = sorted(inst.activities.keys())
        self.num_tasks = len(self.task_ids)
        self.task_id_to_idx = {tid: i for i, tid in enumerate(self.task_ids)}
        self.resource_ids = sorted(inst.renewable_capacity.keys())
        self.num_resources = len(self.resource_ids)
        self.num_parts = len(inst.parts)

        # Modes
        self.num_modes = max_modes
        self.task_num_modes = [inst.activities[tid].num_modes for tid in self.task_ids]

        # Precedence
        self.predecessors: Dict[str, List[str]] = {t: [] for t in self.task_ids}
        self.successors: Dict[str, List[str]] = {t: [] for t in self.task_ids}
        for i, j in inst.precedence:
            self.predecessors[j].append(i)
            self.successors[i].append(j)

        # Adjacency
        adj = torch.zeros(self.num_tasks, self.num_tasks)
        for i, j in inst.precedence:
            ii, jj = self.task_id_to_idx[i], self.task_id_to_idx[j]
            adj[ii, jj] = 1.0
            adj[jj, ii] = 1.0
        adj += torch.eye(self.num_tasks)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        self.adj_norm = (deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)).to(DEVICE)

        # Delay options: [0, 1, ..., max_delay]
        self.delay_options = list(range(max_delay + 1))
        self.num_delays = len(self.delay_options)

        # Action space: task_idx * (num_modes * num_delays) + mode_idx * num_delays + delay_idx
        self.num_actions = self.num_tasks * self.num_modes * self.num_delays

        # Feature dims
        # Base: [scheduled | eligible | efs | succ_ratio] = 4
        # Per mode (Q modes): [duration | res_demands(R) | part_demands(M) | mode_cost | dual_at_delay_0..D-1]
        self.per_mode_dim = 2 + self.num_resources + self.num_parts + self.num_delays
        self.node_feature_dim = 4 + self.num_modes * self.per_mode_dim
        self.global_feature_dim = 3  # mu, frac_scheduled, makespan

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, anchor_task_idx: Optional[int] = None):
        self.scheduled = [False] * self.num_tasks
        self.start_times = [0] * self.num_tasks
        self.selected_modes = [0] * self.num_tasks
        self.num_scheduled = 0
        self.current_makespan = 0

        H = self.inst.horizon
        self.resource_usage: Dict[str, List[int]] = {
            r: [0] * (H + 1) for r in self.resource_ids
        }
        self._update_eligible()

        if anchor_task_idx is not None and anchor_task_idx < self.num_tasks:
            if self.eligible[anchor_task_idx]:
                self._dispatch(anchor_task_idx, 0, 0)

        return self.get_observation()

    def step(self, action: int):
        task_idx = action // (self.num_modes * self.num_delays)
        remainder = action % (self.num_modes * self.num_delays)
        mode_idx = remainder // self.num_delays
        delay_idx = remainder % self.num_delays
        return self._dispatch(task_idx, mode_idx, self.delay_options[delay_idx])

    # ------------------------------------------------------------------
    def _dispatch(self, task_idx: int, mode_idx: int, delay: int):
        if not self.eligible[task_idx]:
            eligible_tasks = [i for i in range(self.num_tasks) if self.eligible[i]]
            if not eligible_tasks:
                return self.get_observation(), 0.0, True, {
                    "cost": 1e6, "makespan": 0, "reduced_cost": 1e6,
                    "alpha_dot_D": 0, "starts": self.get_starts_dict(),
                    "modes": self.get_modes_dict(), "D": {},
                }
            task_idx = eligible_tasks[0]
            mode_idx = 0
            delay = 0

        tid = self.task_ids[task_idx]
        act = self.inst.activities[tid]
        mode = act.modes[mode_idx]

        start = self._find_feasible(task_idx, mode_idx, self.earliest_starts[task_idx] + delay)
        if start < 0:
            start = self._find_feasible(task_idx, mode_idx, self.earliest_starts[task_idx])
            if start < 0:
                start = self.earliest_starts[task_idx]

        self.scheduled[task_idx] = True
        self.start_times[task_idx] = start
        self.selected_modes[task_idx] = mode_idx
        self.num_scheduled += 1
        self.current_makespan = max(self.current_makespan, start + mode.duration)

        for r in self.resource_ids:
            d = mode.demand_renewable.get(r, 0)
            if d:
                for t in range(start, min(start + mode.duration, self.inst.horizon + 1)):
                    self.resource_usage[r][t] += d

        # Per-task dual contribution as step reward
        step_alpha = 0.0
        for m, amt in mode.demand_parts.items():
            if amt:
                step_alpha += self.alpha_duals.get((m, start), 0.0) * amt

        self._update_eligible()

        done = self.num_scheduled == self.num_tasks
        reward = step_alpha
        info: dict = {}
        if done:
            final_r, info = self._final_reward()
            reward += final_r
        return self.get_observation(), reward, done, info

    def _final_reward(self):
        starts_dict = self.get_starts_dict()
        modes_dict = self.get_modes_dict()
        cost, makespan, D = schedule_cost(self.inst, starts_dict, modes_dict)
        alpha_dot_D = 0.0
        for m in self.inst.parts:
            for t, d_mt in enumerate(D[m]):
                if d_mt:
                    alpha_dot_D += self.alpha_duals.get((m, t), 0.0) * d_mt
        rc = cost - alpha_dot_D - self.mu_dual
        return -cost + self.mu_dual, {
            "cost": cost, "makespan": makespan, "reduced_cost": rc,
            "alpha_dot_D": alpha_dot_D, "starts": starts_dict,
            "modes": modes_dict, "D": D,
        }

    # ------------------------------------------------------------------
    def _update_eligible(self):
        self.eligible = [False] * self.num_tasks
        self.earliest_starts = [0] * self.num_tasks
        for idx, tid in enumerate(self.task_ids):
            if self.scheduled[idx]:
                continue
            if all(self.scheduled[self.task_id_to_idx[p]] for p in self.predecessors[tid]):
                act = self.inst.activities[tid]
                min_dur = min(mode.duration for mode in act.modes)
                efs = self._find_feasible_any_mode(idx, self._prec_lb(idx))
                if efs >= 0:
                    self.eligible[idx] = True
                    self.earliest_starts[idx] = efs

    def _prec_lb(self, task_idx: int) -> int:
        lb = 0
        for p in self.predecessors[self.task_ids[task_idx]]:
            pi = self.task_id_to_idx[p]
            if self.scheduled[pi]:
                pact = self.inst.activities[p]
                lb = max(lb, self.start_times[pi] + pact.modes[self.selected_modes[pi]].duration)
        return lb

    def _find_feasible_any_mode(self, task_idx: int, lb: int) -> int:
        """Find earliest feasible start considering the shortest-duration mode."""
        tid = self.task_ids[task_idx]
        act = self.inst.activities[tid]
        min_dur_mode = min(range(act.num_modes), key=lambda q: act.modes[q].duration)
        return self._find_feasible(task_idx, min_dur_mode, lb)

    def _find_feasible(self, task_idx: int, mode_idx: int, lb: int) -> int:
        tid = self.task_ids[task_idx]
        act = self.inst.activities[tid]
        mode = act.modes[mode_idx]
        H = self.inst.horizon
        t = lb
        while t + mode.duration <= H:
            ok = True
            for r in self.resource_ids:
                dr = mode.demand_renewable.get(r, 0)
                if dr == 0:
                    continue
                cap = self.inst.renewable_capacity[r]
                for tp in range(t, t + mode.duration):
                    if self.resource_usage[r][tp] + dr > cap:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return t
            t += 1
        return -1

    def get_starts_dict(self) -> Dict[str, int]:
        return {self.task_ids[i]: self.start_times[i]
                for i in range(self.num_tasks) if self.scheduled[i]}

    def get_modes_dict(self) -> Dict[str, int]:
        return {self.task_ids[i]: self.selected_modes[i]
                for i in range(self.num_tasks) if self.scheduled[i]}

    def get_action_mask(self) -> torch.Tensor:
        """(num_tasks * num_modes * num_delays,) bool mask."""
        H = self.inst.horizon
        mask = [False] * self.num_actions
        for idx in range(self.num_tasks):
            if not self.eligible[idx]:
                continue
            efs = self.earliest_starts[idx]
            act = self.inst.activities[self.task_ids[idx]]
            for qi in range(self.num_modes):
                if qi >= act.num_modes:
                    continue
                dur = act.modes[qi].duration
                for di, d in enumerate(self.delay_options):
                    if efs + d + dur <= H:
                        action_idx = idx * (self.num_modes * self.num_delays) + qi * self.num_delays + di
                        mask[action_idx] = True
        return torch.tensor(mask, dtype=torch.bool, device=DEVICE)

    def get_observation(self):
        H = self.inst.horizon
        feats = []
        for idx, tid in enumerate(self.task_ids):
            act = self.inst.activities[tid]
            f: list[float] = []

            # Base features (4)
            f.append(float(self.scheduled[idx]))
            f.append(float(self.eligible[idx]))
            if self.eligible[idx]:
                f.append(self.earliest_starts[idx] / H)
            elif self.scheduled[idx]:
                f.append(self.start_times[idx] / H)
            else:
                f.append(1.0)
            n_succ = sum(1 for s in self.successors[tid]
                         if not self.scheduled[self.task_id_to_idx[s]])
            f.append(n_succ / max(self.num_tasks - 1, 1))

            # Per-mode features (num_modes * per_mode_dim)
            for qi in range(self.num_modes):
                if qi < act.num_modes:
                    mode = act.modes[qi]
                    f.append(mode.duration / H)
                    for r in self.resource_ids:
                        f.append(mode.demand_renewable.get(r, 0) / max(self.inst.renewable_capacity[r], 1))
                    for m in self.inst.parts:
                        f.append(float(mode.demand_parts.get(m, 0)))
                    f.append(mode.cost / 100.0)

                    # Dual reward at each delay for this mode
                    for d in self.delay_options:
                        if self.eligible[idx] and self.alpha_duals:
                            t = self.earliest_starts[idx] + d
                            dr = sum(self.alpha_duals.get((m, t), 0.0) * mode.demand_parts.get(m, 0)
                                     for m in self.inst.parts)
                            f.append(dr / PENALTY_SLACK)
                        else:
                            f.append(0.0)
                else:
                    # Pad with zeros for tasks with fewer modes
                    f.extend([0.0] * self.per_mode_dim)

            feats.append(f)

        node_features = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
        global_features = torch.tensor([
            self.mu_dual / PENALTY_SLACK,
            self.num_scheduled / self.num_tasks,
            self.current_makespan / H,
        ], dtype=torch.float32, device=DEVICE)

        return node_features, self.adj_norm, global_features, self.get_action_mask()
