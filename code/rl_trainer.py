"""
HA-POMO style trainer for the scheduling GNN policy.

Uses per-step returns for proper credit assignment: each dispatch gives
an intermediate reward (alpha * demand dual contribution), and the final
step gives -cost + mu.  Total return = -reduced_cost (exact decomposition).

POMO: multiple rollouts per instance (each anchored to a different first-task
choice). Shared baseline = mean total return across POMO starts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ColumnGenerationHeuristic import Instance, schedule_cost
from rcpsp_env import RCPSPEnv, DEVICE
from gnn_policy import SchedulingGNN


class POMOTrainer:

    def __init__(
        self,
        policy: SchedulingGNN,
        lr: float = 3e-4,
        num_pomo: int = 8,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        max_delay: int = 10,
        max_modes: int = 3,
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.num_pomo = num_pomo
        self.max_delay = max_delay
        self.max_modes = max_modes
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------

    def _rollout(self, env: RCPSPEnv, greedy: bool = False) -> dict:
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []
        done = False
        obs = env.get_observation()

        while not done:
            node_feat, adj, gf, mask = obs
            logits, value = self.policy(node_feat, adj, gf, mask)

            logits = torch.nan_to_num(logits, nan=float("-inf"))
            if not mask.any():
                break
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax() if greedy else dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())

            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)

            # Safety: if no actions remain, force-finish
            if not done:
                _, _, _, next_mask = obs
                if not next_mask.any():
                    done = True
                    if info:
                        pass
                    else:
                        info = {"cost": 1e6, "makespan": 0, "reduced_cost": 1e6,
                                "alpha_dot_D": 0, "starts": env.get_starts_dict(),
                                "modes": env.get_modes_dict(), "D": {}}

        # Per-step returns: G_t = sum_{t'>=t} r_{t'}
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G += r
            returns.insert(0, G)

        if not log_probs:
            return None

        return {
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
            "entropies": torch.stack(entropies),
            "returns": torch.tensor(returns, dtype=torch.float32, device=DEVICE),
            "total_return": G,
            "info": info,
        }

    # ------------------------------------------------------------------

    def train_on_instance(
        self,
        inst: Instance,
        alpha_duals: Dict[Tuple[str, int], float],
        mu: float,
    ) -> dict:
        self.policy.train()
        env = RCPSPEnv(inst, alpha_duals, mu, max_delay=self.max_delay, max_modes=self.max_modes)

        env.reset()
        initial_eligible = [i for i in range(env.num_tasks) if env.eligible[i]]
        n_anchored = min(self.num_pomo, len(initial_eligible))

        rollouts: List[dict] = []

        for k in range(n_anchored):
            env.reset(anchor_task_idx=initial_eligible[k])
            ro = self._rollout(env)
            if ro is not None:
                rollouts.append(ro)

        for _ in range(max(0, self.num_pomo - n_anchored)):
            env.reset()
            ro = self._rollout(env)
            if ro is not None:
                rollouts.append(ro)

        if not rollouts:
            return {"loss": 0.0, "mean_reward": 0.0, "best_reward": 0.0,
                    "best_rc": 0.0, "best_info": {}}

        # Shared baseline: mean total return across POMO starts
        total_returns = torch.tensor([r["total_return"] for r in rollouts], device=DEVICE)
        baseline = total_returns.mean()
        # Normalize returns for stable gradients
        ret_std = total_returns.std().clamp(min=1.0)

        total_loss = torch.tensor(0.0, device=DEVICE)
        for ro in rollouts:
            normalized_returns = (ro["returns"] - baseline) / ret_std
            advantages = normalized_returns.clamp(-5, 5)
            pg_loss = -(ro["log_probs"] * advantages).mean()
            v_loss = F.mse_loss(ro["values"] / ret_std, ro["returns"] / ret_std)
            entropy = ro["entropies"].mean()
            total_loss = (
                total_loss
                + pg_loss
                + self.value_coef * v_loss
                - self.entropy_coef * entropy
            )

        total_loss = total_loss / len(rollouts)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        best_ro = max(rollouts, key=lambda r: r["total_return"])
        return {
            "loss": total_loss.item(),
            "mean_reward": total_returns.mean().item(),
            "best_reward": total_returns.max().item(),
            "best_rc": -total_returns.max().item(),
            "best_info": best_ro["info"],
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_columns(
        self,
        inst: Instance,
        alpha_duals: Dict[Tuple[str, int], float],
        mu: float,
        num_samples: int = 16,
        greedy_ratio: float = 0.5,
    ) -> List[dict]:
        self.policy.eval()
        env = RCPSPEnv(inst, alpha_duals, mu, max_delay=self.max_delay, max_modes=self.max_modes)
        env.reset()
        initial_eligible = [i for i in range(env.num_tasks) if env.eligible[i]]

        columns: List[dict] = []
        n_greedy = int(num_samples * greedy_ratio)

        for k in range(num_samples):
            anchor = (
                initial_eligible[k % len(initial_eligible)]
                if k < len(initial_eligible)
                else None
            )
            env.reset(anchor_task_idx=anchor)
            greedy = k < n_greedy
            done = False
            info = {}
            obs = env.get_observation()
            while not done:
                nf, adj, gf, mask = obs
                if not mask.any():
                    break
                logits, _ = self.policy(nf, adj, gf, mask)
                logits = torch.nan_to_num(logits, nan=float("-inf"))
                if greedy:
                    action = logits.argmax()
                else:
                    action = torch.distributions.Categorical(logits=logits).sample()
                obs, _, done, info = env.step(action.item())
            if info and "reduced_cost" in info:
                columns.append(info)

        columns.sort(key=lambda c: c["reduced_cost"])
        return columns
