"""
Batched, fully tensorized version of MPPricingEnv: B pricing rollouts advance in lockstep as
torch tensors, so a whole batch of projects x anchors is one set of GPU kernels instead of B
Python environments. Same semantics as mp_pricing_env.MPPricingEnv (verified by
verify_batch_env.py): action (task, mode, delay), start = first feasible period at or after the
predecessor bound plus the delay, reward -c[i,q,s] per step and -w * tardiness + mu at the end,
so the return is still exactly -(reduced cost).

The expensive part is "where can this task start", done here without loops over periods:
  slack   = cap - usage                                  [B,R,W]
  pooled  = sliding-window min of slack over `dur`       (max_pool1d on -slack)
  feasible[b,i,q,s] = min_r pooled[dur][b,r,s] >= demand[b,i,q,r]
  nxt[b,i,q,s]      = first feasible period >= s         (reverse cumulative min)
  start(delay d)    = nxt[..., lb + d]                   (one gather)
Padding: tasks, modes, periods and resources are padded to the batch maximum and masked.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mp_pricing_env import MPPricingEnv

NEG = -1_000_000_000


class BatchPricingEnv:

    STATIC = ("dur", "mode_ok", "kdem", "cap", "coef", "prec", "task_ok", "win", "due_off", "tard_w",
              "mu", "scale", "npred", "coef_res", "coef_mat", "sufmin", "a_dem", "min_dur", "max_dur",
              "tail", "uses_tool", "pi_mean", "pi_max", "is_tool", "cap_ratio", "bt_mean", "bt_max",
              "lead", "inv0", "res_ok", "mat_ok", "adj", "mode_cost", "memb_r", "memb_m", "n_real",
              "win_f", "dur0")

    def __init__(self, problems: List[dict], max_delay: int = 10, max_modes: int = 3,
                 device="cpu", repeat: int = 1):
        """repeat > 1 builds each problem once and clones it (one rollout per anchor), which is
        much cheaper than building the same pricing problem `repeat` times."""
        self.device = torch.device(device)
        self.envs = [MPPricingEnv(p["inst"], p["p"], p["pi"], p["beta"], p["mu"],
                                  max_delay=max_delay, max_modes=max_modes, device="cpu")
                     for p in problems]                      # reused for the static data and duals
        self.problems = problems
        B = len(self.envs)
        self.B, self.Q, self.D = B, max_modes, max_delay + 1
        self.n = max(e.n for e in self.envs)
        self.W = max(e.W for e in self.envs)
        self.R = max(len(e.res) for e in self.envs)
        dev = self.device

        dur = torch.zeros(B, self.n, self.Q, dtype=torch.long)
        mode_ok = torch.zeros(B, self.n, self.Q, dtype=torch.bool)
        kdem = torch.zeros(B, self.n, self.Q, self.R, dtype=torch.long)
        cap = torch.zeros(B, self.R, self.W, dtype=torch.long)
        coef = torch.full((B, self.n, self.Q, self.W), float("inf"))
        prec = torch.zeros(B, self.n, self.n, dtype=torch.bool)      # prec[b,i,j]: i before j
        task_ok = torch.zeros(B, self.n, dtype=torch.bool)
        win = torch.zeros(B, dtype=torch.long)
        due_off = torch.zeros(B, dtype=torch.long)
        tard_w = torch.zeros(B)
        mu = torch.zeros(B)
        scale = torch.ones(B)

        for b, e in enumerate(self.envs):
            task_ok[b, :e.n] = True
            win[b] = e.W
            due_off[b] = e.pr.due - e.rel
            tard_w[b] = e.pr.tardiness_weight
            mu[b] = e.mu[e.p]
            scale[b] = e.scale
            for r_i, r in enumerate(e.res):
                cap[b, r_i, :e.W] = torch.as_tensor(e.cap[r])
            for i, act in enumerate(e.acts):
                for q, md in enumerate(act.modes[:self.Q]):
                    dur[b, i, q] = md.duration
                    mode_ok[b, i, q] = True
                    for r_i, r in enumerate(e.res):
                        kdem[b, i, q, r_i] = md.demand_renewable.get(r, 0)
                    coef[b, i, q, :e.W] = torch.as_tensor(e.coef[i][q], dtype=torch.float32)
            for i, j in e.pr.precedence:
                prec[b, e.idx[i], e.idx[j]] = True

        self.dur, self.mode_ok, self.kdem = dur.to(dev), mode_ok.to(dev), kdem.to(dev)
        self.cap, self.coef, self.prec = cap.to(dev), coef.to(dev), prec.to(dev)
        self.task_ok, self.win = task_ok.to(dev), win.to(dev)
        self.due_off, self.tard_w, self.mu, self.scale = due_off.to(dev), tard_w.to(dev), mu.to(dev), scale.to(dev)
        self.npred = prec.sum(1).to(dev)                              # [B,n]
        self.durs = sorted({int(v) for v in dur.unique() if v > 0})
        self.arange_w = torch.arange(self.W, device=dev)
        self._build_obs_statics()
        if repeat > 1:
            self._expand(repeat)
        self.reset()

    def _expand(self, repeat: int):
        for name in self.STATIC:
            t = getattr(self, name)
            setattr(self, name, t.repeat_interleave(repeat, dim=0))
        self.envs = [e for e in self.envs for _ in range(repeat)]
        self.problems = [p for p in self.problems for _ in range(repeat)]
        self.B *= repeat

    # ------------------------------------------------------------------ static observation data
    def _build_obs_statics(self):
        B, n, Q, W, R = self.B, self.n, self.Q, self.W, self.R
        M = max(len(e.inst.parts) for e in self.envs)
        self.M = M
        dev = self.device
        z = lambda *s: torch.zeros(*s)

        coef_res, coef_mat = torch.zeros(B, n, Q, W), torch.zeros(B, n, Q, W)
        sufmin = torch.full((B, n, Q, W), float("inf"))      # padding past a window: no later start
        a_dem = z(B, n, Q, M)
        min_dur, max_dur, tail = z(B, n), z(B, n), z(B, n)
        uses_tool = torch.zeros(B, n, dtype=torch.bool)
        pi_mean, pi_max, is_tool, cap_ratio = z(B, R), z(B, R), z(B, R), z(B, R)
        bt_mean, bt_max, lead, inv0 = z(B, M), z(B, M), z(B, M), z(B, M)
        res_ok = torch.zeros(B, R, dtype=torch.bool)
        mat_ok = torch.zeros(B, M, dtype=torch.bool)
        adj = torch.zeros(B, n + R + M, n + R + M, dtype=torch.bool)

        for b, e in enumerate(self.envs):
            for i in range(e.n):
                min_dur[b, i] = e.min_dur[i]
                max_dur[b, i] = max(md.duration for md in e.acts[i].modes)
                tail[b, i] = e.tail[i]
                uses_tool[b, i] = bool(e.uses_tool[i])
                for q, md in enumerate(e.acts[i].modes[:Q]):
                    coef_res[b, i, q, :e.W] = torch.as_tensor(e.coef_res[i][q], dtype=torch.float32)
                    coef_mat[b, i, q, :e.W] = torch.as_tensor(e.coef_mat[i][q], dtype=torch.float32)
                    # keep the infinities: the scalar env zeroes that feature when no later
                    # start exists, which is done below in obs()
                    sufmin[b, i, q, :e.W] = torch.as_tensor(np.ascontiguousarray(e.sufmin[i][q]),
                                                            dtype=torch.float32)
                    for mi, m in enumerate(e.inst.parts):
                        a_dem[b, i, q, mi] = md.demand_parts.get(m, 0)
            for r_i, r in enumerate(e.res):
                res_ok[b, r_i] = True
                pi_mean[b, r_i] = e.pi_abs[r].mean() / e.scale
                pi_max[b, r_i] = e.pi_abs[r].max() / e.scale
                is_tool[b, r_i] = float(r == e.tool)
                cap_ratio[b, r_i] = e.cap[r].mean() / max(e.cap[r].max(), 1)
            for mi, m in enumerate(e.inst.parts):
                mat_ok[b, mi] = True
                bt_mean[b, mi] = e.beta_abs[m].mean() / e.scale
                bt_max[b, mi] = e.beta_abs[m].max() / e.scale
                lead[b, mi] = e.inst.lead_time[m] / max(e.W, 1)
                inv0[b, mi] = e.inst.init_inventory.get(m, 0) / 10.0
            a = torch.as_tensor(e.adj)
            k = a.shape[0]                                    # e.n + |res| + |parts|
            nn_, rr = e.n, len(e.res)
            adj[b, :nn_, :nn_] = a[:nn_, :nn_]
            adj[b, :nn_, n:n + rr] = a[:nn_, nn_:nn_ + rr]
            adj[b, n:n + rr, :nn_] = a[nn_:nn_ + rr, :nn_]
            adj[b, :nn_, n + R:n + R + len(e.inst.parts)] = a[:nn_, nn_ + rr:k]
            adj[b, n + R:n + R + len(e.inst.parts), :nn_] = a[nn_ + rr:k, :nn_]
        eye = torch.eye(n + R + M, dtype=torch.bool)
        adj |= eye

        self.coef_res, self.coef_mat, self.sufmin = coef_res.to(dev), coef_mat.to(dev), sufmin.to(dev)
        self.a_dem = a_dem.to(dev)
        self.min_dur, self.max_dur, self.tail = min_dur.to(dev), max_dur.to(dev), tail.to(dev)
        self.uses_tool = uses_tool.to(dev)
        self.pi_mean, self.pi_max, self.is_tool, self.cap_ratio = (t.to(dev) for t in
                                                                   (pi_mean, pi_max, is_tool, cap_ratio))
        self.bt_mean, self.bt_max, self.lead, self.inv0 = (t.to(dev) for t in (bt_mean, bt_max, lead, inv0))
        self.res_ok, self.mat_ok, self.adj = res_ok.to(dev), mat_ok.to(dev), adj.to(dev)
        # duration of mode 0, used by the scalar env's predecessor bound for unplaced predecessors
        self.dur0 = self.dur[:, :, 0].clone()
        self.mode_cost = torch.zeros(B, n, Q, device=dev)
        for b, e in enumerate(self.envs):
            for i in range(e.n):
                for q, md in enumerate(e.acts[i].modes[:Q]):
                    self.mode_cost[b, i, q] = md.cost
        # membership matrices: mean-pool resource / material embeddings per (task, mode) by matmul
        kr = (self.kdem > 0).float()
        self.memb_r = kr / kr.sum(-1, keepdim=True).clamp(min=1)                  # [B,n,Q,R]
        km = (self.a_dem > 0).float()
        self.memb_m = km / km.sum(-1, keepdim=True).clamp(min=1)                  # [B,n,Q,M]
        self.n_real = torch.as_tensor([e.n for e in self.envs], device=dev)
        self.win_f = self.win.clamp(min=1).float()

    # ------------------------------------------------------------------ dynamics
    def reset(self):
        B, n, dev = self.B, self.n, self.device
        self.placed = torch.zeros(B, n, dtype=torch.bool, device=dev)
        self.start = torch.zeros(B, n, dtype=torch.long, device=dev)
        self.mode_sel = torch.zeros(B, n, dtype=torch.long, device=dev)
        self.pred_done = torch.zeros(B, n, dtype=torch.long, device=dev)
        self.lb = torch.zeros(B, n, dtype=torch.long, device=dev)
        self.usage = torch.zeros(B, self.R, self.W, dtype=torch.long, device=dev)
        self.cmax = torch.zeros(B, dtype=torch.long, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)
        self.dead = torch.zeros(B, dtype=torch.bool, device=dev)
        self._starts = None
        self.candidate_starts()
        # a rollout can be stuck before it starts (a task that cannot fit its window at all)
        self.dead = (self.eligible() & ~(self._starts >= 0).any(-1).any(-1)).any(1)
        self.done = self.dead.clone()
        return self._starts

    def eligible(self):
        return self.task_ok & ~self.placed & (self.pred_done == self.npred)

    def candidate_starts(self) -> torch.Tensor:
        """[B,n,Q,D] earliest feasible start per (task, mode, delay); -1 where impossible."""
        B, n, Q, D, W = self.B, self.n, self.Q, self.D, self.W
        slack = (self.cap - self.usage).float()                                   # [B,R,W]
        feas = torch.zeros(B, n, Q, W, dtype=torch.bool, device=self.device)
        for dv in self.durs:
            pooled = -F.max_pool1d(-slack, kernel_size=dv, stride=1)              # [B,R,W-dv+1]
            pooled = F.pad(pooled, (0, dv - 1), value=float(NEG))                 # [B,R,W]
            sel = (self.dur == dv) & self.mode_ok                                 # [B,n,Q]
            if not sel.any():
                continue
            ok = (pooled[:, None, None] >= self.kdem[..., None]).all(3)           # [B,n,Q,W]
            feas |= ok & sel[..., None]
        # the task must also fit inside the window
        last = (self.win[:, None, None] - self.dur).clamp(min=0)                  # [B,n,Q]
        feas &= self.arange_w <= last[..., None]
        feas &= (self.mode_ok & self.eligible()[..., None])[..., None]

        idx = torch.where(feas, self.arange_w.expand_as(feas), torch.full_like(feas, W, dtype=torch.long))
        nxt = idx.flip(-1).cummin(-1).values.flip(-1)                             # first feasible >= s
        nxt = F.pad(nxt, (0, 1), value=W)                                         # index W -> "none"
        thr = (self.lb[:, :, None, None] + torch.arange(D, device=self.device)).clamp(max=W)
        starts = nxt.gather(-1, thr.expand(B, n, Q, D))
        self._starts = torch.where(starts >= W, torch.full_like(starts, -1), starts)
        return self._starts

    def action_mask(self) -> torch.Tensor:
        return (self._starts >= 0).reshape(self.B, -1)

    def step(self, action: torch.Tensor):
        """action [B] flat index ((i*Q)+q)*D+d; finished rollouts ignore their action."""
        B, Q, D = self.B, self.Q, self.D
        live = ~self.done
        i = torch.div(action, Q * D, rounding_mode="floor")
        rem = action - i * Q * D
        q = torch.div(rem, D, rounding_mode="floor")
        d = rem - q * D
        bidx = torch.arange(B, device=self.device)
        s = self._starts[bidx, i, q, d]
        if (live & (s < 0)).any():
            raise ValueError("invalid action in batch step")

        dur = self.dur[bidx, i, q]
        reward = torch.where(live, -self.coef[bidx, i, q, s.clamp(min=0)], torch.zeros_like(self.tard_w))

        # occupy the resources over [s, s+dur)
        t = self.arange_w[None, :]
        window = (t >= s[:, None]) & (t < (s + dur)[:, None]) & live[:, None]     # [B,W]
        self.usage += self.kdem[bidx, i, q][..., None] * window[:, None, :].long()

        upd = live
        self.placed[bidx, i] |= upd
        self.start[bidx, i] = torch.where(upd, s, self.start[bidx, i])
        self.mode_sel[bidx, i] = torch.where(upd, q, self.mode_sel[bidx, i])
        self.cmax = torch.where(upd, torch.maximum(self.cmax, s + dur), self.cmax)
        succ = self.prec[bidx, i] & upd[:, None]                                  # [B,n]
        self.pred_done += succ.long()
        self.lb = torch.where(succ, torch.maximum(self.lb, (s + dur)[:, None]), self.lb)

        self.candidate_starts()
        all_placed = (self.placed | ~self.task_ok).all(1)
        stuck = (self.eligible() & ~(self._starts >= 0).any(-1).any(-1)).any(1)
        newly_done = live & all_placed
        newly_dead = live & ~all_placed & stuck

        tard = (self.cmax - self.due_off).clamp(min=0).float()
        reward = reward + torch.where(newly_done, -self.tard_w * tard + self.mu, torch.zeros_like(reward))
        self.done |= newly_done | newly_dead
        self.dead |= newly_dead
        return reward, self.done.clone(), newly_dead.clone()

    # ------------------------------------------------------------------ observation
    def obs(self) -> Dict[str, torch.Tensor]:
        """Tensor observation matching MPPricingEnv.graph_obs, for the whole batch at once.

        x        [B, n+R+M, 8]     task / resource / material node features
        adj      [B, N, N]         graph (precedence, usage, consumption, self loops)
        g        [B, 7]            global features
        feats    [B, n, Q, D, 12]  per-action "time price" features
        mask     [B, n, Q, D]      valid actions
        memb_r/m [B, n, Q, R|M]    mean-pooling weights of the resource / material embeddings
        """
        B, n, Q, D, W, R, M = self.B, self.n, self.Q, self.D, self.W, self.R, self.M
        sc = self.scale[:, None]
        Wf = self.win_f[:, None]
        s = self._starts.clamp(min=0)                                     # [B,n,Q,D]
        valid = self._starts >= 0

        def at_start(tab):                                                # tab [B,n,Q,W] -> [B,n,Q,D]
            return tab.gather(3, s.clamp(max=W - 1))

        c = torch.where(valid, at_start(self.coef), torch.zeros_like(s, dtype=torch.float))
        c_res, c_mat = at_start(self.coef_res), at_start(self.coef_mat)
        shifted = torch.cat([self.sufmin[..., 1:], torch.full_like(self.sufmin[..., -1:], float("inf"))], -1)
        later = at_start(shifted)
        later_ok = torch.isfinite(later)
        later = torch.where(later_ok, later, torch.zeros_like(later))
        dur = self.dur[..., None].expand(B, n, Q, D)
        end = s + dur
        tool_left = (self.min_dur * (self.uses_tool & ~self.placed).float()).sum(1)        # [B]
        left_after = tool_left[:, None, None, None] - self.min_dur[:, :, None, None] * \
            self.uses_tool[:, :, None, None].float()
        tard_now = (self.cmax - self.due_off).clamp(min=0).float()[:, None, None, None]
        tard_new = (torch.maximum(self.cmax[:, None, None, None], end) -
                    self.due_off[:, None, None, None]).clamp(min=0).float()
        d_idx = torch.arange(D, device=self.device).float().view(1, 1, 1, D)
        W4 = Wf[..., None, None]
        feats = torch.stack([
            c / sc[..., None, None],
            d_idx.expand(B, n, Q, D) / max(D - 1, 1),
            (s - self.lb[:, :, None, None]).float() / W4,
            c_res / sc[..., None, None],
            c_mat / sc[..., None, None],
            (self.mode_cost[..., None] / sc[..., None, None]).expand(B, n, Q, D),
            dur.float() / W4,
            self.tard_w[:, None, None, None] * (tard_new - tard_now) / sc[..., None, None],
            (Wf[..., None, None] - end.float()) / W4,
            left_after / (W4 - end.float()).clamp(min=1),
            torch.where(later_ok, (c - later) / sc[..., None, None], torch.zeros_like(c)),
            (self.tail[:, :, None, None] - self.min_dur[:, :, None, None] + end.float()) / W4,
        ], dim=-1) * valid[..., None].float()

        elig = self.eligible()
        succ_left = (self.prec.float() * (~self.placed).float()[:, None, :]).sum(2)        # [B,n]
        # the scalar env's _prec_lb counts every predecessor, an unplaced one as if it started at
        # 0 in mode 0; only this observation feature depends on that, so mirror it here
        contrib = torch.where(self.placed, (self.start + self.dur.gather(2, self.mode_sel[..., None])
                                            .squeeze(-1)).float(), self.dur0.float())
        lb_obs = (self.prec.float() * contrib[:, :, None]).max(1).values                   # [B,n]
        xt = torch.stack([
            self.placed.float(), elig.float(),
            torch.where(self.placed, self.start.float(), lb_obs) / Wf,
            succ_left / (self.n_real[:, None] - 1).clamp(min=1).float(),
            self.min_dur / Wf, self.max_dur / Wf, self.uses_tool.float(), self.tail / Wf,
        ], dim=-1) * self.task_ok[..., None].float()

        load = self.usage.sum(-1).float() / self.cap.sum(-1).clamp(min=1).float()          # [B,R]
        xr = torch.stack([
            self.pi_mean, self.pi_max, load, self.is_tool,
            torch.where(self.is_tool > 0, tool_left[:, None] / Wf, torch.zeros_like(load)),
            self.cap_ratio, torch.zeros_like(load), torch.zeros_like(load),
        ], dim=-1) * self.res_ok[..., None].float()

        a_min = torch.where(self.mode_ok[..., None], self.a_dem,
                            torch.full_like(self.a_dem, 1e9)).min(2).values               # [B,n,M]
        left_m = (a_min * (~self.placed).float()[..., None] * self.task_ok[..., None].float()).sum(1)
        xm = torch.stack([
            self.bt_mean, self.bt_max, left_m / 10.0, self.lead, self.inv0,
            torch.zeros_like(self.lead), torch.zeros_like(self.lead), torch.zeros_like(self.lead),
        ], dim=-1) * self.mat_ok[..., None].float()

        x = torch.cat([xt, xr, xm], dim=1)
        placed_frac = (self.placed & self.task_ok).sum(1).float() / self.n_real.float()
        elig_frac = (elig & self.task_ok).sum(1).float() / self.n_real.float()
        g = torch.stack([
            self.mu / self.scale, placed_frac, self.cmax.float() / self.win_f,
            (self.cmax - self.due_off).float() / self.win_f, self.tard_w / self.scale,
            1.0 - self.cmax.float() / self.win_f, elig_frac,
        ], dim=-1)
        return {"x": x, "adj": self.adj, "g": g, "feats": feats, "mask": valid,
                "memb_r": self.memb_r, "memb_m": self.memb_m}

    # ------------------------------------------------------------------ results
    def columns(self) -> List[Optional[dict]]:
        """Per rollout: starts/modes in the scalar env's dictionary form, or None if dead."""
        starts = self.start.cpu().numpy()
        modes = self.mode_sel.cpu().numpy()
        out: List[Optional[dict]] = []
        for b, e in enumerate(self.envs):
            if bool(self.dead[b]) or not bool(self.done[b]):
                out.append(None)
                continue
            out.append({"starts": {e.task_ids[i]: e.rel + int(starts[b, i]) for i in range(e.n)},
                        "modes": {e.task_ids[i]: int(modes[b, i]) for i in range(e.n)}})
        return out
