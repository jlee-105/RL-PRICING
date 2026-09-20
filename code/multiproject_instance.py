"""
Multi-project tool ramp-up instances.

A fab ramps up many tools at once, and tools of the same type go through the
same Install/Qual process. Each project p is one tool: it is cloned from one
of a few process templates (tool types) with small duration perturbations,
arrives at a release date r_p (tool move-in) and has a ramp target d_p.
Projects share renewable resources (engineers, trades, metrology) with a
per-period capacity K[r][t], and share non-renewable resources (test wafers,
chemicals, spare parts) that are held in one inventory and replenished by
orders with lead times.

Code reuses Mode/Activity from ColumnGenerationHeuristic; there the
non-renewable resources are called `parts` (Mode.demand_parts).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ColumnGenerationHeuristic import Activity, Mode, Instance


@dataclass
class Project:
    id: str
    template: int
    activities: Dict[str, Activity]       # ids are namespaced "<project>_<task>"
    precedence: List[Tuple[str, str]]
    release: int                          # r_p: no task starts before it (earliest possible arrival)
    due: int                              # d_p: ramp target, tardiness beyond it
    deadline: int                         # every task must finish by it (bounds the pricing window)
    tardiness_weight: float               # w_p per period late
    tool_lead: int = 0                    # periods between ordering the tool and its arrival
    tool_cost: float = 0.0                # capital cost of the tool, charged in the order period
    order_latest: int = 0                 # last period the tool may be ordered (arrival <= deadline)


@dataclass
class MultiProjectInstance:
    projects: List[Project]
    capacity: Dict[str, List[int]]        # K[r][t], t = 0..H-1
    parts: List[str]                      # non-renewable resources
    lead_time: Dict[str, int]
    init_inventory: Dict[str, int]
    unit_cost: Dict[str, float]
    setup_cost: Dict[str, float]
    holding_cost: Dict[str, float]
    horizon: int
    capex_budget: Optional[List[float]] = None   # B_t: capital that may be committed in period t
    meta: dict = field(default_factory=dict)

    @property
    def resources(self) -> List[str]:
        return list(self.capacity)

    def project_instance(self, p: int) -> Instance:
        """Project p alone as a single-project Instance (constant capacity = min over its window)."""
        pr = self.projects[p]
        cap = {r: min(self.capacity[r][pr.release:pr.deadline]) for r in self.capacity}
        return Instance(pr.activities, pr.precedence, cap, self.parts, self.lead_time,
                        self.init_inventory, pr.deadline, self.unit_cost, self.setup_cost,
                        self.holding_cost)


def critical_path(acts: Dict[str, Activity], prec: List[Tuple[str, str]], fastest: bool = True) -> int:
    """Longest path with each task at its shortest (or first-mode) duration; ids must be topologically ordered."""
    dur = {j: (min(m.duration for m in a.modes) if fastest else a.modes[0].duration) for j, a in acts.items()}
    finish: Dict[str, int] = {}
    for j in acts:
        finish[j] = max([finish[i] for i, k in prec if k == j], default=0) + dur[j]
    return max(finish.values())


def _make_template(rng: random.Random, num_tasks: int, resources: List[str], parts: List[str],
                   max_modes: int, max_dur: int, prec_density: float):
    """Random index-ordered DAG with multi-mode tasks (same distributions as instance_generator)."""
    acts: Dict[str, Activity] = {}
    for i in range(num_tasks):
        base_dur = rng.randint(1, max_dur)
        base_k = {r: rng.randint(1, 3) for r in resources if rng.random() < 0.7}
        base_a = {m: rng.randint(1, 3) for m in parts if rng.random() < 0.4}
        modes = [Mode(base_dur, dict(base_k), dict(base_a), 0.0)]
        for _ in range(1, rng.randint(1, max_modes)):
            d = max(1, base_dur + rng.randint(-max(base_dur // 3, 1), max(base_dur // 3, 1)))
            k = {r: max(1, v + rng.choice([-1, 0, 1])) for r, v in base_k.items()}
            a = {m: max(1, v + rng.choice([-1, 0, 1])) for m, v in base_a.items()}
            modes.append(Mode(d, k, a, round(rng.uniform(0.0, 5.0), 1)))
        acts[f"T{i}"] = Activity(f"T{i}", modes)
    prec = [(f"T{i}", f"T{j}") for i in range(num_tasks) for j in range(i + 1, num_tasks)
            if rng.random() < prec_density]
    return acts, prec


def generate_multiproject_instance(
    num_projects: int = 5,
    num_templates: int = 3,
    tasks_per_project: int = 10,
    num_resources: int = 2,
    num_parts: int = 3,
    max_modes: int = 3,
    max_dur: int = 4,
    prec_density: float = 0.25,
    overlap: float = 2.0,             # target number of projects in progress at once
    due_slack: float = 1.3,           # d_p = r_p + ceil(due_slack * CP_p)
    window_slack: float = 2.5,        # deadline = r_p + ceil(window_slack * CP_p)
    capacity_tightness: float = 1.0,  # <1 tighter, >1 looser shared capacity
    vacation_period: int = 0,         # e.g. 7: last `vacation_len` periods of every week run at reduced capacity
    vacation_len: int = 2,
    vacation_factor: float = 0.5,
    rho: float = 1.0,                 # cost of one day late, as a fraction of the project's cost per CP day
    tool_unary: str = "none",         # "qual": tasks after the first third occupy the tool itself (capacity 1);
                                      # "all": every task does; "none": no tool resource
    plan_offset: float = 1.0,         # planning time before the first tool arrives, in units of the
                                      # longest lead time; < 1 makes lead times bind the schedule
    tool_purchase: bool = False,      # the tool itself is ordered with a long lead time; its arrival
                                      # (the release date) becomes a decision, limited by a capex budget
    tool_lead_range: Tuple[int, int] = (10, 25),
    tool_cost_range: Tuple[float, float] = (200.0, 600.0),
    capex_per_period: float = 1.0,    # capital per period, in "average tools"; the budget is never
                                      # below the most expensive tool, so any single order is possible
    seed: Optional[int] = None,
) -> MultiProjectInstance:
    rng = random.Random(seed)
    resources = [f"R{r}" for r in range(num_resources)]
    parts = [f"M{m}" for m in range(num_parts)]

    lead_time = {m: rng.randint(2, 8) for m in parts}
    init_inventory = {m: rng.randint(0, 3) for m in parts}
    unit_cost = {m: round(rng.uniform(1.0, 10.0), 1) for m in parts}
    setup_cost = {m: round(rng.uniform(5.0, 20.0), 1) for m in parts}
    holding_cost = {m: round(rng.uniform(0.1, 1.0), 2) for m in parts}

    templates = [_make_template(rng, tasks_per_project, resources, parts, max_modes, max_dur, prec_density)
                 for _ in range(num_templates)]
    # Bottleneck tool types lose more output per day of delay
    type_factor = [round(rng.uniform(0.5, 1.5), 2) for _ in range(num_templates)]

    # Clone projects from templates; perturb durations a little (same tool type, different tool)
    raw = []
    for p in range(num_projects):
        k = rng.randrange(num_templates)
        t_acts, t_prec = templates[k]
        acts = {}
        for j, a in t_acts.items():
            modes = [Mode(max(1, m.duration + rng.choice([-1, 0, 0, 0, 1])), dict(m.demand_renewable),
                          dict(m.demand_parts), m.cost) for m in a.modes]
            acts[f"P{p}_{j}"] = Activity(f"P{p}_{j}", modes)
        prec = [(f"P{p}_{i}", f"P{p}_{j}") for i, j in t_prec]
        # The tool is a unary resource dedicated to its own project: qualification runs one test at
        # a time on it. Tool tasks are then serialized, so the project needs at least their total time.
        span = critical_path(acts, prec)
        if tool_unary != "none":
            first = 0 if tool_unary == "all" else tasks_per_project // 3
            tool_tasks = [a for idx, a in enumerate(acts.values()) if idx >= first]
            for a in tool_tasks:
                for md in a.modes:
                    md.demand_renewable[f"TOOL_P{p}"] = 1
            span = max(span, sum(min(md.duration for md in a.modes) for a in tool_tasks))
        raw.append((k, acts, prec, span))

    # Staggered move-in dates so that about `overlap` projects run concurrently
    mean_span = sum(math.ceil(window_slack * cp) for *_, cp in raw) / num_projects
    gap = max(1.0, mean_span / (window_slack / due_slack) / overlap)
    releases = sorted(int(round(i * gap + rng.uniform(-gap / 3, gap / 3))) for i in range(num_projects))
    # How much planning time there is before the first tool arrives. At 1.0 the longest lead time
    # fits before the first release, so material is never what delays a task and only its cost
    # matters. Below 1.0 the early tasks of the first tools can only use the initial stock, so the
    # lead times constrain the schedule as they do in a real ramp-up.
    offset = int(round(plan_offset * max(lead_time.values())))
    releases = [offset + max(0, r) for r in releases]
    rng.shuffle(releases)                     # tool type and arrival order are unrelated

    # Tardiness in money: w_p [$ / period] = rho * type factor * (project cost / CP_p), where
    # project cost = material units at unit price + average mode cost (setups are shared, left out)
    projects = []
    for p, ((k, acts, prec, cp), r) in enumerate(zip(raw, releases)):
        proj_cost = sum(sum(a_m * unit_cost[m] for m, a_m in act.modes[0].demand_parts.items())
                        + sum(md.cost for md in act.modes) / len(act.modes) for act in acts.values())
        tool_lead = rng.randint(*tool_lead_range) if tool_purchase else 0
        projects.append(Project(f"P{p}", k, acts, prec, release=r,
                                due=r + math.ceil(due_slack * cp),
                                deadline=r + math.ceil(window_slack * cp),
                                tardiness_weight=round(rho * type_factor[k] * proj_cost / cp, 2),
                                tool_lead=tool_lead,
                                tool_cost=round(rng.uniform(*tool_cost_range), 1) if tool_purchase else 0.0))
    if tool_purchase:
        # The tool is bought, so its arrival is a decision: ordering in period o makes it arrive at
        # o + tool_lead. Earliest arrival is therefore the lead time itself (order at 0), and the
        # project may be ordered up to `span` periods later and still fit before its deadline.
        # The ramp target d_p keeps the staggered dates, so a late order is paid for in tardiness.
        for pr in projects:
            span = pr.deadline - pr.release
            pr.release = pr.tool_lead
            pr.order_latest = span
            pr.deadline = pr.release + 2 * span
    H = max(pr.deadline for pr in projects)

    # Shared capacity sized to the average load of `overlap` concurrent projects
    capacity = {}
    for r in resources:
        loads = [sum(a.modes[0].demand_renewable.get(r, 0) * a.modes[0].duration for a in acts.values()) / cp
                 for _, acts, _, cp in raw]
        peak = max((m.demand_renewable.get(r, 0) for _, acts, _, _ in raw
                    for a in acts.values() for m in a.modes), default=0)
        K = max(peak, math.ceil(capacity_tightness * overlap * sum(loads) / len(loads)))
        cap_t = [K] * H
        if vacation_period:
            for t in range(H):
                if t % vacation_period >= vacation_period - vacation_len:
                    cap_t[t] = max(peak, int(K * vacation_factor))
        capacity[r] = cap_t
    if tool_unary != "none":
        for p in range(num_projects):
            capacity[f"TOOL_P{p}"] = [1] * H

    return MultiProjectInstance(
        projects=projects,
        capacity=capacity,
        parts=parts,
        lead_time=lead_time,
        init_inventory=init_inventory,
        unit_cost=unit_cost,
        setup_cost=setup_cost,
        holding_cost=holding_cost,
        horizon=H,
        capex_budget=([max(max(pr.tool_cost for pr in projects),
                           capex_per_period * sum(pr.tool_cost for pr in projects) / len(projects))] * H
                      if tool_purchase else None),
        meta={"seed": seed, "num_templates": num_templates, "overlap": overlap,
              "due_slack": due_slack, "window_slack": window_slack,
              "capacity_tightness": capacity_tightness, "vacation_period": vacation_period,
              "rho": rho, "type_factor": type_factor, "tool_unary": tool_unary,
              "plan_offset": plan_offset, "tool_purchase": tool_purchase,
              "capex_per_period": capex_per_period},
    )


def summarize(inst: MultiProjectInstance) -> str:
    P = len(inst.projects)
    tasks = sum(len(pr.activities) for pr in inst.projects)
    modes = sum(len(a.modes) for pr in inst.projects for a in pr.activities.values())
    lines = [f"projects={P}  tasks={tasks}  modes={modes}  horizon={inst.horizon}  "
             f"resources={ {r: max(c) for r, c in inst.capacity.items()} }  parts={len(inst.parts)}",
             f"lead={inst.lead_time}  I0={inst.init_inventory}"]
    for pr in inst.projects:
        lines.append(f"  {pr.id}: type={pr.template}  r={pr.release:3d}  d={pr.due:3d}  "
                     f"deadline={pr.deadline:3d}  w={pr.tardiness_weight}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(generate_multiproject_instance(num_projects=5, seed=0)))
