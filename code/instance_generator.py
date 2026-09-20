"""
Random MRCPSP + Procurement instance generator for RL training.

Generates DAG-structured task graphs with random multi-mode resource / part demands,
precedence, lead times, and procurement costs.
"""
from __future__ import annotations

import random
from typing import Optional

from ColumnGenerationHeuristic import Activity, Mode, Instance


def generate_random_instance(
    num_tasks: int = 10,
    num_resources: int = 2,
    num_parts: int = 3,
    horizon: int = 50,
    precedence_density: float = 0.25,
    max_modes: int = 1,
    seed: Optional[int] = None,
) -> Instance:
    if seed is not None:
        random.seed(seed)

    task_ids = [f"T{i}" for i in range(num_tasks)]
    resource_ids = [f"R{r}" for r in range(num_resources)]
    part_ids = [f"M{m}" for m in range(num_parts)]

    activities = {}
    max_dur = max(horizon // num_tasks, 2)
    for tid in task_ids:
        num_modes_j = random.randint(1, max_modes) if max_modes > 1 else 1
        base_dur = random.randint(1, max_dur)
        base_dreq = {r: random.randint(1, 3) for r in resource_ids if random.random() < 0.7}
        base_preq = {m: random.randint(1, 3) for m in part_ids if random.random() < 0.4}

        modes = []
        for qi in range(num_modes_j):
            if qi == 0:
                dur = base_dur
                dreq = dict(base_dreq)
                preq = dict(base_preq)
                mode_cost = 0.0
            else:
                # Vary duration: +/- 30%, minimum 1
                dur = max(1, base_dur + random.randint(-max(base_dur // 3, 1), max(base_dur // 3, 1)))
                # Vary resource demands
                dreq = {}
                for r, v in base_dreq.items():
                    dreq[r] = max(1, v + random.choice([-1, 0, 1]))
                # Vary part demands
                preq = {}
                for m, v in base_preq.items():
                    preq[m] = max(1, v + random.choice([-1, 0, 1]))
                # Faster modes cost more
                mode_cost = round(random.uniform(0.0, 5.0), 1)
            modes.append(Mode(dur, dreq, preq, mode_cost))

        activities[tid] = Activity(tid, modes)

    # Random DAG: only add edges i->j where i < j (topological order by index)
    precedence = []
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if random.random() < precedence_density:
                precedence.append((task_ids[i], task_ids[j]))

    renewable_capacity = {r: random.randint(3, 6) for r in resource_ids}
    lead_time = {m: random.randint(1, max(horizon // 5, 2)) for m in part_ids}
    init_inventory = {m: random.randint(0, 3) for m in part_ids}
    unit_cost = {m: round(random.uniform(1.0, 10.0), 1) for m in part_ids}
    setup_cost = {m: round(random.uniform(1.0, 5.0), 1) for m in part_ids}
    holding_cost = {m: round(random.uniform(0.1, 1.0), 2) for m in part_ids}

    return Instance(
        activities=activities,
        precedence=precedence,
        renewable_capacity=renewable_capacity,
        parts=part_ids,
        lead_time=lead_time,
        init_inventory=init_inventory,
        horizon=horizon,
        unit_cost=unit_cost,
        setup_cost=setup_cost,
        holding_cost=holding_cost,
    )


def generate_training_set(
    num_instances: int = 50,
    num_tasks: int = 10,
    num_resources: int = 2,
    num_parts: int = 3,
    horizon: int = 50,
    max_modes: int = 1,
    base_seed: int = 42,
) -> list[Instance]:
    return [
        generate_random_instance(
            num_tasks=num_tasks,
            num_resources=num_resources,
            num_parts=num_parts,
            horizon=horizon,
            max_modes=max_modes,
            seed=base_seed + i,
        )
        for i in range(num_instances)
    ]
