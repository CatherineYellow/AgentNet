"""[DICE] Node-level virtual queues for backpressure routing.
Q[agent_id][ability] = backlog of FAILED attempts (graph-computed from observed grading,
NOT self-reported -> not spoofable like `abilities`/`success_rate`).
Dimensions = whatever task_to_ability_map maps task_types to (so identical granularity to field/graph)."""
from config.setting import task_to_ability_map

_BP_QUEUES = {}   # {int(agent_id): {ability_name: int_backlog}}

def bp_reset():
    _BP_QUEUES.clear()

def _bp_q(agent_id):
    return _BP_QUEUES.setdefault(int(agent_id), {})

def bp_update(agent_id, task_type, success):
    """Tassiulas-style: success drains backlog (-1, floor 0), failure grows it (+1),
    for every ability the task required. Credited to the EXECUTING agent."""
    q = _bp_q(agent_id)
    for d in task_to_ability_map.get(task_type, []):
        q[d] = max(0, q.get(d, 0) - 1) if success else q.get(d, 0) + 1

def bp_total(agent_id, ability_names):
    """Total backlog of an agent over the abilities a task needs (0 if never touched)."""
    q = _bp_q(agent_id)
    return sum(q.get(d, 0) for d in ability_names)

def bp_stats():
    """Warm-up diagnostic: confirm queues actually populated before the held-out test."""
    cells = [v for q in _BP_QUEUES.values() for v in q.values()]
    nz = sum(1 for v in cells if v > 0)
    return {"agents_touched": len(_BP_QUEUES), "nonzero_cells": nz,
            "total_backlog": sum(cells), "max_backlog": max(cells) if cells else 0}
