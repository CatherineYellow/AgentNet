"""DICE Phase 10: NODE-based backpressure routing (advisor's edge->node change).

Replaces routing-by-claimed-ability with routing-by-graph-computed-backlog:
each agent has a per-ability virtual queue Q^d (graph-maintained, NOT self-reported):
  on a task needing abilities {d}, executor's Q^d -= 1 on success, += 1 on failure (floor 0).
Routing (ROUTE_MODE=backpressure) bounds the candidate set to the K LOWEST-backlog neighbors
(parallel to ROUTE_MODE=field which keeps K HIGHEST-ability), so the LLM router can only pick
among low-backlog agents -> backpressure genuinely drives routing (not just the fallback).
select_an_agent dispatches by min total backlog; find_best_alternative_agent (fallback) by
the positive backlog differential [Q_self - Q_cand]^+.

Key property: routing ignores self-claimed `abilities` entirely -> a Byzantine agent CANNOT
inflate its way to the front; failing only GROWS its backlog -> it is routed away from.

Apply LAST (after robust/robust2/fixes/phase4). Run from AgentNet_Code/. Idempotent.
"""
import py_compile

# ----------------------------------------------------------------------------
# (0) NEW module: src/bp_queue.py — the node-level virtual queues
# ----------------------------------------------------------------------------
BP_QUEUE_SRC = '''"""[DICE] Node-level virtual queues for backpressure routing.
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
'''
with open("src/bp_queue.py", "w") as fh:
    fh.write(BP_QUEUE_SRC)
print("[bp_queue] src/bp_queue.py written")

# ----------------------------------------------------------------------------
# (1) src/experiment.py — queue update hook (next to update_abilities, executor steps)
# ----------------------------------------------------------------------------
g = "src/experiment.py"; s = open(g).read()
assert "bp_update" not in s, "[experiment] already patched"
marker = "agent.update_abilities("
assert s.count(marker) == 1, f"[experiment] expected 1 update_abilities, got {s.count(marker)}"
idx = s.index(marker)
end = s.index("success)", idx) + len("success)")   # end of the update_abilities(...) call
inject = ("\n                from src.bp_queue import bp_update  # [DICE] backpressure queue update"
          "\n                bp_update(single_task_history.current_agent_id, "
          "single_task_history.task.task_type, success)")
s = s[:end] + inject + s[end:]
open(g, "w").write(s); print("[experiment] bp_update hook added in update_agent_graph")

# ----------------------------------------------------------------------------
# (2) src/agentgraph.py — select_an_agent (min-queue dispatch) + collect_neighbors_info (K min-backlog)
# ----------------------------------------------------------------------------
a = "src/agentgraph.py"; s = open(a).read()
assert "bp_total" not in s, "[agentgraph] already patched"

# (2a) select_an_agent: include all agents under backpressure (don't skip on missing success_rate)
old_cont = ("            if task_type not in success_rate:\n"
            "                continue\n")
assert s.count(old_cont) == 1, "[agentgraph] continue anchor"
new_cont = ('            if task_type not in success_rate and '
            '__import__("os").getenv("ROUTE_MODE","graph") != "backpressure":\n'
            "                continue\n")
s = s.replace(old_cont, new_cont, 1)

# (2b) select_an_agent: rank by -backlog under backpressure
old_rank = ("            # [DICE] robust: rank by demonstrated success (reputation), not self-claimed ability\n"
            "            rank_value = success_rate.get(task_type, 0.0) if robust else average_ability_value\n")
assert s.count(old_rank) == 1, "[agentgraph] rank_value anchor"
new_rank = ('            # [DICE] robust: rank by reputation; backpressure: rank by -backlog (min queue)\n'
            '            if __import__("os").getenv("ROUTE_MODE","graph") == "backpressure":\n'
            "                from src.bp_queue import bp_total\n"
            "                rank_value = -bp_total(agent_id, ability_names)\n"
            "            else:\n"
            "                rank_value = success_rate.get(task_type, 0.0) if robust else average_ability_value\n")
s = s.replace(old_rank, new_rank, 1)

# (2c) collect_neighbors_info: backpressure branch keeps K LOWEST-backlog neighbors
old_filt = "        if mode in (\"field\", \"sparse\") and ids:\n"
assert s.count(old_filt) == 1, "[agentgraph] collect filter anchor"
new_filt = ('        if mode == "backpressure" and ids:\n'
            '            K = int(os.getenv("FIELD_K", "4"))\n'
            "            from src.bp_queue import bp_total\n"
            "            _bp_names = task_to_ability_map.get(task.task_type, [])\n"
            "            _random.shuffle(ids)                                  # random tie-break\n"
            "            ids = sorted(ids, key=lambda nid: bp_total(nid, _bp_names))[:K]   # K LOWEST backlog\n"
            '        elif mode in ("field", "sparse") and ids:\n')
s = s.replace(old_filt, new_filt, 1)
open(a, "w").write(s); print("[agentgraph] select_an_agent + collect_neighbors_info backpressure branches added")

# ----------------------------------------------------------------------------
# (3) src/agent.py — find_best_alternative_agent fallback by backlog differential
# ----------------------------------------------------------------------------
b = "src/agent.py"; s = open(b).read()
assert "_bp_mode" not in s, "[agent] already patched"

old_head = ('        robust = os.getenv("ROBUST", "0") == "1"\n'
            "        candidates = []\n")
assert s.count(old_head) == 1, "[agent] head anchor"
new_head = ('        robust = os.getenv("ROBUST", "0") == "1"\n'
            '        _bp_mode = os.getenv("ROUTE_MODE", "graph") == "backpressure"\n'
            "        if _bp_mode:\n"
            "            from src.bp_queue import bp_total\n"
            "            _bp_names = task_to_ability_map.get(task_type, [])\n"
            "            _bp_self = bp_total(self.agent_id, _bp_names)\n"
            "        candidates = []\n")
s = s.replace(old_head, new_head, 1)

old_branch = ("            if robust:\n"
              "                # [DICE] reputation-based forwarding: ignore self-claimed ability (defeats inflation)\n"
              "                if load < 3:\n"
              "                    candidates.append((success_rate * 0.8 + (1 - load / 3) * 0.2, agent_id))\n"
              "            else:\n")
assert s.count(old_branch) == 1, "[agent] branch anchor"
new_branch = ("            if _bp_mode:\n"
              "                # [DICE] backpressure: prefer neighbor with lowest backlog (max positive differential)\n"
              "                if load < 3:\n"
              "                    candidates.append((max(0, _bp_self - bp_total(agent_id, _bp_names)), agent_id))\n"
              "            elif robust:\n"
              "                # [DICE] reputation-based forwarding: ignore self-claimed ability (defeats inflation)\n"
              "                if load < 3:\n"
              "                    candidates.append((success_rate * 0.8 + (1 - load / 3) * 0.2, agent_id))\n"
              "            else:\n")
s = s.replace(old_branch, new_branch, 1)
open(b, "w").write(s); print("[agent] find_best_alternative_agent backpressure branch added")

# ----------------------------------------------------------------------------
# (4) entry — warm-up diagnostic after fit (confirm queues populated before test)
# ----------------------------------------------------------------------------
e = "run_bigbenchhard_train_test.py"; s = open(e).read()
assert "BPSTATS" not in s, "[entry] already patched"
old_run = "    experiment.fit()\n    experiment.evaluate()\n"
assert s.count(old_run) == 1, "[entry] fit/evaluate anchor"
new_run = ("    from src.bp_queue import bp_stats  # [DICE] backpressure warm-up diagnostic\n"
           "    experiment.fit()\n"
           '    print(f"[BPSTATS] after fit: {bp_stats()}", flush=True)\n'
           "    experiment.evaluate()\n")
s = s.replace(old_run, new_run, 1)
open(e, "w").write(s); print("[entry] bp_stats diagnostic added")

# ----------------------------------------------------------------------------
for p in ("src/bp_queue.py", g, a, b, e):
    py_compile.compile(p, doraise=True)
print("[ok] all compile")
