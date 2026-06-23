"""DICE corrections per codex review. Run from AgentNet_Code/.
(A) SMOKE_TASKS -> seeded RANDOM task subset (covers task-selection variance).
(B) Byzantine ids -> seeded RANDOM set (not always low ids).
(C) field prefilter by reputation under ROBUST + random tie-break (clean robust).
(D) forwarding random tie-break.
"""
import py_compile

# (A) entry: random task subset
e = "run_bigbenchhard_train_test.py"; s = open(e).read()
old_a = '''    if _st>0:
        train_dataset=train_dataset[:_st]; test_dataset=test_dataset[:_st]
        print(f"[SMOKE] limited to {_st} train/{_st} test tasks")'''
new_a = '''    if _st>0:
        import random as _rsub
        _sd=int(_os.getenv("SEED","0") or 0)
        _itr=sorted(_rsub.Random(_sd).sample(range(len(train_dataset)), min(_st,len(train_dataset))))
        _ite=sorted(_rsub.Random(1000+_sd).sample(range(len(test_dataset)), min(_st,len(test_dataset))))
        train_dataset=[train_dataset[i] for i in _itr]; test_dataset=[test_dataset[i] for i in _ite]
        print(f"[SMOKE] random subset {_st} train/{_st} test (seed {_sd})")'''
assert old_a in s, "(A) anchor not found"
open(e, "w").write(s.replace(old_a, new_a, 1)); print("[A] ok")

# (B) agent.py: random Byzantine set
f = "src/agent.py"; s = open(f).read()
old_b = '''def _dice_is_byzantine(agent_id):
    import os
    frac = float(os.getenv("BYZANTINE_FRAC", "0") or 0)
    N = int(os.getenv("TOTAL_AGENTS", "0") or 0)
    if frac <= 0 or N <= 0:
        return False
    return int(agent_id) < int(frac * N)'''
new_b = '''_DICE_BYZ_SET = None
def _dice_byz_set():
    global _DICE_BYZ_SET
    if _DICE_BYZ_SET is None:
        import os, random as _r
        frac = float(os.getenv("BYZANTINE_FRAC", "0") or 0)
        N = int(os.getenv("TOTAL_AGENTS", "0") or 0)
        seed = int(os.getenv("SEED", "0") or 0)
        k = int(frac * N)
        _DICE_BYZ_SET = set(_r.Random(seed).sample(range(N), k)) if (frac > 0 and N > 0 and k > 0) else set()
    return _DICE_BYZ_SET
def _dice_is_byzantine(agent_id):
    return int(agent_id) in _dice_byz_set()'''
assert old_b in s, "(B) anchor not found"
s = s.replace(old_b, new_b, 1)

# (D) forwarding random tie-break (in find_best_alternative_agent)
old_d = '''        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]'''
new_d = '''        if candidates:
            import random as _rtb
            _rtb.shuffle(candidates)
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]'''
assert old_d in s, "(D) anchor not found"
open(f, "w").write(s.replace(old_d, new_d, 1)); print("[B,D] ok")

# (C) agentgraph.py: reputation field-prefilter under ROBUST + random tie-break
g = "src/agentgraph.py"; s = open(g).read()
old_c = '''        if mode in ("field", "sparse") and ids:
            K = int(os.getenv("FIELD_K", "4"))
            if mode == "field":
                names = task_to_ability_map.get(task.task_type, [])
                def _ab(nid):
                    ab = self.agents[nid].get_self_info()["abilities"]
                    return (sum(ab[n] for n in names) / len(names)) if names else 0.0
                ids = sorted(ids, key=_ab, reverse=True)[:K]      # mean-field: top-K by task-type ability
            else:
                ids = _random.sample(ids, min(K, len(ids)))       # sparse: K random neighbors (local graph)'''
new_c = '''        if mode in ("field", "sparse") and ids:
            K = int(os.getenv("FIELD_K", "4"))
            _random.shuffle(ids)                                  # random tie-break
            if mode == "field":
                robust = os.getenv("ROBUST", "0") == "1"
                names = task_to_ability_map.get(task.task_type, [])
                def _key(nid):
                    info = self.agents[nid].get_self_info()
                    if robust:
                        return info["success_rate"].get(task.task_type, 0.0)   # clean robust: prefilter by reputation
                    ab = info["abilities"]
                    return (sum(ab[n] for n in names) / len(names)) if names else 0.0
                ids = sorted(ids, key=_key, reverse=True)[:K]
            else:
                ids = _random.sample(ids, min(K, len(ids)))'''
assert old_c in s, "(C) anchor not found"
open(g, "w").write(s.replace(old_c, new_c, 1)); print("[C] ok")

for p in (e, f, g):
    py_compile.compile(p, doraise=True)
print("[ok] all compile")
