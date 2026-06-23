"""DICE Phase 9 (#4): delayed-onset Byzantine — behave honestly for the first BYZ_DELAY tasks
(build reputation), then turn malicious. Tests whether reputation routing can be gamed.
BYZ_DELAY = task index (in the train stream) after which Byzantine activates; 0 = immediate (default).
Run from AgentNet_Code/."""
import py_compile

f = "src/agent.py"; s = open(f).read()
anchor = '''def _dice_is_byzantine(agent_id):
    return int(agent_id) in _dice_byz_set()'''
assert anchor in s and "_dice_byz_now" not in s, "anchor/idempotency"
s = s.replace(anchor, anchor + '''
_DICE_TASK_IDX = [0]
def _dice_bump_task():
    _DICE_TASK_IDX[0] += 1
def _dice_byz_now(agent_id):
    import os
    return _dice_is_byzantine(agent_id) and _DICE_TASK_IDX[0] >= int(os.getenv("BYZ_DELAY", "0") or 0)''', 1)

# gate executor corruption + ability inflation on the delay (use _dice_byz_now)
s = s.replace('        if _dice_is_byzantine(self.agent_id):\n            result = "(no valid answer)"  # [DICE] Byzantine corrupt output',
              '        if _dice_byz_now(self.agent_id):\n            result = "(no valid answer)"  # [DICE] Byzantine corrupt output (delay-aware)')
s = s.replace('if _dice_is_byzantine(self.agent_id) and __import__("os").getenv("BYZ_INFLATE", "0") == "1":',
              'if _dice_byz_now(self.agent_id) and __import__("os").getenv("BYZ_INFLATE", "0") == "1":')
open(f, "w").write(s); print("[agent] delayed-onset (_dice_byz_now) installed")

g = "src/experiment.py"; s = open(g).read()
old = '            self.update_agent_graph(final_task_chain, result)'
assert old in s and "_dice_bump_task" not in s, "fit anchor/idempotency"
s = s.replace(old, old + '\n            __import__("src.agent", fromlist=["x"])._dice_bump_task()', 1)
open(g, "w").write(s); print("[experiment] per-task counter bump added")

for p in (f, g):
    py_compile.compile(p, doraise=True)
print("[ok] compile")
