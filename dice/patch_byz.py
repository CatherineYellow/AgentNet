"""DICE Phase 4: inject Byzantine agents. Agents with id < BYZANTINE_FRAC*TOTAL_AGENTS
(a) report inflated abilities (attract routing) and (b) emit a wrong executor answer.
Controlled by env BYZANTINE_FRAC + TOTAL_AGENTS. Run from AgentNet_Code/."""
f = "src/agent.py"
s = open(f).read()

# 1) helper after the module logger
if "_dice_is_byzantine" not in s:
    anchor = "logger = logging.getLogger(__name__)"
    helper = anchor + '''


def _dice_is_byzantine(agent_id):
    import os
    frac = float(os.getenv("BYZANTINE_FRAC", "0") or 0)
    N = int(os.getenv("TOTAL_AGENTS", "0") or 0)
    if frac <= 0 or N <= 0:
        return False
    return int(agent_id) < int(frac * N)'''
    s = s.replace(anchor, helper, 1)
    print("[ok] inserted _dice_is_byzantine helper")

# 2) corrupt executor result (inject after the result assignment line)
key = 'result = response_dict.get("RESULT", response)'
if "[DICE] Byzantine corrupt" not in s:
    idx = s.index(key)
    eol = s.index("\n", idx)
    inject = '\n        if _dice_is_byzantine(self.agent_id):\n            result = "(no valid answer)"  # [DICE] Byzantine corrupt output'
    s = s[:eol] + inject + s[eol:]
    print("[ok] injected Byzantine result corruption")

# 3) inflate Byzantine abilities in get_self_info (index-based, whitespace-robust)
new_gsi = '''    def get_self_info(self):
        abilities = self.abilities
        if _dice_is_byzantine(self.agent_id):
            abilities = {k: 1.0 for k in self.abilities}  # [DICE] Byzantine inflate to attract routing
        self_info = {
            "agent_id": self.agent_id,
            "current_load": self.current_load,
            "success_rate": self.success_rate,
            "abilities": abilities,
        }
        return self_info'''
if "[DICE] Byzantine inflate" not in s:
    i = s.index("    def get_self_info(self):")
    j = s.index("return self_info", i) + len("return self_info")
    s = s[:i] + new_gsi + s[j:]
    print("[ok] patched get_self_info (Byzantine ability inflation)")

open(f, "w").write(s)
import py_compile
py_compile.compile(f, doraise=True)
print("[ok] src/agent.py compiles")
