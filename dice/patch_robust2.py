"""DICE Phase 7: extend robust routing to FORWARDING too (find_best_alternative_agent).
Under ROBUST=1, forwarding ranks by demonstrated success (reputation), ignoring self-claimed ability.
Combined with the already-robust select_an_agent => full reputation-based routing. Run from AgentNet_Code/."""
f = "src/agent.py"
s = open(f).read()
new = '''    def find_best_alternative_agent(self, task_type, neighbors_info):
        import os
        robust = os.getenv("ROBUST", "0") == "1"
        candidates = []
        for agent_id, info in neighbors_info.items():
            if agent_id == self.agent_id:
                continue
            if not (info['is_incoming'] or info['is_outgoing']):
                continue
            ability = get_average_abilities_from_task_type(task_type, info['abilities'])
            load = info['current_load']
            success_rate = info['success_rate'][task_type]
            if robust:
                # [DICE] reputation-based forwarding: ignore self-claimed ability (defeats inflation)
                if load < 3:
                    candidates.append((success_rate * 0.8 + (1 - load / 3) * 0.2, agent_id))
            else:
                score = (
                    ability * 0.4 +
                    (1 - load / 3) * 0.3 +
                    success_rate * 0.2 +
                    (1.0 if info['is_outgoing'] else 0.5) * 0.1
                )
                if ability >= 0.5 and load < 3:
                    candidates.append((score, agent_id))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        else:
            candidates = [agent_id for agent_id in neighbors_info.keys() if agent_id != self.agent_id]
            if not candidates:
                return None
            return random.choice(candidates)

'''
i = s.index("    def find_best_alternative_agent(self, task_type, neighbors_info):")
j = s.index("    def get_newest_experiences(self, task_type, k=50):")
open(f, "w").write(s[:i] + new + s[j:])
print("[ok] robust forwarding installed")
import py_compile
py_compile.compile(f, doraise=True)
print("[ok] compiles")
