"""DICE Phase 5: (1) seed control via env SEED; (2) ROBUST routing — rank initial agent by
DEMONSTRATED success (reputation) instead of self-claimed ability, defeating capability-misreport.
ROBUST=1 enables it (default off = original). Run from AgentNet_Code/."""

# (1) seed control in the entry
e = "run_bigbenchhard_train_test.py"
s = open(e).read()
anchor = 'os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CLIENT_GPU", "6")'
if "[DICE] seed" not in s and anchor in s:
    inject = anchor + '''
import random as _random, numpy as _np  # [DICE] seed control
_seed = int(os.getenv("SEED", "0") or 0)
if _seed:
    _random.seed(_seed); _np.random.seed(_seed)'''
    open(e, "w").write(s.replace(anchor, inject, 1))
    print("[ok] seed control added to entry")

# (2) robust select_an_agent in agentgraph.py
g = "src/agentgraph.py"
s = open(g).read()
new_sel = '''    def select_an_agent(self, task_type):
        import os as _os
        robust = _os.getenv("ROBUST", "0") == "1"
        neighbors_info = {}
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            agent_info = agent.get_self_info()
            success_rate = agent_info["success_rate"]
            abilities = agent_info["abilities"]
            if task_type not in success_rate:
                continue
            ability_names = task_to_ability_map[task_type]
            total_value, ability_num = 0, 0
            for name in ability_names:
                ability_num += 1
                total_value += abilities[name]
            average_ability_value = total_value / ability_num
            # [DICE] robust: rank by demonstrated success (reputation), not self-claimed ability
            rank_value = success_rate.get(task_type, 0.0) if robust else average_ability_value
            neighbors_info[agent_id] = {
                "agent_info": agent_info,
                "average_ability_value": average_ability_value,
                "rank_value": rank_value,
            }
        if not neighbors_info:
            return self.sample_an_agent()
        max_value = max(info["rank_value"] for info in neighbors_info.values())
        best_agents = [aid for aid, info in neighbors_info.items() if info["rank_value"] == max_value]
        return random.choice(best_agents)

'''
i = s.index("    def select_an_agent(self, task_type):")
j = s.index("    def collect_neighbors_info(self, agent_id, task):")
open(g, "w").write(s[:i] + new_sel + s[j:])
print("[ok] robust select_an_agent installed")
import py_compile
py_compile.compile(e, doraise=True); py_compile.compile(g, doraise=True)
print("[ok] both compile")
