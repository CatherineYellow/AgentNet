"""DICE Phase 6: add ROUTE_MODE=sparse (random-K neighbors = local sparse graph) alongside
field (top-K by ability = mean-field) and graph (all = original). For the structure comparison.
Run from AgentNet_Code/."""
g = "src/agentgraph.py"
s = open(g).read()
new = '''    def collect_neighbors_info(self, agent_id, task):
        import os, random as _random
        outcoming_neighbors_id  = self.agent_neighbor_dict[agent_id]["outcoming_agent_id"]
        ids = [nid for nid in outcoming_neighbors_id if self.edge_weight[agent_id][nid] > 0.3]
        mode = os.getenv("ROUTE_MODE", "graph")
        if mode in ("field", "sparse") and ids:
            K = int(os.getenv("FIELD_K", "4"))
            if mode == "field":
                names = task_to_ability_map.get(task.task_type, [])
                def _ab(nid):
                    ab = self.agents[nid].get_self_info()["abilities"]
                    return (sum(ab[n] for n in names) / len(names)) if names else 0.0
                ids = sorted(ids, key=_ab, reverse=True)[:K]      # mean-field: top-K by task-type ability
            else:
                ids = _random.sample(ids, min(K, len(ids)))       # sparse: K random neighbors (local graph)
        neighbors_info = {}
        for neighbor_id in ids:
            neighbor_agent = self.agents[neighbor_id]
            neighbor_agent_info = neighbor_agent.get_self_info()
            neighbor_agent_info["processed_tasks"] = neighbor_agent.get_relevant_experence(task)
            neighbor_agent_info["success_rate"] = neighbor_agent_info["success_rate"]
            neighbor_agent_info["task_type_success_rate"] = neighbor_agent_info["success_rate"][task.task_type]
            neighbor_agent_info["is_incoming"] = False
            neighbor_agent_info["is_outgoing"] = True
            neighbors_info[neighbor_id]= neighbor_agent_info
        return neighbors_info

'''
i = s.index("    def collect_neighbors_info(self, agent_id, task):")
j = s.index("    def update_edge_weight(")
open(g, "w").write(s[:i] + new + s[j:])
print("[ok] added ROUTE_MODE=sparse")
import py_compile
py_compile.compile(g, doraise=True)
print("[ok] compiles")
