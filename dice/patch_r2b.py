"""DICE Phase 3b: make route-to-field O(K) per step (cap by ability BEFORE the expensive
experience retrieval), so R2 stays cheap at large N. Run from AgentNet_Code/."""
f = "src/agentgraph.py"
s = open(f).read()
new = '''    def collect_neighbors_info(self, agent_id, task):
        import os
        outcoming_neighbors_id  = self.agent_neighbor_dict[agent_id]["outcoming_agent_id"]
        ids = [nid for nid in outcoming_neighbors_id if self.edge_weight[agent_id][nid] > 0.3]
        # [DICE] route-to-field (R2): cheap ability-rank, keep top-K, THEN build full info only for those K
        if os.getenv("ROUTE_MODE", "graph") == "field" and ids:
            K = int(os.getenv("FIELD_K", "4"))
            names = task_to_ability_map.get(task.task_type, [])
            def _ab(nid):
                ab = self.agents[nid].get_self_info()["abilities"]
                return (sum(ab[n] for n in names) / len(names)) if names else 0.0
            ids = sorted(ids, key=_ab, reverse=True)[:K]
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
open(f, "w").write(s[:i] + new + s[j:])
print("[ok] optimized collect_neighbors_info (cap-before-experiences)")
import py_compile
py_compile.compile(f, doraise=True)
print("[ok] compiles")
