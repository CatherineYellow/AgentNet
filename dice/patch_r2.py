"""DICE Phase 1: add a route-to-field (R2) toggle to AgentNet's collect_neighbors_info.
ROUTE_MODE=graph (default) -> original AgentNet (all neighbors, R1).
ROUTE_MODE=field           -> mean-field: keep only top-K neighbors by task-type ability (bounded prompt/interaction).
Run from AgentNet_Code/."""
f = "src/agentgraph.py"
s = open(f).read()
new = '''    def collect_neighbors_info(self, agent_id, task):
        import os
        outcoming_neighbors_id  = self.agent_neighbor_dict[agent_id]["outcoming_agent_id"]
        neighbors_info = {}
        for neighbor_id in outcoming_neighbors_id:
            if self.edge_weight[agent_id][neighbor_id] <= 0.3:
                continue
            neighbor_agent = self.agents[neighbor_id]
            neighbor_agent_info = neighbor_agent.get_self_info()
            neighbor_agent_info["processed_tasks"] = neighbor_agent.get_relevant_experence(task)
            neighbor_agent_info["success_rate"] = neighbor_agent_info["success_rate"]
            neighbor_agent_info["task_type_success_rate"] = neighbor_agent_info["success_rate"][task.task_type]
            neighbor_agent_info["is_incoming"] = False
            neighbor_agent_info["is_outgoing"] = True
            neighbors_info[neighbor_id]= neighbor_agent_info
        # [DICE] route-to-field (R2): keep only top-K neighbors by task-type ability (mean-field neighborhood)
        if os.getenv("ROUTE_MODE", "graph") == "field" and neighbors_info:
            K = int(os.getenv("FIELD_K", "4"))
            names = task_to_ability_map.get(task.task_type, [])
            def _ab(info):
                return (sum(info["abilities"][n] for n in names) / len(names)) if names else 0.0
            top = sorted(neighbors_info.items(), key=lambda kv: _ab(kv[1]), reverse=True)[:K]
            neighbors_info = dict(top)
        return neighbors_info

'''
i = s.index("    def collect_neighbors_info(self, agent_id, task):")
j = s.index("    def update_edge_weight(")
open(f, "w").write(s[:i] + new + s[j:])
print("[ok] patched collect_neighbors_info (R2 toggle)")
import py_compile
py_compile.compile(f, doraise=True)
print("[ok] src/agentgraph.py compiles")
