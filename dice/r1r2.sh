#!/bin/bash
# DICE Phase 2: compare R1 (graph) vs R2 (field) coordination at several N, on a fixed BBH set.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=20
RES=~/r1r2_results.csv
echo "mode,N,train_acc" > "$RES"
for MODE in graph field; do
  for N in 5 10 20; do
    echo "========== mode=$MODE N=$N start $(date +%H:%M:%S) =========="
    python - "$N" <<'PY'
import sys, yaml
N=int(sys.argv[1]); y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=N
b["agent_config"]=[{"id":i,"config":dac} for i in range(N)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY
    ROUTE_MODE=$MODE FIELD_K=4 python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities > ~/r1r2_${MODE}_N${N}.log 2>&1
    TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/r1r2_${MODE}_N${N}.log | tail -1 | grep -oE "[0-9.]+$")
    echo "$MODE,$N,$TR" >> "$RES"
    echo "mode=$MODE N=$N train_acc=$TR done $(date +%H:%M:%S)"
  done
done
echo "=== R1R2 DONE ==="
cat "$RES"
