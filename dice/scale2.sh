#!/bin/bash
# DICE Phase 3b: push R2 (field) to DICE Phase-1 scale (N=500) and beyond (1000).
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=20
RES=~/scale2_results.csv
echo "mode,N,train_acc,max_agents_in_prompt,seconds" > "$RES"

run_one () {
  MODE=$1; N=$2
  echo "========== mode=$MODE N=$N start $(date +%H:%M:%S) =========="
  python - "$N" <<'PY'
import sys, yaml
N=int(sys.argv[1]); y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=N
b["agent_config"]=[{"id":i,"config":dac} for i in range(N)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY
  t0=$(date +%s)
  timeout 5400 env ROUTE_MODE=$MODE FIELD_K=4 python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities > ~/scale2_${MODE}_N${N}.log 2>&1
  rc=$?
  t1=$(date +%s)
  TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/scale2_${MODE}_N${N}.log | tail -1 | grep -oE "[0-9.]+$")
  NB=$(grep -oE "Agent [0-9]+:" ~/scale2_${MODE}_N${N}.log | sort -u | wc -l)
  [ "$rc" -eq 124 ] && TR="TIMEOUT"
  [ -z "$TR" ] && TR="FAIL(rc=$rc)"
  echo "$MODE,$N,$TR,$NB,$((t1-t0))" >> "$RES"
  echo "mode=$MODE N=$N acc=$TR agents_in_prompt=$NB rc=$rc time=$((t1-t0))s"
}

for N in 500 1000; do run_one field $N; done
echo "=== SCALE2 DONE ==="
cat "$RES"
