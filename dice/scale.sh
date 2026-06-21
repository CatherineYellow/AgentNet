#!/bin/bash
# DICE Phase 3: scale N up. Show R2(field) keeps running/flat while R1(graph) prompt explodes at large N.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=20
RES=~/scale_results.csv
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
  timeout 2400 env ROUTE_MODE=$MODE FIELD_K=4 python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities > ~/scale_${MODE}_N${N}.log 2>&1
  rc=$?
  t1=$(date +%s)
  TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/scale_${MODE}_N${N}.log | tail -1 | grep -oE "[0-9.]+$")
  NB=$(grep -oE "Agent [0-9]+:" ~/scale_${MODE}_N${N}.log | sort -u | wc -l)
  [ "$rc" -eq 124 ] && TR="TIMEOUT"
  [ -z "$TR" ] && TR="FAIL(rc=$rc)"
  echo "$MODE,$N,$TR,$NB,$((t1-t0))" >> "$RES"
  echo "mode=$MODE N=$N acc=$TR agents_in_prompt=$NB rc=$rc time=$((t1-t0))s"
}

for N in 50 100 200; do run_one field $N; done
for N in 50 100;    do run_one graph $N; done
echo "=== SCALE DONE ==="
cat "$RES"
