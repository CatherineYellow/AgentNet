#!/bin/bash
# DICE Phase 6: structure comparison — field (mean-field, top-K) vs sparse (random-K) vs graph (all).
# 3 seeds for denoising. No Byzantine.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=30
RES=~/struct_results.csv
echo "mode,N,seed,train_acc,agents_in_prompt,seconds" > "$RES"

run_one () {
  MODE=$1; N=$2; SEED=$3
  echo "========== mode=$MODE N=$N seed=$SEED start $(date +%H:%M:%S) =========="
  python - "$N" <<'PY'
import sys, yaml
N=int(sys.argv[1]); y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=N
b["agent_config"]=[{"id":i,"config":dac} for i in range(N)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY
  t0=$(date +%s)
  timeout 3000 env ROUTE_MODE=$MODE FIELD_K=4 ROBUST=0 SEED=$SEED \
    python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities \
    > ~/struct_${MODE}_N${N}_s${SEED}.log 2>&1
  rc=$?; t1=$(date +%s)
  TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/struct_${MODE}_N${N}_s${SEED}.log | tail -1 | grep -oE "[0-9.]+$")
  NB=$(grep -oE "Agent [0-9]+:" ~/struct_${MODE}_N${N}_s${SEED}.log | sort -u | wc -l)
  [ "$rc" -eq 124 ] && TR="TIMEOUT"
  [ -z "$TR" ] && TR="FAIL(rc=$rc)"
  echo "$MODE,$N,$SEED,$TR,$NB,$((t1-t0))" >> "$RES"
  echo "mode=$MODE N=$N seed=$SEED acc=$TR agents=$NB rc=$rc time=$((t1-t0))s"
}

for SEED in 1 2 3; do
  for N in 20 100 500; do
    run_one field  $N $SEED
    run_one sparse $N $SEED
  done
  run_one graph 20 $SEED   # original baseline (only where it fits the context)
done
echo "=== STRUCT DONE ==="
cat "$RES"
