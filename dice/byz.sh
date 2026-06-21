#!/bin/bash
# DICE Phase 4: Byzantine resilience. Sweep byzantine fraction phi for R1(graph) and R2(field) at N=20.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=20
N=20
RES=~/byz_results.csv
echo "mode,N,byz_frac,train_acc,seconds" > "$RES"

# fix N=20 config once
python - <<'PY'
import yaml; y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=20
b["agent_config"]=[{"id":i,"config":dac} for i in range(20)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY

for MODE in field graph; do
  for PHI in 0 0.1 0.2 0.33 0.5; do
    echo "========== mode=$MODE phi=$PHI start $(date +%H:%M:%S) =========="
    t0=$(date +%s)
    timeout 2400 env ROUTE_MODE=$MODE FIELD_K=4 BYZANTINE_FRAC=$PHI TOTAL_AGENTS=$N \
      python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities \
      > ~/byz_${MODE}_p${PHI}.log 2>&1
    rc=$?; t1=$(date +%s)
    TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/byz_${MODE}_p${PHI}.log | tail -1 | grep -oE "[0-9.]+$")
    [ "$rc" -eq 124 ] && TR="TIMEOUT"
    [ -z "$TR" ] && TR="FAIL(rc=$rc)"
    echo "$MODE,$N,$PHI,$TR,$((t1-t0))" >> "$RES"
    echo "mode=$MODE phi=$PHI acc=$TR rc=$rc time=$((t1-t0))s"
  done
done
echo "=== BYZ DONE ==="
cat "$RES"
