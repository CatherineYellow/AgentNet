#!/bin/bash
# DICE Phase 5: does reputation-based ROBUST routing defeat the capability-misreport collapse?
# Scenario = BYZ_INFLATE=1 (Byzantine lie about ability). Compare naive vs robust across phi. field mode, N=20.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=40
N=20
RES=~/robust_results.csv
echo "defense,byz_frac,train_acc,seconds" > "$RES"

python - <<'PY'
import yaml; y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=20
b["agent_config"]=[{"id":i,"config":dac} for i in range(20)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY

for DEF in naive robust; do
  R=0; [ "$DEF" = robust ] && R=1
  for PHI in 0 0.1 0.2 0.33 0.5; do
    echo "========== defense=$DEF phi=$PHI start $(date +%H:%M:%S) =========="
    t0=$(date +%s)
    timeout 3000 env ROUTE_MODE=field FIELD_K=4 BYZANTINE_FRAC=$PHI TOTAL_AGENTS=$N BYZ_INFLATE=1 ROBUST=$R SEED=1 \
      python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities \
      > ~/robust_${DEF}_p${PHI}.log 2>&1
    rc=$?; t1=$(date +%s)
    TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/robust_${DEF}_p${PHI}.log | tail -1 | grep -oE "[0-9.]+$")
    [ "$rc" -eq 124 ] && TR="TIMEOUT"
    [ -z "$TR" ] && TR="FAIL(rc=$rc)"
    echo "$DEF,$PHI,$TR,$((t1-t0))" >> "$RES"
    echo "defense=$DEF phi=$PHI acc=$TR rc=$rc time=$((t1-t0))s"
  done
done
echo "=== ROBUST DONE ==="
cat "$RES"
