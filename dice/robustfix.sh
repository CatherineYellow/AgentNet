#!/bin/bash
# DICE corrected robustness: clean robust + fair random Byzantine + random task subset per seed.
# naive vs robust x phi x 3 seeds, inflation attack, N=20. Reports BOTH train and held-out TEST acc.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=30
N=20
RES=~/robustfix_results.csv
echo "defense,byz_frac,seed,train_acc,test_acc,seconds" > "$RES"

python - <<'PY'
import yaml; y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=20
b["agent_config"]=[{"id":i,"config":dac} for i in range(20)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY

for SEED in 1 2 3; do
  for DEF in naive robust; do
    R=0; [ "$DEF" = robust ] && R=1
    for PHI in 0 0.2 0.33 0.5; do
      echo "========== def=$DEF phi=$PHI seed=$SEED start $(date +%H:%M:%S) =========="
      t0=$(date +%s)
      timeout 3000 env ROUTE_MODE=field FIELD_K=4 BYZANTINE_FRAC=$PHI TOTAL_AGENTS=$N BYZ_INFLATE=1 ROBUST=$R SEED=$SEED \
        python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities \
        > ~/robustfix_${DEF}_p${PHI}_s${SEED}.log 2>&1
      rc=$?; t1=$(date +%s)
      TRN=$(grep -oE "Train Dataset Is [0-9.]+" ~/robustfix_${DEF}_p${PHI}_s${SEED}.log | tail -1 | grep -oE "[0-9.]+$")
      TST=$(grep -oE "Test Dataset Is [0-9.]+" ~/robustfix_${DEF}_p${PHI}_s${SEED}.log | tail -1 | grep -oE "[0-9.]+$")
      [ "$rc" -eq 124 ] && TST="TIMEOUT"
      echo "$DEF,$PHI,$SEED,${TRN:-NA},${TST:-NA},$((t1-t0))" >> "$RES"
      echo "def=$DEF phi=$PHI seed=$SEED train=${TRN:-NA} test=${TST:-NA} rc=$rc time=$((t1-t0))s"
    done
  done
done
echo "=== ROBUSTFIX DONE ==="
cat "$RES"
