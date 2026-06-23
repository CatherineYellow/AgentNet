#!/bin/bash
# DICE Phase 9 (#4): (1) K ablation; (2) delayed-onset Byzantine vs reputation routing.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=30

setN(){ python - "$1" <<'PY'
import sys, yaml
N=int(sys.argv[1]); y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=N
b["agent_config"]=[{"id":i,"config":dac} for i in range(N)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False)
PY
}
gettest(){ grep -oE "Test Dataset Is [0-9.]+" "$1" | tail -1 | grep -oE "[0-9.]+$"; }

# ===== Part 1: K ablation (field, N=100, no Byzantine) =====
RES1=~/kabl_results.csv; echo "K,seed,test_acc,seconds" > "$RES1"
setN 100
for SEED in 1 2; do for K in 1 2 4 8; do
  echo "== Kabl K=$K seed=$SEED $(date +%H:%M:%S) =="
  t0=$(date +%s)
  timeout 3000 env ROUTE_MODE=field FIELD_K=$K SEED=$SEED \
    python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities > ~/kabl_K${K}_s${SEED}.log 2>&1
  t1=$(date +%s); echo "$K,$SEED,$(gettest ~/kabl_K${K}_s${SEED}.log),$((t1-t0))" >> "$RES1"
done; done

# ===== Part 2: delayed-onset Byzantine (field, N=20, inflation, phi=0.33) =====
RES2=~/delayed_results.csv; echo "defense,onset,seed,test_acc,seconds" > "$RES2"
setN 20
for SEED in 1 2; do for DEF in naive robust; do R=0; [ "$DEF" = robust ] && R=1
  for ONSET in immediate delayed; do DELAY=0; [ "$ONSET" = delayed ] && DELAY=15
    echo "== delayed def=$DEF onset=$ONSET seed=$SEED $(date +%H:%M:%S) =="
    t0=$(date +%s)
    timeout 3000 env ROUTE_MODE=field FIELD_K=4 BYZANTINE_FRAC=0.33 TOTAL_AGENTS=20 BYZ_INFLATE=1 BYZ_DELAY=$DELAY ROBUST=$R SEED=$SEED \
      python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities > ~/delayed_${DEF}_${ONSET}_s${SEED}.log 2>&1
    t1=$(date +%s); echo "$DEF,$ONSET,$SEED,$(gettest ~/delayed_${DEF}_${ONSET}_s${SEED}.log),$((t1-t0))" >> "$RES2"
  done
done; done

echo "=== ABL DONE ==="; echo "--- K ablation ---"; cat "$RES1"; echo "--- delayed-onset ---"; cat "$RES2"
