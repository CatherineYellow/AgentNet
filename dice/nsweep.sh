#!/bin/bash
# DICE Phase 0: N-scaling sweep of (real-graph) AgentNet on a small fixed BBH set.
set -u
cd /home/huang.6330/huangjj/AgentNet/AgentNet_Code
source ~/dice-env/.venv/bin/activate
export HF_HOME=/home/huang.6330/hf_cache CLIENT_GPU=6
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SMOKE_TASKS=10
RES=~/nsweep_results.csv
echo "N,train_acc" > "$RES"
for N in 3 5 7 9; do
  echo "========== N=$N start $(date +%H:%M:%S) =========="
  python - "$N" <<'PY'
import sys, yaml
N=int(sys.argv[1]); y="config/experiment/bigbenchhard_new_abilities.yaml"
b=yaml.safe_load(open(y)); dac=b["default_agent_config"]
b["experiment_config"]["agent_num"]=N
b["agent_config"]=[{"id":i,"config":dac} for i in range(N)]
yaml.safe_dump(b, open(y,"w"), sort_keys=False); print("yaml set N=",N)
PY
  python run_bigbenchhard_train_test.py --experiment_name bigbenchhard_new_abilities > ~/nsweep_N${N}.log 2>&1
  TR=$(grep -oE "Train Dataset Is [0-9.]+" ~/nsweep_N${N}.log | tail -1 | grep -oE "[0-9.]+$")
  echo "$N,$TR" >> "$RES"
  echo "N=$N train_acc=$TR done $(date +%H:%M:%S)"
done
echo "=== NSWEEP DONE ==="
cat "$RES"
