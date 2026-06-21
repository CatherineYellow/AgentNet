# DICE experiments on AgentNet (route-to-field mean-field coordination)

This fork extends **AgentNet** (Yang et al., NeurIPS 2025) with a **route-to-field (mean-field) coordination mode**
and **adversarial-robustness experiments**, for the DARPA DICE proposal direction (decentralized AI, controlled
emergence). All runs are fully local (vLLM + Qwen2.5-7B, local BGE embeddings), no external API.

## What was added (see `dice/patch_*.py` for the exact edits)
- **`ROUTE_MODE=field`** (`patch_r2.py`, `patch_r2b.py`): in `collect_neighbors_info`, keep only the top-`FIELD_K`
  neighbors by task-type ability (a mean-field neighborhood) instead of all peers. `graph` (default) = original AgentNet.
  Makes the router prompt O(1) in N instead of O(N).
- **Byzantine injection** (`patch_byz.py`, `patch_byz_gate.py`): agents with id < `BYZANTINE_FRAC`*`TOTAL_AGENTS`
  emit wrong answers; with `BYZ_INFLATE=1` they also report inflated abilities to attract routing.
- **Robust routing** (`patch_robust.py`): `ROBUST=1` ranks the initial agent by demonstrated success
  (reputation) instead of self-claimed ability — a defense against capability-misreport. Plus `SEED` control.
- `SMOKE_TASKS` env to limit task count for fast iteration.

## Headline result — scalability (structural, robust to noise)
Centralized/all-to-all coordination (R1) hits a hard context-length wall as the population grows; mean-field
coordination (R2) does not. (`dice/scale_results.csv`, `dice/scale2_results.csv`)

| N | R1 (graph) | R2 (field) | R2 prompt (agents) | R2 time |
|---|---|---|---|---|
| 20 | 0.55 | 0.65 | 5 | ~300s |
| 50 | **crash (8192 ctx)** | 0.55 | 6 | 316s |
| 100 | **crash** | 0.70 | 5 | 371s |
| 200 | — | 0.65 | 5 | 386s |
| 500 | — | 0.55 | 5 | 344s |
| 1000 | — | 0.60 | 5 | 441s |

R1 crashes at N=50 with `BadRequestError 400: maximum context length is 8192 tokens` (the router prompt enumerates
all peers → O(N) tokens). R2 scales to N=1000 (2× the DICE Phase-1 target of 500) at constant O(1) prompt and ~flat time.

## Adversarial robustness (directional; 20–40 tasks/seed → noisy)
- **Capability-misreport collapse**: with `BYZ_INFLATE=1`, naive ability-routing collapses 0.6→0.0 at just φ=0.1.
- **Robust routing** (`ROBUST=1`, reputation-based) is designed to prevent this — see `dice/robust_results.csv`.
- Wrong-output-only Byzantine → graceful degradation (~0.45–0.60 even at φ=0.5), no cascade. (`dice/byz_results.csv`)

## How to run
1. Start vLLM (no JIT, since the box has no nvcc):
   `CUDA_VISIBLE_DEVICES=4 HF_HOME=~/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.4 --max-model-len 8192 --enforce-eager`
2. Run a sweep, e.g. `bash dice/scale.sh` (env knobs: `ROUTE_MODE`, `FIELD_K`, `SMOKE_TASKS`, `BYZANTINE_FRAC`, `BYZ_INFLATE`, `ROBUST`, `SEED`).

## Caveats
Accuracy numbers are 20–40 tasks / single seed → directional, not statistically settled. The scaling/crash result is
structural and solid. Task = BigBenchHard reasoning with Qwen2.5-7B (not yet a DICE mission/heterogeneous setting).
