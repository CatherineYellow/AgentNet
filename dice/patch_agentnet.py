"""DICE: patch AgentNet to run fully locally (vLLM + local embeddings). Run from AgentNet_Code/."""
import re

# 1) src/utils.py : get_gpt_response -> local vLLM (OpenAI-compatible)
u = "src/utils.py"
s = open(u).read()
new_fn = '''def get_gpt_response(system_prompt, query_prompt):
    """[DICE patch] Call a local vLLM OpenAI-compatible server instead of OpenAI."""
    import os
    from openai import OpenAI
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    model = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
                top_p=1.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[local-llm] attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(1)


'''
i = s.index("def get_gpt_response(")
j = s.index("def get_qwen_response(")
open(u, "w").write(s[:i] + new_fn + s[j:])
print("[ok] patched", u)

# 2) entry: CUDA_VISIBLE_DEVICES "0" -> env CLIENT_GPU (default 6)
e = "run_bigbenchhard_train_test.py"
s = open(e).read()
assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in s, "CUDA line not found"
s = s.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
              'os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CLIENT_GPU", "6")')
open(e, "w").write(s)
print("[ok] patched", e)

# 3) src/pool.py : FlagEmbedding.FlagModel -> sentence-transformers (avoid dep conflict with vllm)
p = "src/pool.py"
s = open(p).read()
s = s.replace("from FlagEmbedding import FlagModel",
              "from sentence_transformers import SentenceTransformer")
s = re.sub(r"FlagModel\(.*?\)", 'SentenceTransformer("BAAI/bge-large-en-v1.5")', s, flags=re.S)
open(p, "w").write(s)
print("[ok] patched", p)

# 4) yaml: task_num 100 -> 5 (smoke test)
y = "config/experiment/bigbenchhard_new_abilities.yaml"
s = open(y).read()
s = s.replace("task_num: 100", "task_num: 5")
open(y, "w").write(s)
print("[ok] patched", y, "(task_num -> 5)")
print("ALL PATCHES DONE")
