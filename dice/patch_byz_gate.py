"""DICE Phase 4 fix: gate Byzantine ability-inflation behind env BYZ_INFLATE (default OFF),
so the default Byzantine only emits wrong output -> graded resilience curve.
(Ability-inflation = catastrophic instant collapse, kept as a separate toggle.) Run from AgentNet_Code/."""
f = "src/agent.py"
s = open(f).read()
old = 'if _dice_is_byzantine(self.agent_id):\n            abilities = {k: 1.0 for k in self.abilities}'
new = 'if _dice_is_byzantine(self.agent_id) and __import__("os").getenv("BYZ_INFLATE", "0") == "1":\n            abilities = {k: 1.0 for k in self.abilities}'
if old in s:
    s = s.replace(old, new, 1)
    open(f, "w").write(s)
    print("[ok] ability-inflation now gated by BYZ_INFLATE (default off)")
elif "BYZ_INFLATE" in s:
    print("[ok] already gated")
else:
    print("WARN: inflation block not found")
import py_compile
py_compile.compile(f, doraise=True)
print("[ok] compiles")
