# DICE 实验 — 详细报告（背景 / 算法 / 各 Phase / 结果 / 审计 / 下一步）

> 这是一份完整、力求诚实的写作版报告（README.md 是流水账，这份是体系化版本）。
> 写于 2026-06-22。所有"准确率"若无特别说明都是**训练集/在线(prequential)准确率，不是 held-out 测试**（见 §8 局限）。

---

## 0. TL;DR

把一篇只测 9 个 agent 的去中心化多智能体论文（**AgentNet**, NeurIPS 2025）改造成能在本地跑到 **1000 个 agent**，并做了一系列实验。**两个结构性强结论（不受样本噪声影响）**：

1. **可扩展性**：原版"全连接"协调在 **N=50** 因 router prompt 超 8192 上下文而**崩溃**；任何"**bound 邻居视野**"的结构（平均场 top-K / 随机稀疏）都扩到 **N=1000**，每个 prompt 成本恒定。
2. **鲁棒性**：当坏 agent **虚报能力**时，原版按自报能力的路由在 **≥20% 坏 agent 时即全崩(0.0)**；改成按**实际战绩(信誉)**路由后**防止了完全崩溃**（held-out 测试：50% 坏 agent 仍 **0.33**，naive=0），但**随 φ 明显降级**（无攻击 0.59 → 50% 时 0.33），**并非完全免疫**（修正版数字，已剔除早期被美化的 artifact）。

**一个诚实的负面结论**：就准确率而言，平均场(按能力 top-K) **≈** 随机稀疏——可扩展的关键是"bound 视野"本身，不是平均场那套"挑最强"。平均场的价值在别处（可证明保证 / 负载集中 / 可分析的 N→∞ 极限）。

**重要限定**（§7–8 详述）：这是 BBH 推理问答上的"有界路由注意力"实验，**不是严格意义的 mean-field 实验**；准确率是在线训练分；robust 的信誉用了 ground-truth(=oracle)；多个 caveat 待修。

---

## 1. 背景

### 1.1 DICE 项目想要什么
DARPA **DICE**（去中心化 AI / 受控涌现）押一个赌注：未来要协同**成百上千上万**个 AI agent（目标 500→5K→100K），而**中心化协调撑不到那个规模**，必须用**去中心化 / 平均场**式协调，且要在对抗（拜占庭）环境下鲁棒。核心指标包括交互复杂度 O(N²)→O(N)、拜占庭比例 20%→33%、恢复时间等。

### 1.2 分工与定位
- **师兄**：做 mean-field **理论**（极限方程、收敛率、阈值的可证明保证）。
- **我（实证）**：做**实验**去验证/演示。
- **AgentNet** 是老师点名的参考（去中心化、router/executor、拓扑自演化、RAG 记忆），但论文只测到 3–9 个 agent。所以一个自然的实验目标：**拿 AgentNet 当试验台，本地推到大规模，实测"中心化会不会崩、去中心化/平均场能不能扩"，并测对抗鲁棒性。**

### 1.3 要回答的核心问题
1. 中心化协调在大 N 真的会崩吗？崩在哪、为什么？
2. 把"每个 agent 只跟群体摘要(场)交互"的平均场改法装上，能不能扩展？代价多少？
3. 平均场是不是"特殊"，还是随便哪种可扩展结构都行？
4. 对抗（拜占庭）下系统怎么垮、能不能修？

---

## 2. 系统与算法（AgentNet）

### 2.1 一个 agent 是什么（agent ≠ 模型）
一个 agent = **它自己的状态 + 一个策略**，不是一份 LLM 拷贝：
- **状态**：角色、**能力向量** `abilities={reasoning,math,language,...}`、**每类任务的成功率** `success_rate[task_type]`、**经验池**（过去任务轨迹）。
- **策略**：要"思考"时，把一段 prompt 发给一个**共享的 LLM 服务**，拿回文本。

**所有 N 个 agent 共享同一个 vLLM 模型服务**（一份 7B 权重）。agent 之间的差异全在权重之外（能力数字 / 记忆 / 边权重），靠在很多任务上**演化**而分化（做对某类任务→该类能力上调→更可能被派同类任务→专业化）。

### 2.2 agent 内部：router + executor
- **router**：决定一个任务 **forward(转给别人) / split(拆开) / execute(自己做)**。
- **executor**：真正作答。
- 两者各有记忆池，作答时把**检索到的相关历史经验(RAG, top-3, 用 BGE 嵌入算相似度) + 当前任务**拼进 prompt 发给 LLM。

### 2.3 路由：任务交给谁（这是本工作改动的核心）
- **初始选择** `select_an_agent(task_type)`：从所有 agent 里挑（原版按自报**能力**最高）。
- **转发** `find_best_alternative_agent`：从邻居里挑。
- **邻居信息** `collect_neighbors_info`：原版在"complete 图"下返回**所有 N−1 个**邻居 → `router_make_prompt` 把它们**逐条写进 prompt** → **prompt 随 N 线性增长 O(N)**（这就是大 N 崩溃的根因）。

### 2.4 我加的几个开关（全用环境变量切换，其余代码不动）
| 开关 | 作用 |
|---|---|
| `ROUTE_MODE=graph` | 原版：每 agent 看所有邻居（O(N) prompt） |
| `ROUTE_MODE=field` | **平均场**：`collect_neighbors_info` 只留"本任务能力 top-K(=4)"个邻居（O(1) prompt） |
| `ROUTE_MODE=sparse` | 随机稀疏：每次随机留 K 个邻居 |
| `ROBUST=1` | **信誉路由**：`select_an_agent` 和 `find_best_alternative_agent` 都改成按**实际战绩 success_rate** 排序，而非自报能力 |
| `BYZANTINE_FRAC`,`TOTAL_AGENTS` | 把 id < frac·N 的 agent 设为拜占庭：executor 输出错答案 |
| `BYZ_INFLATE=1` | 拜占庭额外**虚报满能力(1.0)**以吸引路由 |
| `SEED` | seed `random`+`numpy`（注意：不改任务集） |
| `SMOKE_TASKS` | 限制任务数（快速迭代） |

---

## 3. 实验设置
- **服务器**：`ece-drf95318s`（OSU，8×H200 NVL，共享机）。**只用 GPU 4（vLLM 服务 Qwen2.5-7B）和 GPU 6（BGE 嵌入）**，从不碰 0–3（别人的）。
- **推理**：本地 vLLM，**全程不用云 API**。因机器无 nvcc，vLLM 必须免 JIT 启动（`VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN --enforce-eager`）。
- **任务**：BigBenchHard（27 类推理 QA）。**指标 = 训练集/在线准确率**（见局限）。
- 改 AgentNet 走本地：LLM 派发→本地 vLLM；FlagEmbedding→sentence-transformers；入口 `CUDA_VISIBLE_DEVICES`"0"→GPU6。

---

## 4. 各 Phase 详解

### Phase 0 — N-sweep 工具链
- **目的**：研究"行为随 N 变化"前，先有能自动变 N 跑+收指标的仪器。
- **做法**：脚本 `nsweep.sh`，N=3/5/7/9 各 10 题。
- **结果/含义**：工具通了；数字(0.7/0.9/0.7/0.7)噪声大，**只验证工具，不是结论**。

### Phase 1 — 实现平均场(route-to-field)
- **目的**：要对比两种协调方式，先把它们做进一份可切换代码。
- **做法**：`patch_r2.py` 给 `collect_neighbors_info` 加 `ROUTE_MODE=field`（top-K 按能力）；`patch_r2b.py` 优化为"先廉价按能力排序砍 top-K、再只对这 K 个做昂贵的经验检索"（O(K)）。
- **结果/含义**：实现+编译通过；改动只在一个函数，graph/field 一键切换 → R1 vs R2 是公平对比。

### Phase 2 — R1 vs R2 小 N 对照
- **目的**：在说"平均场更能扩展"前，先确认它小 N 不比原版差。
- **做法**：`r1r2.sh`，graph/field × N=5/10/20，20 题/单 seed。
- **结果**：

  | N | graph | field |
  |---|---|---|
  | 5 | 0.75 | 0.65 |
  | 10 | 0.65 | 0.65 |
  | 20 | 0.55 | 0.65 |
- **含义**：可比；且 N=20 时 graph prompt 列 20 个 agent、field 只列 5 个（实测 O(N) vs O(1)）。差异在噪声内。

### Phase 3 — 把 N 推大（核心结果）
- **目的**：找原版崩溃点 + 证明平均场越过它。
- **做法**：`scale.sh`(field/graph N=50/100/200)、`scale2.sh`(field N=500/1000)，20 题。
- **结果**：

  | N | graph | field | field prompt | field 耗时 |
  |---|---|---|---|---|
  | 50 | **崩** | 0.55 | 6 | 316s |
  | 100 | **崩** | 0.70 | 5 | 371s |
  | 200 | — | 0.65 | 5 | 386s |
  | 500 | — | 0.55 | 5 | 344s |
  | 1000 | — | 0.60 | 5 | 441s |
- **含义（最硬结论）**：graph 在 N=50 崩，真实报错 `BadRequestError 400: maximum context length is 8192 tokens ... prompt contains at least 6145 input tokens`——router 把 ~50 个 peer 全列进 prompt 超窗。field 把 prompt 压到 O(1)，**扩到 N=1000**（DICE Phase-1 目标 500 的 2×）。**这是结构性的（O(N) prompt 迟早撞任何固定窗口），不依赖噪声。**

### Phase 4 — 拜占庭
- **做法**：`byz.sh`，注入坏 agent，扫 φ。
- **结果**：
  - **虚报能力(BYZ_INFLATE=1)**：坏 agent 谎报满能力 → `select_an_agent` 总选它 → **φ=0.1 即 0.70→0.0** 全崩。**= 一个脆弱性**（能力路由对虚报零防御）。
  - **只输出错答案**：两种模式都**优雅降级**（φ=0.5 仍 ~0.45），无级联崩溃。
- **含义**：DICE 极重视对抗鲁棒性；这暴露了 AgentNet 路由的一个真实弱点。

### Phase 5 — 信誉路由（修复，TA1 贡献）
- **目的**：把"脆弱性"升级成"修复"。
- **做法**：`patch_robust.py` 让 `select_an_agent` 在 `ROBUST=1` 时按**实际战绩**而非自报能力选；`robust.sh` naive vs robust × φ，虚报场景。
- **结果**（select-only robust，单 seed，40 题）：naive φ≥0.1 全 0.0；robust φ=0/0.1/0.2/0.33/0.5 = 0.675/0.625/0.575/0.6/0.55。
- **含义**："简历能伪造，战绩不能"——按战绩选直接化解虚报攻击。

### Phase 6 — 结构对比（3 seeds 去噪）
- **目的**：回答"平均场是否特殊，还是随便哪种 bound 结构都行"。
- **做法**：`patch_sparse.py` 加 `ROUTE_MODE=sparse`(随机 K)；`struct.sh` field/sparse/graph × N=20/100/500 × 3 seeds，30 题。
- **结果**（3-seed 均值）：

  | N | graph | field 平均场 | sparse 随机 |
  |---|---|---|---|
  | 20 | 0.72 | 0.60 | 0.67 |
  | 100 | 崩 | 0.66 | 0.62 |
  | 500 | 崩 | 0.62 | 0.61 |
- **含义（诚实修正）**：**field ≈ sparse，无一致赢家**（我单 seed 时看到的"field 大 N 胜出"是噪声，已修正）。→ **可扩展性来自"bound 视野"本身，不是平均场的"挑最强"**。graph 全信息在 N=20 最高但扩不了。结构差异：field 集中用 ~6 个 agent，sparse 铺到 ~190+ 个（负载分布不同，可能影响鲁棒性）。

### Phase 7 — 鲁棒路由补全 + 去噪验证
- **做法**：`patch_robust2.py` 把信誉路由也用到转发；`robust2.sh` naive vs 全 robust × φ × 2 seeds，虚报场景。
- **结果**（2-seed 均值）：

  | φ | naive | 全 robust |
  |---|---|---|
  | 0 | 0.60 | 0.57 |
  | 0.2 | **0.0** | 0.50 |
  | 0.33(实为30%) | **0.0** | 0.53 |
  | 0.5 | **0.0** | 0.43 |
- **含义**：全 robust（选择+转发）、2-seed 平均下仍完全挡住虚报崩溃。鲁棒贡献完整+去噪（但见 §7 漏洞）。

---

## 5. 结果汇总（solid vs 方向性）
- **Solid（结构性、大效应）**：①graph N=50 崩 / field 到 N=1000；②虚报下 naive 0.0 / robust 0.43–0.57。这两个不受样本噪声影响。
- **方向性（噪声大）**：所有 0.5–0.7 之间的准确率小差异（R1 vs R2、field vs sparse、错答案拜占庭曲线）。

---

## 5.5 held-out 测试集结果（修正 #1，2026-06-23）
§4 各表是**训练集/在线**准确率。`evaluate()` 其实在 fit 之后于**held-out 测试集**上评估（**测试时图冻结、不再学习**=正经 held-out），日志里一直有 `Test Dataset Is`，之前漏取了。从现有 log 重新抽取（**无需重跑**），测试分系统性高于训练分（训练分含早期学习曲线的失败），**且所有结论在 held-out 上依然成立、甚至更干净**：

**结构对比（held-out TEST，3-seed 均值）**
| N | graph | field 平均场 | sparse 随机 |
|---|---|---|---|
| 20 | 0.74 | 0.73 | 0.81 |
| 100 | 崩 | 0.73 | 0.73 |
| 500 | 崩 | 0.70 | 0.72 |
→ **field ≈ sparse 在测试集上同样成立**（sparse 小 N 还略高）；honest negative 坐实。

**可扩展性（held-out TEST）**：field N=50/100/200/500/1000 = 0.65/0.70/0.70/0.65/0.65（平稳）；graph N≥50 崩。

**鲁棒性（held-out TEST，⭐修正版 = 干净 robust + 随机拜占庭 id + 随机任务子集 + 3-seed 均值；`robustfix.sh`/`patch_fixes.py`）**
| φ(虚报) | naive | robust |
|---|---|---|
| 0 | 0.62 | 0.59 |
| 0.2 | **0.0** | 0.46 |
| 0.33(实30%) | **0.0** | 0.41 |
| 0.5 | **0.0** | 0.33 |
→ **修正后的诚实结论**：naive 在 φ≥0.2 时完全崩(0.0)；robust **防止完全崩溃**但**随 φ 明显降级**（0.59→0.33），**不是"完全免疫"**。⚠️ 之前 robust2 报的 0.48–0.78 被"拜占庭恒为低 id + 偏袒诚实的 tie-break + 固定任务子集 + field 预筛仍按能力"几个 artifact 美化了，**此修正版取代之**。定性差距(robust>0 vs naive=0)仍稳固，量级大幅下调。seed 间波动仍大（如 robust φ=0.33 测试 0.23/0.47/0.53），需更多 seed。

**结论**：换 held-out 测试 + 修干净后，两个 solid 定性结论不变（graph 崩 vs bound 扩展；robust>0 vs naive=0），但**鲁棒的定量幅度从"完全挡住"下调为"防止完全崩溃、随 φ 降级"**。§7.8/§7.1/§7.caveat 均已解决。

## 5.6 消融 + 延迟发作攻击（#4，2026-06-23，`abl.sh`/`patch_phase4.py`）

**K 消融**（field，N=100，无拜占庭，held-out TEST，2-seed 均值）：
| K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| test | 0.65 | 0.62 | 0.65 | 0.58 |
→ **无明显趋势，连 K=1（每 agent 只看 1 个邻居）都和 K=8 一样好**。bound 的"大小"几乎不影响准确率，可砍到极小、更省——再次印证"bound 本身才是关键"。

**延迟发作拜占庭**（field，N=20，虚报，φ=0.33；坏 agent 前 15 个任务正常干、攒够信誉后叛变；held-out TEST，2-seed 均值）：
| defense | 立即发作 | 延迟发作 |
|---|---|---|
| naive | 0.0 | 0.0 |
| robust | 0.52 | **0.42** |
→ **信誉路由可被"先取信后背叛"部分 game**：robust 在延迟攻击下(0.42)比立即(0.52)更差——坏 agent 攒够信誉再叛变，robust 仍把任务派给它。但仍 >naive(0.0)（叛变后信誉下降、robust 终会绕开，故只是**部分**被 game）。**证实 codex 批评：基于 ground-truth 的即时信誉不足以防延迟发作。**

## 6. 代码审计（我自查）
对所有补丁逐函数审计，**没有破坏正确性的硬 bug**（select_an_agent/collect_neighbors_info/find_best_alternative_agent/get_self_info/executor 注入/seed/smoke 都按预期工作）。但发现若干**方法学/实现 caveat**（见下，与 codex 评审合并）。

## 7. codex 独立评审的发现（重要，已诚实采纳）
用 codex-cli(gpt) 对补丁做了独立评审，确认我的 caveat 并补了几条关键的：

**实现层面**
1. ⚠️ **robust 在 field 下不干净**：`collect_neighbors_info`(field) **先按自报能力**选 top-K 候选，**再**给 robust 转发——虚报者能占满候选集。robust 结果**仍成立**（因初始 `select_an_agent` 是 robust 且占主导），但要干净，**field 的邻居预筛也应按信誉**。← 待修。
2. ⚠️ **转发 tie-break 偏置**：`find_best_alternative_agent` 用 `sort(reverse)` 对 `(score,id)` 排序，打平时偏**高 id**；而拜占庭恒为**低 id** → 系统性偏向诚实 agent，**轻微 flatter 了 robust**。应随机化"哪些 id 是拜占庭"+随机 tie-break。
3. ⚠️ **BYZANTINE_FRAC 向下取整**：N=20、φ=0.33 → int(6.6)=6 = **30%**（非 33%）。报告需按实际比例写。
4. **直接索引可能 KeyError**（`task_to_ability_map[task_type]` 等）对未映射任务类型——本数据集没触发，但脆。
5. **fallback 随机**：候选为空时随机选，可能把"robust"行为混成随机；应记录 fallback 频率。
6. **graph 模式是否严格等于原版未正式核对**（我整体重写了该函数，应等价但没做 parity check）。

**方法学层面**
7. ⚠️⚠️ **这不是严格的 mean-field 实验**：top-K 截断是"有界路由注意力"，BBH 无群体级状态/序参量/相互作用动力学。**scaling 是"上下文工程结果"，不验证平均场集体行为。** 真要支撑 DICE 平均场主张，得换 GovSim 这类有聚合动力学的环境（测序参量、N^(−1/2) 方差、级联阈值 φ_c）。
8. ✅**已解决（见 §5.5）**：之前报的是训练集/在线分；evaluate() 其实有 held-out 测试（测试时图冻结），已从现有 log 重抽测试分重报，结论不变且更干净。
9. ⚠️ **信誉是 oracle**：success_rate 用 ground-truth 正确性算的；真去中心化里信誉本身也能被伪造/操纵。我的 robust 更接近"oracle 审计路由"。
10. **seeds 不覆盖任务方差**：30 题由固定 `rng(888)` 选定，SEED 只 seed `random`，LLM 贪心(temp0)。30 道二元题的二项噪声很大，**任务子集才是主导噪声源**。需 10–20 个任务子集 seed + bootstrap 置信区间。
11. **sparse 是动态随机重采样，非固定稀疏图**（每次路由重抽 K）——和"固定局部连通"是不同 baseline。
12. **scaling 要区分 prompt 成本 vs 算法成本**：field 选择仍 O(N) 扫描、select_an_agent 全局 O(N)，到 5K–100K 会显现。

### 7.x 已修复（2026-06-23, `patch_fixes.py` + `robustfix.sh`）
- ✅ **#7.1 robust 在 field 下不干净** → field 预筛在 ROBUST 下改按**信誉**。
- ✅ **#7.2 / caveat-1 拜占庭低 id + tie-break 偏置** → 拜占庭 id **按 seed 随机选**；select/forward/field 全部**随机 tie-break**。
- ✅ **#7.8 训练分 vs held-out** → 改报 held-out 测试（§5.5）。
- ✅ **#10 / caveat-2 seeds 不覆盖任务方差** → `SMOKE_TASKS` 改成**按 seed 随机抽任务子集**，3 seed。
- **修正后果**：鲁棒数字大幅下调（"完全挡住"→"防止完全崩溃、随 φ 降级"，见 §5.5）；可扩展性/honest-negative 结论不变。
- ⏳ **仍未做**：#7.3 fallback 频率日志、#7.6 graph parity 正式核对、#7.9 oracle 信誉、#7.11 固定稀疏图、#4 K 消融、#5 真·mean-field(GovSim)、更多 seed/bootstrap CI。

## 8. 局限性（汇总）
- 设定：单一任务族(BBH 推理 QA)、单一骨干(Qwen2.5-7B)、本地试验台——**非 DICE mission / 非异构 agent**。
- 指标：训练集/在线准确率（非 held-out）。
- "平均场"是有界路由近似，非严格平均场极限。
- robust 的信誉是 oracle、拜占庭攻击很简单(固定错答案+虚报)、拜占庭 id 固定为低 id。
- 去噪只覆盖路由随机性，未覆盖任务子集方差。
- 结论分级：**只有"崩/全崩"这种大效应是 solid，其余定量数字是方向性的。**

## 9. 下一步（状态更新 2026-06-23）
1. ✅ **held-out 测试集重报**（§5.5）。
2. 🟡 **去噪**：已做随机任务子集 + 2–3 seed（§5.5/5.6）；完整 10–20 seed + bootstrap CI 仍可加。
3. ✅ **修干净 robust + 公平化拜占庭**（§5.5，`patch_fixes.py`）。
4. ✅ **K 消融 + 延迟发作攻击**（§5.6，`patch_phase4.py`）。
5. 🟡 **更强拜占庭套件**：已加延迟发作；剩 谎报信誉 / 恶意转发 / 经验池投毒 / 合谋 / trimmed-median 聚合。
6. ⏳ **真·平均场实验（GovSim，支撑 DICE 主张的关键，最大工程）**：扫 N，记序参量、测 N^(−1/2) 方差收敛、估级联阈值 φ_c——对接师兄极限方程。**当前最大缺口。**
   - **进度（2026-06-23）**：已 clone GovSim 到 `~/dice/GovSim` 并 recon。渔业/牧场/污染三场景，hydra 配置，`is_api` 标志支持 API 后端（→ 可指向本地 vLLM 服务器，绕开它自带 vllm 0.6.4）。默认 ~5 agent、agent 间全自然语言对话（大 N 会撞同样的上下文墙）。
   - **剩余工程量大且分两块**：(A) 配环境 + 跑通 5-agent 基线（纯 infra，无需师兄，但像 AgentNet 一样要调试）；(B) **真正的科学**——N-scaling + mean-field 耦合(agent 只看聚合量) + 序参量记录 + N^(−1/2) 收敛 + 级联阈值。**(B) 的设计(序参量是什么、对哪个极限、耦合形式)应与师兄理论共同设计**，否则可能测错对象。
- 小项：fallback 频率日志、graph parity 核对。

## 10. 文件 / 复现
- **代码+结果**：fork `github.com/CatherineYellow/AgentNet`，`dice/` 文件夹（补丁 `patch_*.py`、脚本 `*.sh`、结果 `*_results.csv`、`RESULTS.md`）。最新 commit 见 a1ac49b 系列。
- **服务器**：代码 `~/huangjj/AgentNet`；环境 `~/dice-env`（vLLM）；日志/结果 `~/*_results.csv`、`~/*.log`。
- **启动 vLLM**（无 nvcc 必须免 JIT）：
  `CUDA_VISIBLE_DEVICES=4 HF_HOME=~/hf_cache VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.4 --max-model-len 8192 --enforce-eager`
- **跑实验**：`bash dice/<script>.sh`，旋钮 `ROUTE_MODE / FIELD_K / SMOKE_TASKS / BYZANTINE_FRAC / BYZ_INFLATE / ROBUST / SEED`。
- ⚠️ 关 vLLM 要杀 `VLLM::EngineCore` 子进程（`pkill -f "vllm serve"` 杀不掉它，按 `nvidia-smi` 的 compute-apps PID 杀）。

---
*本报告对结果做了诚实分级与限定；§7 的 codex 评审条目是后续工作的明确清单。最强的可写进 proposal 的是 §0 的两个结构性结论；定量准确率需按 §9.1–9.2 修正后再用。*
