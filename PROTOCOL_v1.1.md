# Fixed-Checkpoint Workspace-Dependence Audit — 完整试验设计 v1.1

**状态**:Freeze-1 候选(设计 + 分析计划 + 工程 fixture)。经两轮外部方法学审查(v0:11 项;v1:5 项 blocking + 1 项治理),本文档整合全部修订,为可直接搭建仓库的完整版本。
**日期**:2026-09-02
**前身项目**:https://github.com/Alanjiao1988/J-space-observation(head `d87c1b9e4e9dca062cebad7b6eee981d9dba8c25`,2026-08-29,`DISCONTINUED — RESEARCH QUESTION UNANSWERED`)
**上游论文**:Gurnee, Sofroniew, …, Lindsey. *Verbalizable Representations Form a Global Workspace in Language Models.* Transformer Circuits, 2026-07-06. https://transformer-circuits.pub/2026/workspace/index.html
**上游代码**:https://github.com/anthropics/jacobian-lens(Apache-2.0;单 commit 参考实现,标注 not maintained)
**治理规则**:本文档冻结后不再进入全量审查;之后的 gate 是 Phase A 数据,不是第三份文档(SC5)。

---

## 第一部分 · 背景与约束

### 1. 一句话定义

> 本项目测量:在指定的一对固定 checkpoint(`DeepSeek-R1-Distill-Qwen-14B` 与 `Qwen2.5-14B-Instruct`,共享 parent `Qwen2.5-14B`)上,**J-space 消融对最终答案准确率的影响,是否随"外部 rationale 是否在上下文中"而不同,以及这一交互在两个 checkpoint 之间是否有实质差异。**

主结论的合法名称是 **J-space ablation × response-format interaction**。"externalization"是候选解释,不是结论。

### 2. 文献核实(最终版)

| 主张 | 结论 | 证据 |
|---|---|---|
| 论文被试为 Claude 系列 | 确认 | 默认 Sonnet 4.5,Haiku 4.5 / Opus 4.5 佐证,部分用 Opus 4.6 |
| 后期层 J_ℓ → I 是预期行为 | 确认 | 论文:logit lens 即 J_ℓ = I,"在后期层因残差连接而合理";motor 转变处 "J_ℓ approaches the identity" |
| logit lens 捕获大部分 workspace 结构 | 确认 | 论文原文:logit lens "quite useful in practice… captures much of the workspace-like structure… with somewhat lower reliability (particularly in earlier layers)" |
| GSM8K CoT 比 direct 对消融更鲁棒 | 确认 | 论文 ablation battery;解释为"外化到纸面" |
| capacity 核心是 related vs unrelated | 确认 | 论文 Fig. 31B;四段 block 是 Fig. 31E/F 子变体 |
| ignition 对比是 J-限制分量 vs 全激活 | 确认 | 论文 Fig. 29B/C |
| verbal-report 无 strength sweep | 确认 | sweep 属 verbal-introspection(Fig. 7) |
| 上游代码含 patch / ablate 接口 | **不能确认** | 公开 README 仅显示 `jlens.fit`、`JacobianLens.from_pretrained`、`lens.apply`;消融/patching/gradient-pursuit 分解须自行实现 |
| `data/` 含 GSM8K 或 direct/CoT runner | **否** | `data/experiments/` 为 prompt set;无 GSM8K、无评分脚本 |
| DeepSeek 阶梯 parent | 确认(官方模型表) | 1.5B ← Qwen2.5-Math-1.5B;7B ← Qwen2.5-Math-7B;14B ← Qwen2.5-14B;32B ← Qwen2.5-32B;均以 800k R1 样本微调 |
| Neel Nanda 外部 review 结果 mixed | 确认 | rhyme-planning、mental-arithmetic 未复现;probe-swap 弱;假阳性多;multi-fact editing(flexible generalization)复现干净 |
| tao-hpu/jspace-replication | 确认 | 加入 direct final-token substitution baseline、amplitude-matched random controls、1.7B–14B ladder |
| amaljithkuttamath/jlens-replication | **确认(v1 记录有误)** | Qwen2.5-3B-Instruct,n=25 拟合:shuffled-corpus lens 在全部 6 个 eval 上 MRR ≥ 真实 lens(typo 0.806 vs 0.318);概念方向消融无系统效应;Gemma-4-E2B 上 shuffled 同样失败但消融有强负效应;作者自认严重欠拟合(n=25 vs 1000;skip_first 4 vs 16) |
| arXiv 2608.25347 | 确认 | Jacobian energy 随深度衰减、高度稀疏、集中于对角通路与关键位置 |

### 3. 从前身继承的事实(设计约束)

| # | 事实 | 前身路径 |
|---|---|---|
| F1 | 1.5B 在上游 evaluations 上 2/93、2/55、5/90;lens 从未被读取 | `studies/study1/README.md` |
| F2 | 四选一 restricted-logit 接口下全臂随机;NT 臂 C = 0/384 | `studies/study2/analysis/stage_bd_posthoc_interface_diagnostic.md` |
| F3 | 内容面 `" 0"`..`" 9"` 为两 token `[220, digit]` | `studies/study3/README.md`(P0-T) |
| F4 | 7B/14B 在 template + CoT 下 97/104、104/104、101/104;raw-direct 下 240/240 不可解析,7B 首 token `</think>`(151649),14B 首 token `Okay`(32313) | `studies/study4f/execution-m1/M1_FINAL_DISCLOSURE.md` §11, §13 |
| F5 | 4F 的 E0 envelope 无 template、无强制闭合 | `studies/study4f/analysis/study4f_interfaces.py` |
| F6′ | Study 5 EQ2 在 Qwen2.5-Math-7B(28 层)上**扫描了全部层**:readrate 于 L0–12 精确为 0,约 L20 起上升,注册规则 band 为 [21…26],条件 (ii) 压至长度 1;L23–26 identity energy 0.749、α ≈ 1.0。**修正**:后期层 J ≈ I 是 motor 层预期行为,不构成仪器病理;但"早/中层无 readrate 信号"的终止结果**成立**。另:注册的 band 规则无绝对下限、无最小长度,阴性对照亦产出 band [9],规则本身有缺陷 | `studies/study5/closure/STUDY5_CLOSURE.md`;`studies/study5/qualification-eq2/TIMELINE.md` |
| F7 | bf16 归约偏移 0.11–0.48 logit,与效应 +0.0386 同量级 | `STUDY5_CLOSURE.md` §5–6 |
| F8 | J-lens 线从未获得阳性对照 | 同上 §6–8 |
| F9 | 上游 `data/experiments/` 度量全部为同模型内条件差;前身只用 `evaluations/` 与 probe-swap | `third_party/jacobian-lens/581d3986…/data/experiments/README.md` |
| F10 | 前身 vendored 仅三个 JSON,无代码 | 同上目录 |
| F11 | 前身所有判定为绝对阈值 gate | Study 2/3/4F 各决策文件 |
| F12 | 公开 checkpoint 比较不能辨识蒸馏训练导致了什么 | `PROJECT_DISCONTINUATION.md` §3 |
| F13 | 4F-M1 执行纪律:seal 先于首次调用;hard-kill / CPU 恢复为预注册 canary;**科学层面**零 rerun | `M1_FINAL_DISCLOSURE.md` "Execution integrity" |

**推论**:1.5B 不是被试(F1);绝对水平判定被接口污染,只用同模型内差分(F2、F4、F11);消融必须在 fp32 或 bit-exact 证明下进行(F7);仪器资格与阳性对照先于任何科学推断(F6′、F8);消融实现为自建工程,须有 conformance fixture(§2 上游能力核实)。

---

## 第二部分 · 科学设计

### 4. 研究问题与 claim ceiling

**RQ0(移植与仪器资格)**:自建的 J-lens 拟合 + 消融实现能否通过 conformance fixture,并在 held-out 判据上通过仪器资格?

**RQ1(主假设)**:在 14B 指定对上,ablation × response-format 交互量 G 的差 D 是否有实质差异?

**RQ2(分层重复,条件性)**:7B(Math-parent 层)与 32B(general-parent 层)上,D 是否同向?各层独立报告。

**允许的措辞**:"在 checkpoint X 与 Y 之间,G 的差为 D = …(CI …),判定为 有意义 / 等价 / 不确定。"
**禁止的措辞**:`reasoning-specific`、`推理蒸馏模型类别`、`scale trend`、`蒸馏导致`、`externalization 已证实`、`J-space 存在/不存在`、任何从两个 checkpoint 到训练类型的推广。

### 5. 主实验:三个回答格式条件

所有条件使用模型**原生 chat template**。

| 条件 | 上下文 | 消融窗口 |
|---|---|---|
| **C_direct** | 题目 + 指令"只输出最终数字" | 窗口 W |
| **C_frozen**(primary 对照) | 题目 + **预先冻结的正确 rationale**(由生成器给出,非模型生成)+ 指令"只输出最终数字" | 窗口 W(与 C_direct 完全相同的定义与长度) |
| **C_gen**(secondary,论文可比) | 题目 + 指令"逐步推理后给出最终数字" | 全部生成位置(与论文 GSM8K 设置对齐) |

**窗口 W 的定义(冻结)**:最后 L_p 个 prompt 位置(答案提示后缀,如 `Final answer:`,L_p 在 Freeze-1 固定)+ 全部生成的答案位置(`max_new_tokens` 相同)。C_direct 与 C_frozen 的 W 在 token 数上完全一致,故**消融暴露次数相同**,唯一差异是上下文中是否存在正确 rationale。

**量的定义**:
- `Δ(c) = acc_unablated(c) − acc_ablated(c)`,acc 为 ITT 正确率(§9.4)
- **G_primary = Δ(C_direct) − Δ(C_frozen)**
- G_gen = Δ(C_direct) − Δ(C_gen)(secondary;受生成长度、暴露次数、自我纠错混杂,仅作论文对照)
- **Primary endpoint:D = G_primary(R1-Distill-14B) − G_primary(Qwen2.5-14B-Instruct)**

**服从性**:R1-Distill 在 C_direct 下可能拒绝直接作答而进入 `<think>`(F4)。处置:主条件**不使用 prefill**;parseability endpoint 按条件分列报告;若 C_direct 可解析率低于 Freeze-1 阈值,记为 feasibility result。敏感性变体 `C_direct_prefill`(prefill `<think>\n\n</think>`)作为 exploratory 臂,不进入 primary。

**消融的定义(冻结要素)**:在 workspace band 的每一层、窗口 W 的每个位置,取 clean pass 下投影最强的 k 个 J-lens 方向(排除 clean pass 输出分布 top-10 中的 token),将残差流在这些方向上的投影置零。k、band、强度档位(light / medium / heavy 按层带定义)在 Freeze-2 固定。

### 6. 仪器

**6.1 lens 拟合**(每个被试)
- `J_ℓ = E_{t, t'≥t, prompt}[∂h_final,t' / ∂h_ℓ,t]`;读出 `softmax(W_U · norm(J_ℓ h_ℓ))`
- 两个**独立** real-corpus lens(不相交语料抽样,seed 不同)
- 一个 **shuffled-corpus** lens(同规模语料,prompt 内 token 顺序随机置换)
- **logit lens** 基线(J_ℓ = I)
- tuned lens 基线(可选)
- 拟合规模:论文 1000 × 128 token;上游称约 100 条可用;amaljithkuttamath 在 n=25 下 shuffled 不劣于真实——**拟合规模是 Phase A 校准对象,下限不得低于 200 条,skip_first 不得低于 16**

**6.2 lens 合并规则(冻结)**:两个 real-corpus lens **都运行**,主估计为二者配对平均;lens 拟合作为 bootstrap 的一个层级。**禁止事后选择效果更强的一套。**

**6.3 workspace band 定位(冻结为自动规则,人不介入)**:按论文四项统计在校准模型上自动计算——(a) J-lens top-k 命中模型 top-1 的准确率;(b) readout 超额峰度;(c) top-1 token 跨位置自相关(相对 position-shuffled null);(d) 有效线性维度。band = 峰度上升起点至 next-token 准确率陡升起点之间的连续层段。规则含**绝对下限与最小长度**(修正 F6′ 的规则缺陷)。

### 7. 对照组(全部强制)

论文自身使用的:

| 对照 | 用途 |
|---|---|
| matched-norm 随机扰动 | 排除等范数任意扰动 |
| 层匹配 random-direction ablation(k 个随机方向,同层带) | ablation 阴性对照 |
| 互补分量 clamp 至 clean-pass 值 | 判定残余效应是否经 J-space 中介 |
| answer-swap vs intermediate-swap 深度差 | 排除中间量向量偷带答案 |
| position control | 注入的选择性 |
| no-instruction baseline | directed modulation 零点 |
| shuffled-position null | 自相关零假设 |
| 不出现于 prompt 的 control words | capacity 零点 |
| 跳过 clean pass top-10 token | 消融不触碰待输出内容 |

社区补充:

| 对照 | 理由 |
|---|---|
| shuffled-corpus lens | 论文未用;拟合语料必要性须自证(amaljithkuttamath 的失败即此项) |
| direct final-token substitution baseline | tao-hpu;probe-swap 关键替代解释 |
| logit lens 基线 | 论文自述其捕获大部分结构,为必须跨越的下限 |
| **general-damage 检查** | 每次消融同时记录输出分布 KL(vs clean)与 pretraining-like 文本 top-1 一致率;若 J-space 消融与 random-direction 消融的 KL 不可分,该强度档无效 |

### 8. 被试、校准模型与题库

**8.1 被试(Phase C 主比较)**

| 角色 | checkpoint | revision |
|---|---|---|
| 处理 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 前身注册 `1df85071…`,新仓库重新解析并锁定 |
| 对照 | `Qwen/Qwen2.5-14B-Instruct` | 新增,锁定 |
| parent anchor | `Qwen/Qwen2.5-14B` | 描述性锚定,不进入 primary |

**8.2 校准模型(Phase A 专用,与被试不相交)**:`Qwen/Qwen2.5-7B-Instruct`(已有社区公开 lens 可作 conformance 参照)。Phase A 的全部超参(band 规则输出、k、强度档、拟合规模、δ 的方差校准)只在校准模型上进行。

**8.3 分层重复(Phase D)**
- Math-parent 层:`DeepSeek-R1-Distill-Qwen-7B` vs `Qwen/Qwen2.5-Math-7B-Instruct`
- general-parent 层:`DeepSeek-R1-Distill-Qwen-32B` vs `Qwen/Qwen2.5-32B-Instruct`
- **不合并,不做趋势。** 1.5B 不是被试。

**8.4 题库**
- **主集(primary)**:冻结的程序生成题库。规格:2–4 步算术应用题;确定性求解器给出唯一整数答案;每题附模板化正确 rationale(供 C_frozen);答案为 1–3 位整数;按步数分层;生成器 seed 与版本哈希入 Freeze-2。理由:R1-Distill 的 800k 训练样本极可能覆盖 GSM8K 风格题,污染风险真实。
- **次集(secondary,论文锚点)**:GSM8K 测试集的 held-out 切分,附官方 rationale。保留它是为了与论文的 GSM8K 现象对照;不进入 primary。
- 探索轮与确认轮使用**不同 seed 生成的不相交题集**。

**8.5 assay 分层**
- **Primary(唯一 gate-bearing)**:主集上的 G_primary → D
- **Secondary confirmatory**(Holm 校正,仅在 primary 有意义后解释):GSM8K 上的 G_primary;flexible generalization(外部复现最干净)
- **Exploratory**(报告,不推断):G_gen;C_direct_prefill;probe-swap(强制配 direct-substitution baseline);verbal report;directed modulation;selectivity ×2(n=8/11 过小);capacity;ignition;dual-task
- **不纳入**:rhyme-planning、mental-arithmetic(外部未复现)

### 9. 统计设计

**9.1 δ(先验给定,Freeze-1)**:最小有意义准确率差。默认值 **δ = 0.10**(10 个百分点的交互差),由操作者在 Freeze-1 签字确认;探索轮的方差**只用于**样本量与功效计算,不得反向调整 δ。

**9.2 判定规则(冻结)**
- **有意义差异**:D 的 95% CI 完全位于 +δ 以上,或完全位于 −δ 以下
- **等价**:D 的 90% CI 完全落入 [−δ, +δ]
- **不确定**:其余。合法第三结局,不得改判。

**9.3 功效与样本量**:D 是四个配对比例之差的差。同一题在同一模型的四个 cell(2 条件 × 2 消融状态)中配对出现,方差由 Phase A 在校准模型上的实测配对协方差给出。目标:在 δ = 0.10、双侧 α = 0.05 下功效 ≥ 0.80。粗略无配对上界:每 cell n ≈ 800 题;配对后预期显著减少,以 Phase A 实测为准并写入 Freeze-2。

**9.4 三重 endpoint**(每个行为类 assay 按条件分列):
1. parseability:可解析率
2. conditional:可解析子集上的正确率(标注为条件量)
3. **ITT composite**:不可解析计为不正确 —— **进入 primary 的量**

任一条件可解析率低于 Freeze-1 阈值 → feasibility result,报告,不删分母。

**9.5 重抽样**:hierarchical bootstrap,层级为 lens fit → item(→ template / target,若适用)。禁止 trial 层面简单 bootstrap。

**9.6 secondary 校正**:Holm;exploratory 不校正、不推断。

---

## 第三部分 · 执行

### 10. 阶段与冻结点

**Freeze-1(本文档 + 分析计划 + 工程 fixture)** — 在任何模型调用之前。内容:§4–§9 全部、δ、可解析率阈值、band 自动规则、消融实现、conformance fixture、题库生成器代码(不含 seed)。

**Phase 0 — conformance(无科学推断)**
1. 安装上游参考实现,取 `data/` 全部 prompt set。
2. 自建消融 / patching / 稀疏分解模块,通过 fixture:
   - `noop_returns_zero`:k = 0 或强度 0 时,logits 与 clean pass **bit-exact**(fp32)
   - `projection_removed`:消融后残差在被消方向上的投影 = 0(容差 1e-6)
   - `random_control_norm_matched`:随机方向消融移除的范数与 J-space 消融相同(容差注册)
   - `lens_readout_matches_reference`:在校准模型上,本地拟合 lens 的 top-k 读出与社区公开 lens 在同一 prompt 上的 Jaccard ≥ 注册阈值
   - `known_intermediate_positive`:上游 `probe-swap.json` 中至少 N 题,在校准模型上 intermediate swap 使目标答案 logit 上升且 matched-norm random 不上升
3. 出口:全通过 → Phase A;否则修复并重做(冻结前允许);**Freeze-2 之后任何 fixture 失败 = 该 assay 无效并停止**。

**Phase A — 盲化校准(校准模型 Qwen2.5-7B-Instruct;零证据权)**
- 校准对象:拟合规模、band(自动规则输出)、k、强度档、δ 的方差、可解析率实测、C_direct 服从性。
- **盲化规则**:Phase A 不得接触被试模型;不得计算或查看任何 D、任何 G;只允许依据结构统计(§6.3)、正控制(fixture 5)、general-damage 检查选参。
- 产出:参数表,进入 Freeze-2。

**Phase B — 仪器资格(被试模型,held-out)**

| 判据 | 通过条件 |
|---|---|
| readout reliability | 两个独立 real-corpus lens 在 held-out 语料上 top-25 recall ≥ 阈值 |
| 语料必要性 | real-corpus lens 在 held-out readout 判据上**优于** shuffled-corpus lens(否则重复 amaljithkuttamath 的失败,被试出局) |
| 相对 logit lens | 在 band 的**前半段**(论文声称二者分歧处),real-corpus lens 不劣于 logit lens;**不要求全层优于** |
| causal positive control | 已知中间量 swap 在注册判据上方向正确,且 matched-norm random 不产生 |
| general-damage | J-space 消融与 random-direction 消融的输出 KL 可分 |

任一 FAIL → 该被试出局,不修补、不重跑。两个被试都必须通过,否则 SC1。

**Freeze-2(参数 + 题库 seed + lens 哈希 + 被试 revision)** — Phase B 之后、Phase C 之前。seal 先于首次确认轮模型调用。

**Phase C — 确认轮**:primary + secondary,按 §9 判定。

**Phase D — 分层重复(仅当 Phase C 为"有意义差异")**:7B 层、32B 层各自完整重复 Phase B–C,独立报告。

### 11. 重跑规则(分两层,继承 F13)

- **基础设施重跑(允许,预注册)**:硬件故障、OOM、进程被杀、checkpoint 校验失败。触发条件、最大次数、恢复方式在 Freeze-1 写死;每次重跑入日志。
- **科学重跑(禁止)**:因结果不理想而重跑、换 seed、换 band、换 k、换 δ、加样本、换 endpoint。一律不允许。

### 12. 停止条件

| 条件 | 触发 | 结论与处置 |
|---|---|---|
| SC0 | Phase 0 fixture 在冻结前无法通过 | 自建实现不可信;停止,报告 conformance |
| SC1 | Phase B 任一被试未通过资格 | 该 checkpoint 上无法建立合格 lens;停止。若失败方为处理组,RQ1 永久关闭 |
| SC2 | Phase C 判定"等价" | 该对上交互无实质差异;停止,报告阴性;RQ2 不启动 |
| SC3 | Phase C 判定"不确定" | 当前样本量下不可判定;**不加样本、不换 endpoint、不改 δ**;报告并停止 |
| SC4 | Freeze-2 后任一 fixture 失败 | 该 assay 无效,停止;不修补 |
| SC5 | 冻结后出现新 finding | 只接受能说明"哪个推断因此无效"的 BLOCKING;不启动全量审查;不产生 v2 |

每个科学停止条件对应一个可报告结局。不存在"再试一次"分支。

### 13. 明确不主张

- 任何 `reasoning-specific`、训练类型级、规模趋势级推广
- 蒸馏**导致**任何差异(需受控训练,附录 C)
- externalization 已被证实(它是 G_primary 的候选解释之一)
- J-space 存在 / 不存在;等同于推理、理解、意识
- 关于 1.5B 的任何陈述
- "首次在开源模型复现 J-space"
- 前身项目的任何仪器层失败构成关于 J-space 的证据

---

## 第四部分 · 仓库蓝图(供 Copilot 搭建)

### 14. 目录结构

```
workspace-dependence-audit/
├── README.md                      # 本文档 §1、§4、§13 的摘要 + 状态徽标
├── PROTOCOL_v1.1.md               # 本文档全文(冻结后只读)
├── FREEZE-1.json                  # Freeze-1 哈希清单(见 §16)
├── FREEZE-2.json                  # Phase B 后生成
├── references/
│   ├── predecessor_index.md       # 附录 A 的前身路径索引
│   ├── predecessor_facts.md       # §3 F1–F13 原文摘录
│   ├── literature_verification.md # §2 表格
│   └── paper_reference_values.md  # 附录 B
├── third_party/
│   └── jacobian-lens/             # 上游 clone,pin 到 PINNED_COMMIT.txt;永不修改
├── src/wda/
│   ├── models/registry.py         # 被试/校准模型 id + revision 锁定;禁止未注册模型
│   ├── lens/fit.py                # 包装上游 fit;real ×2 / shuffled / logit / tuned
│   ├── lens/band.py               # §6.3 四项统计 + 自动 band 规则(含下限与最小长度)
│   ├── lens/aggregate.py          # §6.2 双 lens 配对平均
│   ├── intervene/ablate.py        # §5 消融(top-k J 方向置零,跳过 clean top-10)
│   ├── intervene/patch.py         # lens-coordinate swap(V^† 读坐标、σ 置换、写回)
│   ├── intervene/controls.py      # matched-norm random、layer-matched random、clamp
│   ├── intervene/precision.py     # fp32 累加强制;bit-exact no-op 检查
│   ├── conditions/envelopes.py    # C_direct / C_frozen / C_gen / C_direct_prefill;窗口 W
│   ├── items/generator.py         # §8.4 程序生成题库(求解器 + rationale 模板)
│   ├── items/gsm8k_split.py       # held-out 切分
│   ├── scoring/parse.py           # 可解析判定;三重 endpoint
│   ├── stats/bootstrap.py         # hierarchical bootstrap(lens → item → template)
│   ├── stats/decision.py          # §9.2 三结局判定;Holm
│   ├── stats/power.py             # §9.3
│   ├── assays/                    # 每个 assay 一个模块 + 一页 estimand spec(.md)
│   │   ├── primary_interaction.py
│   │   ├── flexible_generalization.py
│   │   ├── probe_swap.py
│   │   └── ...
│   └── governance/
│       ├── seal.py                # 哈希清单生成与校验;seal 先于首次调用
│       ├── blind.py               # Phase A 盲化:拒绝加载被试模型;拒绝计算 D/G
│       └── rerun_policy.py        # §11 基础设施重跑规则
├── configs/
│   ├── freeze1/                   # δ、阈值、L_p、max_new_tokens、fixture 容差
│   ├── phase_a/                   # 校准搜索空间(仅校准模型)
│   └── freeze2/                   # Phase B 后写入:band、k、强度、n、seed、lens 哈希
├── tests/
│   ├── conformance/               # §10 Phase 0 五个 fixture
│   ├── governance/                # 盲化、seal、重跑规则的单元测试
│   └── stats/                     # 判定规则与 bootstrap 的合成数据测试
├── runs/
│   ├── phase0/ phaseA/ phaseB/ phaseC/ phaseD/
│   └── <run_id>/{config.json, seal.json, log.jsonl, results.json}
└── reports/
    └── <phase>_report.md          # 每阶段一份,含全部 endpoint 与 feasibility result
```

### 15. 关键接口与数据格式

**15.1 模型注册**(`registry.py`):只允许 §8 列出的 id;每个条目含 `repo_id`、`revision`、`role ∈ {treatment, comparator, parent_anchor, calibration}`、`allowed_phases`。加载未注册模型或在错误阶段加载 → 抛异常。

**15.2 条件配置**(`envelopes.py`):

```json
{
  "condition": "C_frozen",
  "chat_template": "native",
  "system": "Answer with the final integer only.",
  "rationale_source": "generator",
  "prefill": null,
  "answer_cue": "Final answer:",
  "L_p": 3,
  "max_new_tokens": 8,
  "ablation_window": "last_L_p_prompt_plus_generated"
}
```

**15.3 消融配置**(`ablate.py`):

```json
{
  "band": [11, 24],
  "k": 10,
  "strength": "medium",
  "skip_clean_top": 10,
  "precision": "fp32",
  "control": "layer_matched_random",
  "seed": 0
}
```

**15.4 逐 trial 记录**(`log.jsonl` 每行):`run_id, phase, model_role, condition, ablation_state, lens_id, item_id, template_id, target, output_tokens, parsed, correct, kl_vs_clean, noop_bitexact, seal_hash`。

**15.5 题库条目**(`generator.py`):`item_id, generator_version, seed, n_steps, question, rationale, answer_int, answer_token_ids, single_token(bool)`。

### 16. Seal 与盲化(治理实现)

- `seal.py`:对 `PROTOCOL_v1.1.md`、`configs/freeze1/*`、`src/wda/**`、`tests/**`、题库生成器代码计算哈希清单写入 `FREEZE-1.json`;Phase B 后追加 `configs/freeze2/*`、lens 文件哈希、题库 seed、被试 revision 写入 `FREEZE-2.json`。每个 run 启动时校验清单,不匹配即拒绝启动。
- `blind.py`:Phase A 进程内禁止 `registry` 返回 `treatment` / `comparator` 角色;禁止 `stats.decision` 被调用;违反即抛异常并记入日志。
- `rerun_policy.py`:只识别注册的基础设施故障码;其他任何退出码不允许重跑同一 `run_id`。

### 17. 报告模板(每阶段)

1. 阶段与 seal 哈希
2. 被试与 revision
3. 每个 endpoint:点估计、CI、判定
4. parseability / conditional / ITT 三重表
5. 全部对照组结果(含 general-damage KL)
6. feasibility result 清单
7. 基础设施重跑日志
8. 触发的停止条件(若有)

---

## 附录 A · 前身仓库引用索引

`https://github.com/Alanjiao1988/J-space-observation`(head `d87c1b9e4e9dca062cebad7b6eee981d9dba8c25`)
- `README.md` — 总览;Study 1/2 诚实结论
- `PROJECT_DISCONTINUATION.md` — 项目级停止与 claim boundary
- `studies/README.md` — 六个 study 终态路由表
- `studies/study1/README.md` — 2/93、2/55、5/90
- `studies/study2/analysis/stage_bd_posthoc_interface_diagnostic.md` — C=0/384
- `studies/study3/README.md` — I0–I5;P0-T 两 token
- `studies/study3r/README.md` — RP-B 阶梯
- `studies/study4f/execution-m1/M1_FINAL_DISCLOSURE.md` — 7B/14B/32B CoT 与 E0;执行纪律
- `studies/study4f/analysis/study4f_interfaces.py` — W1/C1 envelope
- `studies/study5/closure/STUDY5_CLOSURE.md` — J≈αI、bf16、无阳性对照
- `studies/study5/qualification-eq2/TIMELINE.md` — 全层 readrate 扫描;band 规则缺陷
- `studies/study5/closure/STUDY5_HANDOFF.md` — bit-exact harness
- `third_party/jacobian-lens/581d398613e5602a5af361e1c34d3a92ea82ba8e/` — 上游 pin(仅 data)

## 附录 B · 论文参考数值(conformance 与效应量校准用,非目标)

workspace band ≈ L38–L92(重标定 0–100)· occupancy 平台 ≈ 25 · 超额解释方差 < 10% · concept vector J-space 分量占方差中位 6–7% · probe J-space 分量 ≈ 10–15% · verbal report swap:J-space 分量 59% top-5 / 纯 J-lens 88% / non-J 5% · probe-swap(n=90):J-space 分量 61% / 原始 swap 60% / non-J 28% / non-J 且 clamp 6% · multihop swap top-1:Haiku 4.5 54%,Sonnet 4.5 70%,Opus 4.5 70% · 中间量 swap 比答案 swap 提前 ≈ 17% 深度 · flexible generalization 76/192(α=1)、101/192(α=2) · capacity:不相关列表 ≈ 6 词、单层 1–2 词 · broadcast head ablation:recall@25 0.67 vs 0.86;top-1 改变 5% vs 2%;injected-thought 0.54 → 0.09 · ablation:k=10,三档层带;multihop 近天花板 → 近零;GSM8K CoT 显著比 direct 鲁棒

## 附录 C · 因果分支(不授权)

若 RQ1/RQ2 得到一致结果并希望升级为因果表述,唯一路径是受控配对训练:同一 base、同源同题数据、A 臂含推理轨迹 / B 臂仅答案 / C 臂不训练,其余训练参数逐项相同,再施加本方案 Phase B–C 全部测量。结论对象是"推理轨迹蒸馏这一过程",不是 DeepSeek 的特定 checkpoint。

## 附录 D · 外部来源

- 论文:https://transformer-circuits.pub/2026/workspace/index.html
- 上游代码:https://github.com/anthropics/jacobian-lens
- Anthropic 研究页:https://www.anthropic.com/research/global-workspace
- Neel Nanda review:LessWrong `zFJ3ZdQwrTWE9jT5S`
- https://github.com/tao-hpu/jspace-replication
- https://github.com/amaljithkuttamath/jlens-replication(`RESULTS.md`)
- arXiv:2608.25347
- DeepSeek-R1 官方模型表:https://github.com/deepseek-ai/DeepSeek-R1
- 社区公开 lens(校准参照):HF `Kameshr/jspace-lens-Qwen7B`

## 附录 E · 术语

- **G_primary**:Δ(C_direct) − Δ(C_frozen);ablation × response-format 交互
- **D**:两个 checkpoint 的 G_primary 之差;唯一 primary endpoint
- **δ**:最小有意义准确率差,先验给定
- **窗口 W**:最后 L_p 个 prompt 位置 + 全部生成位置;两主条件完全一致
- **ITT**:不可解析计为不正确
- **Freeze-1 / Freeze-2**:设计冻结 / 参数冻结
- **feasibility result**:可解析率低于阈值时的合法报告结局