# Memento 功能开发总工作报告（Step 0–9 全量总览）

> 目的：在不修改 `examples/spider/train_sql_agent.py` 的前提下，为 Spider Text-to-SQL 示例引入“可控启用”的 Memento 记忆增强能力（检索 few-shot、静态校验、错误修复提示、离线索引构建与评测），并保证默认关闭时行为与原版一致。

---

## 1. 项目目标与硬约束（全程遵守）

- **训练入口不变**：不修改 `examples/spider/train_sql_agent.py` 的调用方式与接口。
- **Graph 协议不变**：不修改 LangGraph 节点名：`write_query/execute_query/check_query/rewrite_query/should_continue`。
- **Prompt 常量不变**：不修改 `WRITE_QUERY_PROMPT` / `CHECK_QUERY_PROMPT` / `REWRITE_QUERY_PROMPT` 文本与变量接口。
- **LLM 调用次数不变**：write/check/rewrite 各自仍是一次 LLM 调用，不新增额外 LLM 调用。
- **默认行为完全一致**：`MEMENTO_ENABLE=0` 时，不触发任何 memory_module 初始化与检索/注入逻辑；输出结构保持一致：
  - `LitSQLAgent.rollout()` 仍返回 `reward: float | None`
  - `agent.invoke()` 输入仍为 `{"question": question}`
  - 最终 SQL 写入 `result["query"]`

---

## 2. 总体架构概览（从 Step 0 到 Step 9）

本次开发将“记忆增强”拆为三层：

1) **开关与策略层（Step 0）**：环境变量控制是否启用，以及 train/eval 的策略分流（`skeleton_only` vs `tiered`）。

2) **运行时容器层（Step 1）**：`MementoRuntime` 作为统一入口，进程级单例懒加载；只在启用时创建，默认路径完全不加载。

3) **能力组件层（Step 2–7）**：
- Skeletonizer：跨库结构参考（去标识符、失败安全）
- CaseBank：specific/skeleton 双库检索 + 分级策略 + 训练防泄漏
- StaticValidator：静态校验覆盖 LLM 漏判
- ErrorNormalizer + ErrorFixBank：归一化错误与修复提示检索

4) **离线数据与评测层（Step 8–9）**：
- build_casebanks：从 Spider parquet 离线构建 two collections（含 exec_verified 门禁、增量、manifest）
- eval_sql_agent：baseline vs memento 对比评测，输出诊断指标
- train_smoke：训练模式 smoke（强调防泄漏与入口兼容）

---

## 3. Step 0：总开关与训练/评估策略选择（默认不改变任何行为）

涉及文件：`examples/spider/sql_agent.py`

### 3.1 新增配置读取（无侵入）
- 环境变量：
  - `MEMENTO_ENABLE`（默认 `0`）
  - `MEMENTO_TRAIN_POLICY`（默认 `skeleton_only`）
  - `MEMENTO_EVAL_POLICY`（默认 `tiered`）
- 策略校验：仅允许 `skeleton_only/tiered`，非法值回退默认并 warning。

### 3.2 rollout 内策略选择与传递
- `LitSQLAgent.rollout(..., rollout.mode)`：
  - `mode == "train"` → `train_policy`
  - 否则 → `eval_policy`
- 将 policy 与 config 挂到 `SQLAgent` 实例字段：`memento_config/memento_policy`（不更改训练脚本调用方式）。

---

## 4. Step 1：MementoRuntime 懒加载骨架（进程级单例）

新增目录：`examples/spider/memory_module/`
- `runtime.py`：
  - `MementoRuntime` 容器（占位字段：casebank/skeletonizer/error_fix_bank/validator 等）
  - `get_memento_runtime()`：模块级单例，第一次使用才初始化

接入点：`examples/spider/sql_agent.py`
- 通过 `_maybe_init_memento_runtime(config)` 进行懒加载 import，仅在 enable=1 时触发。

保证：`MEMENTO_ENABLE=0` 时不 import、不初始化、不改变执行路径。

---

## 5. Step 2：SQL Skeletonizer（结构保真、跨库不泄漏）

新增：`examples/spider/memory_module/skeletonizer.py`

### 5.1 关键实现
- 解析：`sqlglot.parse_one(sql, read=dialect)`
- 替换：Table/Column/Alias/Literal 统一替换为占位符（同一 SQL 内一致映射）
  - table → `_tab1/_tab2/...`
  - column → `_col1/_col2/...`（qualified 与 unqualified 都有确定性键）
  - literal → 按类型 `_val_num/_val_str/...`
- 失败安全：解析失败或缺少 sqlglot → `failed=True` 且 `skeleton_sql=""`（严禁回退原 SQL）
- `op_signature`：稳定结构标签串（便于后续 filter）
- `assert_no_identifiers(original_sql, skeleton_sql, dialect)`：测试/调试兜底，防遗漏泄漏面。

### 5.2 自检
- `examples/spider/tests/test_skeletonizer.py`
- `examples/spider/scripts/smoke_test_memento.py`

---

## 6. Step 3：CaseBank（双 collection、分级检索、训练防泄漏）

新增：
- `examples/spider/memory_module/casebank.py`
- `examples/spider/memory_module/vector_store.py`
- `examples/spider/memory_module/embedder.py`

### 6.1 two collections（强制）
- `memento_case_specific`：同库 few-shot（可含真实 SQL）
- `memento_case_skeleton`：全局 skeleton few-shot（严禁真实标识符）

### 6.2 统一 API：retrieve_tiered
- `retrieve_tiered(question, db_id, dialect, policy, k)` 返回：
  - `type: specific|skeleton|none`
  - `cases: [{text, score, metadata}]`
  - `debug: {reason, min_score_*, specific_score/skeleton_score, ...}`

### 6.3 策略与防泄漏
- `policy=skeleton_only`：永远不返回 specific（即使命中）
- `policy=tiered`：先查 specific（db_id filter），不足则 fallback skeleton

### 6.4 阈值默认值（稳定可控）
- 固化默认阈值：
  - `min_score_specific=0.35`
  - `min_score_skeleton=0.30`

### 6.5 向量后端与评分语义统一
- InMemory / FAISS / Lazy 统一：
  - 向量 L2 normalize
  - `score = dot = cosine similarity`（越大越相似）

### 6.6 自检
- `examples/spider/tests/test_casebank.py`
- `examples/spider/scripts/smoke_test_memento.py`

---

## 7. Step 4：CaseBank 接入 write_query（动态 few-shot 注入，不改输出协议）

修改：`examples/spider/sql_agent.py`

### 7.1 接入方式
- 仅当 enable=1 且 runtime.casebank 可用时：调用 `retrieve_tiered(...)`。
- 注入位置：不改 prompt 变量接口，将 memory_context 拼到 `table_info` 前置块：
  - `### Relevant Past Cases (Same Database)`：允许复用命名但必须来自 CURRENT SCHEMA
  - `### Structural References (Different Database)`：明确禁止拷贝命名，仅作结构参考
  - `type=none`：不注入

### 7.2 长度预算与 schema 保护
- `_build_table_info_with_memory_context(...)`：优先保留 schema，再按预算截断 memory_context。

### 7.3 db_id 兜底
- 若 `self.db_id` 缺失：直接跳过检索，并写 debug reason，避免错误 metadata filter。

---

## 8. Step 5：ErrorNormalizer + ErrorFixBank（只提供 API 与 runtime 挂载）

新增：
- `examples/spider/memory_module/error_normalizer.py`
- `examples/spider/memory_module/error_fix_bank.py`

### 8.1 ErrorNormalizer
- 基于 regex 归一化常见 SQLite/Spider 错误：MissingColumn/MissingTable/Ambiguous/Syntax/NoSuchFunction/TypeMismatch/GroupByError 等。

### 8.2 ErrorFixBank
- 存储并检索“修复策略提示（非完整 SQL）”，按 `error_type`/`dialect` 过滤。
- 内置 seed hints（覆盖 MissingColumn/MissingTable/Ambiguous/Syntax 等）。
- embedding 懒加载，sentence-transformers 不可用则 hashing 兜底。

---

## 9. Step 6：StaticValidator 接入 check_query（保持路由协议不变）

新增：`examples/spider/memory_module/validator.py`
- 从 `table_info` 提取 schema + `sqlglot` 解析 SQL，检测：MissingTable/MissingColumn/AmbiguousColumn/SyntaxError。

修改：`examples/spider/sql_agent.py`
- check_query 仍先调用 LLM（不减少调用次数）。
- 若静态校验命中硬错误：
  - **替换结论**：移除 LLM feedback 中所有 `THE QUERY IS CORRECT/INCORRECT` 结论行
  - 追加 `### Static Validation Findings`
  - 最后一行严格写入 `THE QUERY IS INCORRECT.`（只出现一次）
- 写入调试字段：`state["validation_error"]`、`state["llm_feedback_raw"]`

目的：避免 feedback 同时包含 CORRECT 与 INCORRECT 导致 should_continue 误路由。

---

## 10. Step 7：rewrite_query 注入归一化错误 + Fix Hints（不改输出协议）

修改：`examples/spider/sql_agent.py`
- 仅 enable=1 时：
  - 从 execution 提取 raw error → normalize_error
  - 构造 query_text（error_type + raw + question + 可选 failing_sql_skeleton）
  - 调用 error_fix_bank 检索 hints
  - 将 hints 追加到 `{feedback}` 字段末尾（不改 prompt 变量接口）
- 记录 `state["normalized_error_type"]` 便于评测统计。

---

## 11. Step 8：离线构建 CaseBank 索引（高价值轨迹门禁、增量、可复现）

新增：`examples/spider/scripts/build_casebanks.py`

### 11.1 输入/输出
- 输入：Spider parquet（question/db_id/query）
- 输出：`specific.jsonl/specific.npy` 与 `skeleton.jsonl/skeleton.npy`

### 11.2 高价值门禁
- `exec_verified`：在对应 db 上执行 GT SQL 成功才入库
- skeletonizer `failed=True` 不入库 skeleton
- skeleton 文本通过 `assert_no_identifiers` 做泄漏检查

### 11.3 增量与去重
- `case_id = sha1(f"{db_id}\n{question}\n{sql}")`（显式分隔符，避免歧义）
- `--mode {skip,upsert}` 控制重复写入行为

### 11.4 并发与缓存
- `--workers N` 并行 exec 验证
- 线程内 db_cache 复用（减少反复打开 DB）
- exec_cache：同 `(db_id, sql_hash)` 不重复执行

### 11.5 manifest.json（工程级可复现）
- 在 persist_dir 写入 `manifest.json`，包含：
  - `casebank_format_version`
  - `created_at/split/source/counts/exec_stats`
  - embedder 元信息（type/model/dim/normalize）
  - similarity 定义（cosine）

运行时加载（`CaseBank`）：
- 读取 manifest 并基于 manifest 选择 embedder；
- 对版本/模型/维度/similarity/normalize/split 不一致：warning；
- 设置 `MEMENTO_CASEBANK_STRICT=1` 时：不一致直接报错，防止 silent mismatch。

---

## 12. Step 9：端到端评测 + 训练 smoke（baseline 对齐与诊断指标）

新增：
- `examples/spider/scripts/eval_sql_agent.py`
- `examples/spider/scripts/train_smoke.py`

### 12.1 eval_sql_agent 指标体系
- Exec@1（reward=1 占比）
- avg_turns
- avg_prompt_chars / avg_query_chars
- validation_counts
- retrieval_counts（specific/skeleton/none）
- **best_score 分布**（specific/skeleton 的 top score 统计）
- **StaticValidator 覆盖率**：LLM 判 CORRECT 但 validator 判错的比例
- **rewrite 成功率分解**：按 `normalized_error_type` 统计成功率

### 12.2 train_smoke 验证
- 强制：`MEMENTO_ENABLE=1` + `MEMENTO_TRAIN_POLICY=skeleton_only`
- 统计 retrieval type 分布 + 泄漏检测（训练时不得出现 specific）
- import `train_sql_agent` 作为入口兼容 smoke（不改训练脚本）。

---

## 13. 核心环境变量（默认关闭，开启才生效）

- `MEMENTO_ENABLE`：0/1，总开关（默认 0）
- `MEMENTO_TRAIN_POLICY`：训练策略（默认 skeleton_only）
- `MEMENTO_EVAL_POLICY`：评估策略（默认 tiered）
- `MEMENTO_CASEBANK_DIR`：CaseBank 索引目录
- `MEMENTO_CASEBANK_ALLOW_SPLIT`：允许检索的 split（默认 train）
- `MEMENTO_CASEBANK_STRICT`：manifest 强一致性（1 时不一致直接报错）
- `MEMENTO_EMBEDDER_MODEL`：sentence-transformers 模型名（可选）

---

## 14. 全量变更文件清单（便于总体 review）

### 14.1 新增文件
- `examples/spider/memory_module/__init__.py`
- `examples/spider/memory_module/runtime.py`
- `examples/spider/memory_module/config.py`
- `examples/spider/memory_module/skeletonizer.py`
- `examples/spider/memory_module/casebank.py`
- `examples/spider/memory_module/vector_store.py`
- `examples/spider/memory_module/embedder.py`
- `examples/spider/memory_module/validator.py`
- `examples/spider/memory_module/error_normalizer.py`
- `examples/spider/memory_module/error_fix_bank.py`
- `examples/spider/memory_module/README.md`
- `examples/spider/scripts/smoke_test_memento.py`
- `examples/spider/scripts/build_casebanks.py`
- `examples/spider/scripts/eval_sql_agent.py`
- `examples/spider/scripts/train_smoke.py`
- `examples/spider/tests/test_skeletonizer.py`
- `examples/spider/tests/test_casebank.py`
- `examples/spider/memento_step0_1_report.md`
- `examples/spider/memento_step2_3_report.md`
- `examples/spider/memento_step4_5_report.md`
- `examples/spider/memento_step6_7_report.md`
- `examples/spider/memento_step8_9_report.md`

### 14.2 修改文件
- `examples/spider/sql_agent.py`（开关/策略、runtime 接入、write/check/rewrite 注入、调试统计字段）
- `examples/spider/memory_module/runtime.py`（挂载 skeletonizer/casebank/validator/error_fix_bank 等）
- `examples/spider/memory_module/README.md`（使用说明与可复现要素）

---

## 15. 快速复现与自检（无 pytest 环境）

1) 组件 smoke：
- `python examples/spider/scripts/smoke_test_memento.py`

2) 离线索引构建：
- `python examples/spider/scripts/build_casebanks.py --input data/train_spider.parquet --split train --workers 4`

3) baseline vs memento 评测：
- `python examples/spider/scripts/eval_sql_agent.py --input data/test_dev.parquet --mode baseline`
- `python examples/spider/scripts/eval_sql_agent.py --input data/test_dev.parquet --mode memento`

4) 训练模式 smoke（防泄漏）：
- `python examples/spider/scripts/train_smoke.py --input data/train_spider.parquet --limit 20`

---

## 16. 结论

本次开发以“默认关闭、启用才生效”为总原则，逐步搭建了可扩展的 MementoRuntime，并实现了 Skeletonizer/CaseBank/Validator/FixHints 的端到端闭环；同时补齐离线索引构建（含 manifest 可复现要素）与评测/训练 smoke 工具链，便于后续迭代与稳定回归。
