# Memento Step 4–5 工作报告（详细）

> 范围：在 Step 0–3 的基础上完成 **CaseBank 注入 write_query（Step 4）** 与 **ErrorNormalizer/ErrorFixBank 骨架（Step 5）**。仍保持训练入口与输出协议不变，不修改 prompt 文本，不新增 LLM 调用。

## 0. 约束对照（逐条自检）

- 不修改 `examples/spider/train_sql_agent.py`：✅ 未改动。
- 不修改 LangGraph 节点名：✅ 未改动。
- 不修改 WRITE/CHECK/REWRITE prompt 文本：✅ 未改动任何 prompt 常量。
- 不新增 LLM 调用：✅ 新增逻辑全部为本地检索与字符串拼接。
- `MEMENTO_ENABLE=0` 时行为不变：✅ 仅在 runtime 启用时才注入检索结果。

## 1. Step 4：CaseBank 分级检索接入 write_query

### 1.1 交付物
- 修改：`examples/spider/sql_agent.py`

### 1.2 接入位置与行为

在 `SQLAgent.write_query(state)` 内：
- 当且仅当 `self.memento_config.enable` 且存在 `self.memento_runtime.casebank` 时执行检索。
- 调用 `casebank.retrieve_tiered(...)`：
  - `question=state["question"]`
  - `db_id=self.db_id`（由 rollout 时写入）
  - `dialect=self.db.dialect`
  - `policy=self.memento_policy`（空则回退 eval_policy）
  - `k=4`
- 若 `db_id` 缺失：直接跳过检索并记录 `memento_retrieval` 原因，避免无效 filter 行为。
- 根据返回 `type` 构造 `memory_context`：
  - `specific`：标题 `### Relevant Past Cases (Same Database)`，允许引用命名但必须来自 CURRENT SCHEMA
  - `skeleton`：标题 `### Structural References (Different Database)`，明确禁止拷贝命名
  - `none`：不注入

### 1.3 长度预算与 schema 保护

新增 helper：`_build_table_info_with_memory_context(...)`，策略如下：
- 固定总上限 `max_chars`（沿用 `self.table_info_truncate`）。
- 先构造 `### Current Schema` + 原始 table_info。
- 在剩余预算内截断 memory_context（宁可少注入，也不丢 schema）。

最终 prompt 仍使用原变量接口：
```
WRITE_QUERY_PROMPT.invoke({dialect, input, table_info})
```

### 1.4 训练防泄漏与 debug 字段

- `policy=skeleton_only` 时即使同库命中也不会注入 specific（由 CaseBank 保证）。
- 追加调试字段：`state["memento_retrieval"] = retrieval.debug`，不影响输出协议。

### 1.5 rollout 中 db_id 注入

在 `LitSQLAgent.rollout` 内，实例化 `SQLAgent` 后：
- `sql_agent.db_id = task.get("db_id")`
用于 `retrieve_tiered` 的 db_id filter。

## 2. Step 5：ErrorNormalizer + ErrorFixBank（仅实现与 runtime 挂载）

### 2.1 交付物

- 新增：`examples/spider/memory_module/error_normalizer.py`
- 新增：`examples/spider/memory_module/error_fix_bank.py`
- 修改：`examples/spider/memory_module/runtime.py`
- 更新：`examples/spider/memory_module/README.md`
- 更新：`examples/spider/scripts/smoke_test_memento.py`

### 2.2 ErrorNormalizer

文件：`examples/spider/memory_module/error_normalizer.py`

实现 `normalize_error(raw, dialect)`：
- 正则覆盖 SQLite/Spider 常见报错：
  - `no such column: X` → `MissingColumn`
  - `no such table: T` → `MissingTable`
  - `ambiguous column name: X` → `AmbiguousColumn`
  - `near "xxx": syntax error` → `SyntaxError`
  - `no such function: F` → `NoSuchFunction`
  - `datatype mismatch` → `TypeMismatch`
  - `group by` 相关 → `GroupByError`
- 输出 `NormalizedError(error_type, entities, raw)`，实体按 columns/tables/tokens/functions 分类。

### 2.3 ErrorFixBank

文件：`examples/spider/memory_module/error_fix_bank.py`

功能：
- 向量库检索 Fix Hint，支持 `error_type` 与 `dialect` 过滤。
- `add_fix_hint(text, metadata)`：写入自定义 hint。
- `retrieve_fix_hints(error_type, dialect, query_text, k, min_score)`：返回过滤后的高相似 hint。
- 内置一组 seed hints（MissingColumn/MissingTable/Ambiguous/Syntax），首次使用时懒加载写入。
- embedding 仍采用 LazyEmbedder，缺失 sentence-transformers 时自动回退 HashingEmbedder。

### 2.4 Runtime 挂载

文件：`examples/spider/memory_module/runtime.py`

在 `MementoRuntime.build(config)` 中挂载：
- `runtime.error_fix_bank = ErrorFixBank()`

## 3. Smoke 脚本更新

文件：`examples/spider/scripts/smoke_test_memento.py`

新增验证项：
- 预算函数 `_build_table_info_with_memory_context` 保证 schema 不丢。
- `normalize_error("...no such column: flight_id")` 输出 `MissingColumn` 且包含实体。
- ErrorFixBank 添加 hint 后检索可命中。

该脚本可在无 pytest 环境直接运行。

## 4. 变更文件清单

新增：
- `examples/spider/memory_module/error_normalizer.py`
- `examples/spider/memory_module/error_fix_bank.py`
- `examples/spider/memento_step4_5_report.md`

修改：
- `examples/spider/sql_agent.py`
- `examples/spider/memory_module/runtime.py`
- `examples/spider/scripts/smoke_test_memento.py`
- `examples/spider/memory_module/README.md`

## 5. 自检与注意事项

- 未执行 pytest（环境缺少 pytest）；已通过 `py_compile` 校验核心文件语法。
- write_query 输出协议保持不变：仍返回 `state["query"]`，且 prompt 变量接口不变。
- Memento 未启用时不触发检索与注入逻辑，默认路径一致。
