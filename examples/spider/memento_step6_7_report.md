# Memento Step 6–7 工作报告（详细）

> 范围：在 Step 0–5 基础上完成 **StaticValidator 接入 check_query（Step 6）** 与 **ErrorNormalizer + Fix Hints 注入 rewrite_query（Step 7）**。保持 LangGraph 路由协议不变（仍依赖 `THE QUERY IS CORRECT/INCORRECT` 判断），不修改 prompt 文本，不增加 LLM 调用次数。

## 0. 约束对照（逐条自检）

- 不修改 `examples/spider/train_sql_agent.py`：✅ 未改动。
- 不修改 LangGraph 节点名：✅ 未改动。
- 不修改 WRITE/CHECK/REWRITE prompt 文本：✅ 未改动任何 prompt 常量。
- 不新增 LLM 调用：✅ check/rewrite 仍各自调用一次 LLM。
- `MEMENTO_ENABLE=0` 时行为不变：✅ 静态校验与 Fix Hint 注入均受 Memento gating。

## 1. Step 6：StaticValidator 接入 check_query

### 1.1 交付物
- 新增：`examples/spider/memory_module/validator.py`
- 修改：`examples/spider/sql_agent.py`
- 修改：`examples/spider/memory_module/runtime.py`
- 更新：`examples/spider/scripts/smoke_test_memento.py`
- 更新：`examples/spider/memory_module/README.md`

### 1.2 StaticValidator 设计

文件：`examples/spider/memory_module/validator.py`

接口：
- `validate(sql, table_info, dialect) -> ValidationResult`
  - `ok: bool`
  - `error_type: str`
  - `message: str`
  - `entities: dict`

策略：
- 从 `table_info` 中解析表/列信息（兼容 `CREATE TABLE ...` 与 `table: ... columns: ...` 简易格式）。
- 使用 `sqlglot.parse_one` 解析 SQL，获取表/列引用。
- 支持错误类型：
  - `MissingTable`
  - `MissingColumn`
  - `AmbiguousColumn`
  - `SyntaxError`（解析失败）
- 若 `sqlglot` 不可用：返回 `error_type="Unavailable"`，不阻断流程。

### 1.3 check_query 的接入方式

文件：`examples/spider/sql_agent.py`

- 仍先调用 LLM `CHECK_QUERY_PROMPT`（不改变调用次数与 prompt 文本）。
- 当 `MEMENTO_ENABLE=1` 且 `self.memento_runtime.validator` 可用时：
  - 调用 `validator.validate(...)`
  - 结果写入 `state["validation_error"]`
  - 若命中硬错误（MissingColumn/MissingTable/AmbiguousColumn/SyntaxError）：
    - 将反馈强制追加 `THE QUERY IS INCORRECT.`
    - 追加一条“静态校验”说明区块，确保 `should_continue` 仍按字符串路由

## 2. Step 7：rewrite_query 注入归一化错误 + Fix Hints

### 2.1 接入点

文件：`examples/spider/sql_agent.py`

在 `rewrite_query(state)` 内（仅 `MEMENTO_ENABLE=1`）：
- 从 `state["execution"]` 取 raw error
- `normalize_error(raw, dialect)` 得到 `error_type`
- 构造 `query_text`（error_type + raw + question + 可选 skeleton）
- `error_fix_bank.retrieve_fix_hints(...)` 获取 Fix Hints
- 将 Fix Hints 与静态校验摘要（若存在）追加到 `feedback` 末尾

注入格式：
- `### Retrieved Fix Hints (Do not copy SQL verbatim)`
- 每条 hint 编号列出
- 强约束：`You MUST use only columns/tables from Current Schema.`

保证：
- rewrite 输出协议不变（仍需输出 `REWRITTEN QUERY` block）
- 不新增 LLM 调用

## 3. Runtime 挂载

文件：`examples/spider/memory_module/runtime.py`

在 `MementoRuntime.build(config)` 中挂载：
- `runtime.validator = StaticValidator()`

## 4. Smoke 覆盖

文件：`examples/spider/scripts/smoke_test_memento.py`

新增验证：
- StaticValidator 对 `SELECT age FROM users` + 简单 schema 的缺失列检测。

## 5. 变更文件清单

新增：
- `examples/spider/memory_module/validator.py`
- `examples/spider/memento_step6_7_report.md`

修改：
- `examples/spider/sql_agent.py`
- `examples/spider/memory_module/runtime.py`
- `examples/spider/scripts/smoke_test_memento.py`
- `examples/spider/memory_module/README.md`

## 6. 自检与注意事项

- 未执行 pytest（环境缺少 pytest）；已通过 `py_compile` 校验相关文件语法。
- `check_query` 仍以 `THE QUERY IS CORRECT/INCORRECT` 字符串决定路由，保证 graph 协议不变。
- Memento 未启用时，静态校验与 Fix Hint 注入不生效。
