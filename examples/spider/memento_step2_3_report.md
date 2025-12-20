# Memento Step 2–3 工作报告（详细）

> 范围：基于 Step 0–1 已有的总开关/策略选择与 `MementoRuntime` 骨架，新增 **SQL Skeletonizer（Step 2）** 与 **CaseBank 存储/检索层（Step 3）**。本次改动仍保持“默认行为不变”：不修改任何 prompt，不接入 write/rewrite/check，不改变训练脚本接口与输出协议。

## 0. 约束对照（逐条自检）

- 不修改 `examples/spider/train_sql_agent.py`：✅ 未改动。
- 不改变 LangGraph 节点名：✅ 未改动 `write_query/execute_query/check_query/rewrite_query/should_continue`。
- 不修改 WRITE/CHECK/REWRITE prompt 文本：✅ 未改动任何 prompt 常量。
- 不增加 LLM 调用：✅ Skeletonizer/CaseBank 全部为本地逻辑。
- `MEMENTO_ENABLE=0` 默认路径不触发 skeletonizer/casebank 初始化/运行：✅ 只有当 Step 0–1 的 runtime 构建被触发（即 `MEMENTO_ENABLE=1`）时才会创建这些对象。

## 1. Step 2：SQL Skeletonizer（结构保真、跨库不泄漏）

### 1.1 交付物

- 新增：`examples/spider/memory_module/skeletonizer.py`
- 新增测试：`examples/spider/tests/test_skeletonizer.py`
- 新增说明：`examples/spider/memory_module/README.md`（补充安全策略说明）

### 1.2 设计目标与输出结构

实现 `SqlSkeletonizer`，提供：

- 输入：`sql: str`, `dialect: str`（默认 `sqlite`）
- 输出：`SkeletonResult`（dataclass）
  - `skeleton_sql: str`：去词汇化后的 SQL（table/column/literal 均替换）
  - `op_signature: str`：结构标签串（稳定字符串，便于后续 filter）
  - `entities: dict`：`tables/columns/literals` 映射信息（为后续 schema pruning 预留）
  - `failed: bool`：失败安全标记

### 1.3 核心实现（sqlglot AST + 确定性映射）

文件：`examples/spider/memory_module/skeletonizer.py`

#### (1) 解析与失败安全
- 使用 `sqlglot.parse_one(sql, read=dialect)` 解析 AST。
- 两类失败均 **不返回原 SQL**：
  - 环境缺少 `sqlglot`：`failed=True`, `op_signature="MISSING_SQLGLOT"`, `skeleton_sql=""`
  - SQL 解析失败：`failed=True`, `op_signature="PARSE_ERROR"`, `skeleton_sql=""`

这满足“失败安全、严禁返回原 SQL”以避免跨库泄漏风险。

#### (2) 确定性映射（同一 SQL 内一致）
- `table_map: original_table -> _tab1/_tab2/...`
- `table_alias_map: alias -> _tabN`（将别名也统一映射到 table placeholder）
- `col_map: qualified("t.c" 或 "c") -> _col1/_col2/...`
- `col_alias_map: alias -> _alias1/_alias2/...`（避免输出中出现原别名）
- literal：按类型替换为字符串字面量占位符（保持可读且不泄漏）
  - 数值：`'_val_num'`
  - 字符串：`'_val_str'`
  - 其它：`'_val_other'`
  - bool：`'_val_bool'`

> 注：literal 使用“字符串字面量”承载占位符，目的是在更多上下文中保持 SQL 的基本可读性与稳定性，同时不泄漏原值。

#### (3) AST 节点替换策略（只替换语义节点）
通过 `tree.transform(...)` 遍历替换关键表达式节点：
- `exp.Table`：替换表名与别名
- `exp.Column`：替换列名与（可用时）表限定符
- `exp.Alias`：替换别名标识符
- `exp.Literal` / `exp.Boolean`：替换字面量
- `exp.Subquery` / `exp.CTE`：对可见 alias 做 placeholder 化（避免泄漏子查询/CTE 名）

补充兜底断言（用于测试/调试）：
- `SqlSkeletonizer.assert_no_identifiers(original_sql, skeleton_sql, dialect)` 会解析原 SQL 并检查 skeleton 中是否残留标识符，若检测到泄漏直接抛出断言错误。

### 1.4 op_signature 生成逻辑

文件：`examples/spider/memory_module/skeletonizer.py`

- `op_signature` 以 `tree.key.upper()` 起始（例如 `SELECT`），并按 AST 是否包含以下结构追加标签：
  - `JOIN/GROUPBY/HAVING/ORDER/LIMIT/UNION/EXISTS/SUBQUERY/WINDOW`
- 输出为稳定字符串：`SELECT|JOIN|SUBQUERY|...`（去重但保序）

### 1.5 “不泄漏”保证点

- 表/列/字面量均被替换，不应在 `skeleton_sql` 中出现原始 table/column/value。
- 解析失败时返回空 skeleton，绝不回退为原 SQL。
- `entities` 中保留映射信息仅用于内部逻辑（后续步骤），不用于对外注入。

### 1.6 最小自检/单测

新增：`examples/spider/tests/test_skeletonizer.py`

- 用例 1：`SELECT name FROM users WHERE age > 18`
  - 断言 `failed=False`
  - 断言 skeleton 不包含 `users/name/age/18`
  - 断言 skeleton 含 `SELECT/FROM/WHERE`
- 用例 2（JOIN）：验证同一列多次引用占位符一致
  - 通过正则统计 `_col\d+` 出现次数，确保每个占位符至少出现 2 次（来自 `SELECT` 与 `ON`）

说明：当前环境未安装 `pytest`，因此本次无法在容器内执行 pytest；已至少通过 `python -m py_compile` 做语法级校验，且测试文件对 `sqlglot` 做了 `importorskip` 保护。

另外新增了一个可直接运行的 smoke 脚本（不依赖 pytest）：
- `examples/spider/scripts/smoke_test_memento.py`
  - 覆盖 Skeletonizer 不泄漏断言
  - 覆盖 CaseBank `skeleton_only`/`tiered` 策略分流

## 2. Step 3：CaseBank（双 collection + 分级检索 + 训练策略屏蔽）

### 2.1 交付物

- 新增：`examples/spider/memory_module/casebank.py`
- 新增：`examples/spider/memory_module/embedder.py`
- 新增：`examples/spider/memory_module/vector_store.py`
- 修改：`examples/spider/memory_module/runtime.py`（在 runtime 构建时挂载 casebank 与 skeletonizer）
- 新增测试：`examples/spider/tests/test_casebank.py`
- 更新说明：`examples/spider/memory_module/README.md`

### 2.2 对外 API 与返回结构

文件：`examples/spider/memory_module/casebank.py`

统一 API：

- `retrieve_tiered(question, db_id, dialect, policy, k, ...) -> RetrievalResult`
  - `policy`：
    - `skeleton_only`：只能返回 skeleton collection（永不返回 specific）
    - `tiered`：优先同库 specific（db_id filter），否则 skeleton
  - `RetrievalResult`：
    - `type: "specific" | "skeleton" | "none"`
    - `cases: list[{text, score, metadata}]`
    - `debug: dict`（包含命中原因、阈值与 top score 等）

阈值与“宁可不注入”：
- 支持 `min_score_specific`、`min_score_skeleton`
- top score 低于阈值直接返回 `type="none"`
默认阈值（在 `CaseBank.__init__` 固定）：
- `min_score_specific=0.35`
- `min_score_skeleton=0.30`

### 2.3 双 collection 的实现

文件：`examples/spider/memory_module/casebank.py`

强制两套 store：
- `memento_case_specific`：允许包含真实 SQL（同库 few-shot），仅 `tiered` 模式可返回
- `memento_case_skeleton`：全局 skeleton few-shot（训练与评估均可用）

写入接口：
- `add_specific(text, metadata)`：写入 specific
- `add_skeleton(text, metadata, forbidden_identifiers=...)`：
  - 可选安全断言：若 skeleton 文本包含禁用标识符（如真实表列名），直接抛错，防止误入库

### 2.4 向量库抽象与“可运行兜底后端”

文件：`examples/spider/memory_module/vector_store.py`

实现 `VectorStore` 抽象，并提供：
- `InMemoryVectorStore`：默认兜底（余弦/点积相似度，功能正确优先）
- `FaissVectorStore`：若环境可 import `faiss` 则可用（IP 相似度）
- `LazyVectorStore`：关键点：**延迟创建具体 store**，避免在 runtime 初始化阶段触发 embedding 模型加载

`create_vector_store(...)` 返回 `LazyVectorStore`，只有在第一次 `add_texts` 时才会决定使用 FAISS 还是 InMemory，并在需要时才会做一次 probe embedding 来确定维度。
评分语义统一为 cosine similarity（向量 L2 normalize + dot），保证“分数越大越相似”，便于阈值稳定。

### 2.5 Embedding：LazyEmbedder（仅在真正 add/query 时加载模型）

文件：`examples/spider/memory_module/embedder.py`

- `LazyEmbedder`：第一次 `embed_texts` 才构建真实 embedder
- 优先 `sentence-transformers`（默认模型 `all-MiniLM-L6-v2`，可通过 `MEMENTO_EMBEDDER_MODEL` 覆盖）
- 若 sentence-transformers 不可用或加载失败：自动回退 `HashingEmbedder`（轻量可运行）

这满足“默认路径不加载大模型；仅在启用且实际使用 casebank 时才初始化 embedder”的要求。

### 2.6 检索策略与训练防泄漏保证

文件：`examples/spider/memory_module/casebank.py`

- `policy="skeleton_only"`：
  - 仅查询 skeleton store（即使 specific 能命中也不会返回）
  - 返回 `type="skeleton"` 或 `type="none"`
- `policy="tiered"`：
  - 先查 specific（带 db_id filter）
  - 若命中且超过 `min_score_specific`：返回 `type="specific"`
  - 否则 fallback 查 skeleton（超过 `min_score_skeleton` 返回 `type="skeleton"`，否则 `none`）

### 2.7 最小自检/单测

新增：`examples/spider/tests/test_casebank.py`

- 构造假案例：
  - `db_id="db1"`, `question="how many users"`
  - specific 文本包含真实 SQL：`SELECT COUNT(*) FROM users`
  - skeleton 文本由 Step 2 skeletonizer 生成
- 验证：
  - `policy="skeleton_only"`：永不返回 `type="specific"`
  - `policy="tiered"` 且 db_id 相同：返回 `type="specific"`

同样地，本环境缺少 `pytest`，无法执行 pytest；已做 `py_compile` 校验。

## 3. Runtime 接入（仅在启用时初始化）

文件：`examples/spider/memory_module/runtime.py`

在 `MementoRuntime.build(config)` 中挂载：
- `runtime.casebank = CaseBank()`
- `runtime.skeletonizer = SqlSkeletonizer()`

说明：
- runtime 的构建本身只会创建“轻量句柄”；embedding/vector store 的重操作被推迟到首次 add/query。
- 由于 Step 0–1 的 gating 逻辑，只有当 `MEMENTO_ENABLE=1` 时才会构建 runtime，因此默认训练/评估行为不变。

## 4. 已知限制与后续工作

- Skeletonizer 目前覆盖了 Spider 常见结构（SELECT/JOIN/CTE/SUBQUERY 等），但对极端方言/复杂表达式的“完全去标识符”仍可在后续迭代补强（例如更全面的 Identifier 场景）。
- CaseBank 目前只提供存储/检索 API，不接入 prompt（符合本 Step 约束）；后续 Step 4 才会把检索结果注入 write_query。
- 当前环境缺少 `pytest`，建议在本机安装后执行：
  - `python -m pytest -q examples/spider/tests/test_skeletonizer.py examples/spider/tests/test_casebank.py`

## 5. 文件变更清单

新增：
- `examples/spider/memory_module/skeletonizer.py`
- `examples/spider/memory_module/casebank.py`
- `examples/spider/memory_module/embedder.py`
- `examples/spider/memory_module/vector_store.py`
- `examples/spider/memory_module/README.md`
- `examples/spider/tests/test_skeletonizer.py`
- `examples/spider/tests/test_casebank.py`
- `examples/spider/scripts/smoke_test_memento.py`

修改：
- `examples/spider/memory_module/runtime.py`
