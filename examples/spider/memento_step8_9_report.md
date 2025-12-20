# Memento Step 8–9 工作报告（详细）

> 范围：实现 **CaseBank 离线构建与增量更新（Step 8）** 以及 **端到端评测 + 训练 smoke（Step 9）**。保持训练入口不变，不修改 prompt 文本与在线输出协议。

## 0. 约束对照（逐条自检）

- 不修改 `examples/spider/train_sql_agent.py`：✅ 未改动。
- 不修改 LangGraph 节点名：✅ 未改动。
- 不修改 WRITE/CHECK/REWRITE prompt 文本：✅ 未改动任何 prompt 常量。
- 不新增 LLM 调用：✅ 新增脚本与离线构建均为本地逻辑。
- `MEMENTO_ENABLE=0` 时行为不变：✅ CaseBank 仅在运行时启用时加载。

## 1. Step 8：离线构建 CaseBank 索引

### 1.1 交付物
- 新增：`examples/spider/scripts/build_casebanks.py`
- 修改：`examples/spider/memory_module/casebank.py`
- 修改：`examples/spider/memory_module/vector_store.py`
- 更新：`examples/spider/memory_module/README.md`

### 1.2 脚本能力与 CLI

脚本：`examples/spider/scripts/build_casebanks.py`

功能：
- 输入 Spider parquet（包含 `question/db_id/query`）。
- 通过数据库执行 GT SQL 进行 `exec_verified` 门禁。
- 构建 two collections：
  - `memento_case_specific`：同库 full SQL（允许真实表列名）
  - `memento_case_skeleton`：跨库 skeleton（禁止真实标识符）
- skeletonizer 解析失败或识别泄漏的样本不会入库。
- 支持增量与去重：通过 `case_id = sha1(db_id + question + sql)`。
- `--mode {skip, upsert}` 控制重复处理。
- `--build_specific/--build_skeleton` 控制构建哪类 collection。
- `--dry_run` 输出统计但不写盘。

持久化格式：
- `specific.jsonl` / `skeleton.jsonl`（text + metadata）
- `specific.npy` / `skeleton.npy`（embedding 向量）

### 1.3 “高价值轨迹”门禁

- 仅当 SQL 在对应 db 上执行成功时入库（无报错 -> exec_verified）。
- skeletonizer `failed=True` 的样本不写入 skeleton collection。
- skeleton document 通过 `assert_no_identifiers` 做泄漏检查。

### 1.4 Runtime 加载与增量更新

文件：`examples/spider/memory_module/casebank.py`

- 新增 `persist_dir` 支持（默认读取 `MEMENTO_CASEBANK_DIR`）。
- `retrieve_tiered` 首次调用触发 `_ensure_loaded()`：
  - 若目录存在则读 `jsonl` + `npy`。
  - 优先加载 `npy` 向量（减少 embedder 开销）。
- 新增 `add_texts_with_embeddings(...)` 支持向量直写（`VectorStore` 接口扩展）。

### 1.5 向量库存取统一

文件：`examples/spider/memory_module/vector_store.py`

- 增加 `add_texts_with_embeddings`，支持离线向量加载。
- InMemory/FAISS/Lazy 统一向量 L2 normalize，score 为 cosine similarity。

## 2. Step 9：评测 + 训练 smoke 脚本

### 2.1 评测脚本

新增：`examples/spider/scripts/eval_sql_agent.py`

能力：
- baseline vs memento 模式切换：
  - `baseline`：`MEMENTO_ENABLE=0`
  - `memento`：`MEMENTO_ENABLE=1` + `MEMENTO_EVAL_POLICY=tiered`
- 输出 JSON + JSONL + CSV 指标
- 指标包含：
  - Exec@1（reward=1 占比）
  - avg_turns
  - avg_prompt_chars / avg_query_chars
  - validation_counts
  - retrieval_counts（specific/skeleton/none）

### 2.2 训练 smoke 脚本

新增：`examples/spider/scripts/train_smoke.py`

能力：
- 强制设置 `MEMENTO_ENABLE=1` + `MEMENTO_TRAIN_POLICY=skeleton_only`
- 用小样本跑完整 write→execute→check→rewrite 循环
- 统计 `retrieval_counts`，并记录 `leakage_detected`（是否出现 specific）
- 通过 import `train_sql_agent` 做入口兼容性 smoke

## 3. 运行示例

构建索引：
```
python examples/spider/scripts/build_casebanks.py --input data/train_spider.parquet --split train
```

评测：
```
python examples/spider/scripts/eval_sql_agent.py --input data/test_dev.parquet --mode baseline
python examples/spider/scripts/eval_sql_agent.py --input data/test_dev.parquet --mode memento
```

训练 smoke：
```
python examples/spider/scripts/train_smoke.py --input data/train_spider.parquet --limit 20
```

## 4. 变更文件清单

新增：
- `examples/spider/scripts/build_casebanks.py`
- `examples/spider/scripts/eval_sql_agent.py`
- `examples/spider/scripts/train_smoke.py`
- `examples/spider/memento_step8_9_report.md`

修改：
- `examples/spider/memory_module/casebank.py`
- `examples/spider/memory_module/vector_store.py`
- `examples/spider/memory_module/README.md`
- `examples/spider/sql_agent.py`（增加 `prompt_table_info_chars` 统计字段）

## 5. 自检与注意事项

- 未执行 pytest（环境缺少 pytest）；已通过 `py_compile` 校验相关文件语法。
- CaseBank 索引文件为本地持久化，运行时通过 `MEMENTO_CASEBANK_DIR` 加载。
- 训练 smoke 仅验证流程与泄漏检测，不代表收敛效果。
