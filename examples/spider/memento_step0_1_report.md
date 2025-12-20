# Memento Step 0–1 工作报告（详细）

> 目标：在不修改 `train_sql_agent.py` 的前提下，为 Spider Text-to-SQL 示例加入 “Memento 记忆增强” 的总开关与训练/评估策略选择逻辑（Step 0），并搭建进程级懒加载的 `MementoRuntime` 框架（Step 1）。
>
> 关键前提：默认不开启（`MEMENTO_ENABLE` 未设置或不为 `1`）时，系统行为必须与当前完全一致（prompt、循环、SQL 输出格式、reward 计算逻辑都不变）。

## 1. 修改范围与严格约束对照

### 1.1 未触碰/保证不变的部分
- 未修改 `examples/spider/train_sql_agent.py` 的任何调用方式与接口。
- 未修改 LangGraph 节点名：`write_query/execute_query/check_query/rewrite_query/should_continue`。
- `LitSQLAgent.rollout()` 仍只返回 `reward: float | None`；不引入额外返回结构。
- `agent.invoke()` 输入仍为 `{"question": question}`。
- 最终 SQL 仍写在 `result["query"]`，并用于 `evaluate_query()` 计算 reward。
- 未修改三段 prompt：`WRITE_QUERY_PROMPT` / `CHECK_QUERY_PROMPT` / `REWRITE_QUERY_PROMPT` 的任何内容。
- 未增加新的 LLM 调用（Memento 只引入配置解析与对象挂载）。

### 1.2 新增能力（仅在开启时可见）
- 新增环境变量总开关：`MEMENTO_ENABLE`（默认关闭）。
- 新增策略配置：
  - `MEMENTO_TRAIN_POLICY`（默认 `skeleton_only`）
  - `MEMENTO_EVAL_POLICY`（默认 `tiered`）
- rollout 时策略选择：`rollout.mode == "train"` 用 train policy；否则用 eval policy。
- Step 1：新增 `MementoRuntime` 占位容器 + 进程级单例懒加载（只搭框架，不改变任何 prompt/输出）。

## 2. Step 0：总开关与策略选择（默认行为不变）

### 2.1 新增配置结构与读取逻辑
文件：`examples/spider/sql_agent.py`

新增常量与数据结构：
- `MEMENTO_POLICY_SKELETON_ONLY = "skeleton_only"`
- `MEMENTO_POLICY_TIERED = "tiered"`
- `MEMENTO_VALID_POLICIES = {"skeleton_only", "tiered"}`
- 默认值：
  - `DEFAULT_MEMENTO_TRAIN_POLICY = "skeleton_only"`
  - `DEFAULT_MEMENTO_EVAL_POLICY = "tiered"`
- `@dataclass(frozen=True) class MementoConfig:`
  - `enable: bool`
  - `train_policy: str`
  - `eval_policy: str`

新增读取函数：`read_memento_config()`
- 读取 env：
  - `MEMENTO_ENABLE`：仅当值为 `"1"` 时视为开启，其余均视为关闭（默认 `"0"`）。
  - `MEMENTO_TRAIN_POLICY`：默认 `skeleton_only`
  - `MEMENTO_EVAL_POLICY`：默认 `tiered`
- 策略校验：`_validate_memento_policy(value, env_key, default)`
  - 仅允许 `skeleton_only/tiered`
  - 非法值：回退默认值，并写入 `logger.warning(...)`

说明：在默认环境（未设置这些 env）下，不会触发任何 warning 日志；仅多了一次轻量 env 读取，不影响执行路径与输出。

### 2.2 rollout 内策略选择与可见性
文件：`examples/spider/sql_agent.py`

在 `LitSQLAgent.rollout(...)` 开始处：
- 调用 `memento_config = read_memento_config()`
- 若 `memento_config.enable` 为真：
  - `runtime_policy = memento_config.train_policy if rollout.mode == "train" else memento_config.eval_policy`
  - 记录一条 debug 日志（用于验收“policy 生效可见，但默认不刷屏”）：
    - `"[Rollout {rollout_id}] Memento enabled (policy=..., mode=...)." `

### 2.3 policy 传递与存储位置（不改外部调用）
文件：`examples/spider/sql_agent.py`

由于 `train_sql_agent.py` 不能改动，本 Step 采用 “在 rollout 内部创建 SQLAgent 实例后挂载字段” 的方式，把 policy 信息保存到 SQLAgent 实例可访问位置：
- `sql_agent.memento_config = memento_config`
- `sql_agent.memento_policy = runtime_policy`（仅开启时非空）
- `sql_agent.memento_runtime = ...`（Step 1，见后文）

同时，为避免动态属性影响类型检查/可读性，在 `SQLAgent.__init__` 中显式初始化：
- `self.memento_config: MementoConfig | None = None`
- `self.memento_policy: str | None = None`
- `self.memento_runtime: Any | None = None`

> 注意：本 Step 不改变 graph 的节点、边、prompt、返回结构；`agent = sql_agent.graph()` 之后仍按原逻辑 `agent.invoke({"question": question}, ...)`，并使用 `result["query"]` 做评估。

## 3. Step 1：MementoRuntime 懒加载骨架（只搭框架，不改变行为）

### 3.1 新增 memory_module 目录结构
新增目录：`examples/spider/memory_module/`
- `__init__.py`：模块占位说明
- `runtime.py`：定义 `MementoRuntime` 与懒加载单例
- `config.py`：Step 0 配置结构的备用实现（当前未被 `sql_agent.py` 引用；用于后续把配置统一下沉到 memory_module 时使用）

### 3.2 进程级懒加载单例：get_memento_runtime()
文件：`examples/spider/memory_module/runtime.py`

实现要点：
- `class MementoRuntime:` 只做轻量初始化，包含后续组件的占位字段（均为 `None`）：
  - `embedder`
  - `casebank`
  - `error_fix_bank`
  - `skeletonizer`
  - `schema_pruner`
  - `validator`
- `@classmethod build(cls, config)`：当前等价于 `return cls(config)`，为后续扩展预留工厂入口。
- 模块级缓存 `_RUNTIME: Optional[MementoRuntime] = None`
- `get_memento_runtime(config)`：
  - 第一次调用创建 `_RUNTIME = MementoRuntime.build(config)` 并 `logger.info("Initialized MementoRuntime (config=...).")`
  - 后续调用直接返回同一实例（进程级复用）
  - 若同进程后续传入 config 与首次不一致：打印 warning（不刷新 runtime，只提示风险）

### 3.3 sql_agent.py 中的懒加载接入（只在 enable=1 时触发）
文件：`examples/spider/sql_agent.py`

新增桥接函数：`_maybe_init_memento_runtime(config: MementoConfig) -> Any`
- 当 `config.enable` 为假：直接返回 `None`
- 当 `config.enable` 为真：在函数内部才执行 `from memory_module.runtime import get_memento_runtime as _get_memento_runtime`
  - 这样保证 `MEMENTO_ENABLE=0` 时不会 import `memory_module.runtime`，从而满足“默认完全不加载任何新模块、不改变执行路径”的要求。

在 rollout 中，当 `memento_config.enable` 为真时：
- `sql_agent.memento_runtime = _maybe_init_memento_runtime(memento_config)`
- 该 runtime 在同一进程内只初始化一次；多次 rollout 复用同一实例。

> 当前 runtime 单例缓存的是首次启用时传入的 config；如果后续在同一进程内变更 env 导致 config 变化，runtime 仍会复用首次实例（这是预期的“进程级 runtime”语义，后续如需支持动态刷新可再扩展）。

## 4. 行为一致性检查（默认不变）

### 4.1 默认环境（未设置 env）
- `read_memento_config()` 读取到：`enable=False, train_policy="skeleton_only", eval_policy="tiered"`
- 不会：
  - import `memory_module.runtime`
  - 初始化 runtime
  - 输出任何 “Memento enabled / Initialized MementoRuntime” 的日志
- rollout 的 prompt、循环、SQL 生成、reward 计算与原版一致。

### 4.2 开启环境（`MEMENTO_ENABLE=1`）
- rollout 中会额外看到：
  - `"[Rollout X] Memento enabled (policy=..., mode=...)." `
  - 第一次触发 runtime 时：`"Initialized MementoRuntime (config=...)." `
- 不会：
  - 改变 prompt 文本
  - 注入检索结果
  - 改变输出 SQL 格式、reward 逻辑

## 5. 交付物清单（实际改动）

### 5.1 修改文件
- `examples/spider/sql_agent.py`
  - 新增 MementoConfig 与 env 读取/校验
  - rollout 内策略选择与日志
  - SQLAgent 实例字段挂载：`memento_config/memento_policy/memento_runtime`
  - 懒加载 runtime 的桥接函数 `_maybe_init_memento_runtime(...)`

### 5.2 新增文件
- `examples/spider/memory_module/__init__.py`
- `examples/spider/memory_module/runtime.py`
- `examples/spider/memory_module/config.py`（当前未接入，仅为后续统一配置位置预留）
- `examples/spider/memento_step0_1_report.md`（本报告）

## 6. 建议的最小自检方式（不改变训练脚本）

### 6.1 仅验证“不开启不生效”
在同一终端中确保未设置 env（或显式：`set MEMENTO_ENABLE=` / `Remove-Item Env:MEMENTO_ENABLE`），运行原有训练/验证流程，确认日志中不出现：
- `Memento enabled`
- `Initialized MementoRuntime`

### 6.2 验证 train/eval policy 分流
设置：
- `setx MEMENTO_ENABLE 1`
- `setx MEMENTO_TRAIN_POLICY skeleton_only`
- `setx MEMENTO_EVAL_POLICY tiered`

运行训练与验证（或触发不同 `rollout.mode`），检查日志：
- train rollout：`policy=skeleton_only`
- val/test rollout：`policy=tiered`

## 7. 后续扩展建议（面向 Step 2+）
- 建议把“组件注入点”集中放到 `MementoRuntime`：
  - Step 2：casebank 检索/注入
  - Step 3：error_fix_bank
  - Step 4：skeletonizer（将 question/sql skeleton 化）
  - Step 5：schema_pruner（schema 裁剪）
  - Step 6：validator（更强校验）
- 所有新行为必须继续遵循：仅当 `MEMENTO_ENABLE=1` 时生效，并确保默认行为不变。
