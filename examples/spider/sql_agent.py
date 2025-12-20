# Copyright (c) Microsoft. All rights reserved.

"""Sample code that demonstrates an SQL agent using LangGraph and LangChain,
trainable with Agent-lightning.

Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, cast

import pandas as pd
import termcolor
from langchain.chat_models import init_chat_model
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from spider_eval.exec_eval import eval_exec_match

import agentlightning as agl

agl.setup_logging(apply_to=[__name__])

logger = logging.getLogger(__name__)

MEMENTO_POLICY_SKELETON_ONLY = "skeleton_only"
MEMENTO_POLICY_TIERED = "tiered"
MEMENTO_VALID_POLICIES = {MEMENTO_POLICY_SKELETON_ONLY, MEMENTO_POLICY_TIERED}
DEFAULT_MEMENTO_TRAIN_POLICY = MEMENTO_POLICY_SKELETON_ONLY
DEFAULT_MEMENTO_EVAL_POLICY = MEMENTO_POLICY_TIERED


@dataclass(frozen=True)
class MementoConfig:
    enable: bool
    train_policy: str
    eval_policy: str


def _validate_memento_policy(value: str, env_key: str, default: str) -> str:
    if value in MEMENTO_VALID_POLICIES:
        return value
    logger.warning("Invalid %s=%s. Falling back to %s.", env_key, value, default)
    return default


def read_memento_config() -> MementoConfig:
    enable = os.environ.get("MEMENTO_ENABLE", "0") == "1"
    train_policy = os.environ.get("MEMENTO_TRAIN_POLICY", DEFAULT_MEMENTO_TRAIN_POLICY)
    eval_policy = os.environ.get("MEMENTO_EVAL_POLICY", DEFAULT_MEMENTO_EVAL_POLICY)
    train_policy = _validate_memento_policy(train_policy, "MEMENTO_TRAIN_POLICY", DEFAULT_MEMENTO_TRAIN_POLICY)
    eval_policy = _validate_memento_policy(eval_policy, "MEMENTO_EVAL_POLICY", DEFAULT_MEMENTO_EVAL_POLICY)
    return MementoConfig(enable=enable, train_policy=train_policy, eval_policy=eval_policy)


def _maybe_init_memento_runtime(config: MementoConfig) -> Any:
    if not config.enable:
        return None
    from memory_module.runtime import get_memento_runtime as _get_memento_runtime

    return _get_memento_runtime(config)


def _build_table_info_with_memory_context(
    memory_context: str,
    original_table_info: str,
    max_chars: int,
) -> str:
    if not memory_context:
        return original_table_info
    schema_block = "### Current Schema\n" + original_table_info
    available = max_chars - len(schema_block) - 2
    if available <= 0:
        return schema_block
    trimmed = memory_context.strip()
    if len(trimmed) > available:
        trimmed = trimmed[:available].rstrip()
    if not trimmed:
        return schema_block
    return f"{trimmed}\n\n{schema_block}"


def _append_static_validation_feedback(feedback: str, message: str) -> str:
    trimmed = _strip_query_conclusion(feedback).rstrip()
    if trimmed:
        trimmed += "\n\n"
    trimmed += "### Static Validation Findings\n"
    trimmed += f"- {message}\n"
    trimmed += "THE QUERY IS INCORRECT."
    return trimmed


def _strip_query_conclusion(feedback: str) -> str:
    lines = []
    for line in feedback.splitlines():
        if line.strip() in {"THE QUERY IS CORRECT.", "THE QUERY IS INCORRECT."}:
            continue
        lines.append(line)
    return "\n".join(lines)
WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
""".strip(),
        ),
        ("user", "Question: {input}"),
    ]
)


CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasoable query execution results

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the list above are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.
""".strip(),
        ),
        (
            "user",
            """Question: {input}

Query:

```{dialect}
{query}
```

Execution result:

```
{execution}
```""",
        ),
    ]
)


REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        (
            "system",
            """
You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```
""".strip(),
        ),
        (
            "user",
            """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback.""",
        ),
    ]
)


class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class SQLAgent:

    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: Dict[str, Any] | None = None,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ):
        self.db_id: str | None = None
        self.memento_config: MementoConfig | None = None
        self.memento_policy: str | None = None
        self.memento_runtime: Any | None = None
        self.db = SQLDatabase.from_uri(db)  # type: ignore
        self.db_schema = db_schema
        self.debug = debug
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        if verl_replacement is not None:
            self.model_name: str = verl_replacement["model"]  # type: ignore
            assert endpoint is not None
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint,
                openai_api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                temperature=verl_replacement["temperature"],
                max_retries=0,
                max_tokens=2048,
            )
        else:
            self.model_name: str = os.environ.get("MODEL", "gpt-4.1-mini")
            self.llm = init_chat_model(
                self.model_name,
                model_provider="openai",
                openai_api_base=endpoint or os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0,
                max_retries=1,
                max_tokens=2048,
            )

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                table_info = table_info[: self.table_info_truncate] + "\n... (truncated)"
            return table_info
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            if self.db_schema:
                if len(self.db_schema) > self.table_info_truncate:
                    return self.db_schema[: self.table_info_truncate] + "\n... (truncated)"
                return self.db_schema
            return "No schema available."

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")

        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Failed to invoke prompt: {e}")
            # FIXME: fallback to create a random trajectory
            result = self.llm.invoke([HumanMessage(content="Please create a random SQL query as an example.")])

        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")

        return result  # type: ignore

    def truncate_execution(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: AnyMessage) -> str | None:
        result: str | None = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):  # type: ignore
            result = match.group(1).strip()  # type: ignore
        return result  # type: ignore

    def write_query(self, state: State) -> State:
        """Generate SQL query to fetch information."""
        table_info = self.get_table_info()
        retrieval_debug = None
        if (
            self.memento_config
            and self.memento_config.enable
            and self.memento_runtime
            and getattr(self.memento_runtime, "casebank", None)
        ):
            policy = self.memento_policy or self.memento_config.eval_policy
            db_id = self.db_id
            if not db_id:
                retrieval_debug = {"reason": "missing_db_id"}
            else:
                retrieval = self.memento_runtime.casebank.retrieve_tiered(
                    question=state["question"],
                    db_id=db_id,
                    dialect=self.db.dialect,
                    policy=policy,
                    k=4,
                )
                retrieval_debug = retrieval.debug
                if retrieval.type == "specific":
                    cases = "\n\n".join(
                        f"Case {idx + 1}:\n{case.text}" for idx, case in enumerate(retrieval.cases)
                    )
                    memory_context = (
                        "### Relevant Past Cases (Same Database)\n"
                        "You may reuse table/column names ONLY if they appear in CURRENT SCHEMA.\n\n"
                        f"{cases}"
                    )
                    table_info = _build_table_info_with_memory_context(
                        memory_context,
                        table_info,
                        self.table_info_truncate,
                    )
                elif retrieval.type == "skeleton":
                    cases = "\n\n".join(
                        f"Case {idx + 1}:\n{case.text}" for idx, case in enumerate(retrieval.cases)
                    )
                    memory_context = (
                        "### Structural References (Different Database)\n"
                        "DO NOT copy table/column names from these examples. Use only CURRENT SCHEMA.\n\n"
                        f"{cases}"
                    )
                    table_info = _build_table_info_with_memory_context(
                        memory_context,
                        table_info,
                        self.table_info_truncate,
                    )

        prompt: Any = WRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "table_info": table_info,
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        query = self.parse_query(result) or result.content  # type: ignore

        next_state = {  # type: ignore
            **state,
            "query": query,  # type: ignore
            "num_turns": 1,
            "messages": [*prompt.messages, result],
            "prompt_table_info_chars": len(table_info),
        }
        if retrieval_debug is not None:
            next_state["memento_retrieval"] = retrieval_debug
        return next_state  # type: ignore

    def execute_query(self, state: State) -> State:
        """Execute SQL query."""
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        execution_result = execute_query_tool.invoke(state["query"])  # type: ignore
        if not isinstance(execution_result, str):
            # Convert to string if it's not already
            execution_result = str(execution_result)
        if self.debug:
            termcolor.cprint(execution_result, "yellow")
        return {**state, "execution": execution_result}

    def check_query(self, state: State) -> State:
        """Check the SQL query for correctness."""
        table_info = self.get_table_info()
        prompt: Any = CHECK_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execution(state["execution"]),
                "table_info": table_info,
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore
        feedback = result.content  # type: ignore
        llm_feedback_raw = feedback

        validation_error = None
        messages = [*state.get("messages", []), *prompt.messages, result]
        if (
            self.memento_config
            and self.memento_config.enable
            and self.memento_runtime
            and getattr(self.memento_runtime, "validator", None)
        ):
            validation = self.memento_runtime.validator.validate(
                state["query"],
                table_info,
                self.db.dialect,
            )
            validation_error = {
                "ok": validation.ok,
                "error_type": validation.error_type,
                "message": validation.message,
                "entities": validation.entities,
            }
            if not validation.ok and validation.error_type in {
                "MissingColumn",
                "MissingTable",
                "AmbiguousColumn",
                "SyntaxError",
            }:
                feedback = _append_static_validation_feedback(feedback, validation.message)
                messages.append(AIMessage(content=feedback))

        res = {  # type: ignore
            **state,
            "feedback": feedback,  # type: ignore
            "messages": messages,
            "llm_feedback_raw": llm_feedback_raw,
        }
        if validation_error is not None:
            res["validation_error"] = validation_error
        return res  # type: ignore

    def rewrite_query(self, state: State) -> State:
        """Rewrite SQL query if necessary."""
        feedback = state["feedback"]
        if (
            self.memento_config
            and self.memento_config.enable
            and self.memento_runtime
            and getattr(self.memento_runtime, "error_fix_bank", None)
        ):
            from memory_module.error_normalizer import normalize_error

            normalized = normalize_error(state.get("execution", ""), self.db.dialect)
            query_parts = [normalized.error_type, normalized.raw]
            question = state.get("question")
            if question:
                query_parts.append(question)
            skeleton_sql = None
            if getattr(self.memento_runtime, "skeletonizer", None):
                skeleton_result = self.memento_runtime.skeletonizer.skeletonize(
                    state["query"],
                    self.db.dialect,
                )
                if not skeleton_result.failed:
                    skeleton_sql = skeleton_result.skeleton_sql
                    query_parts.append(skeleton_sql)
            hints = []
            if normalized.error_type not in {"Other", "Unavailable"}:
                query_text = "\n".join(part for part in query_parts if part)
                hints = self.memento_runtime.error_fix_bank.retrieve_fix_hints(
                    error_type=normalized.error_type,
                    dialect=self.db.dialect,
                    query_text=query_text,
                    k=4,
                    min_score=0.30,
                )
            extra_sections = []
            validation_error = state.get("validation_error")
            if validation_error:
                extra_sections.append(
                    "### Static Validation Summary\n"
                    f"- {validation_error.get('error_type')}: {validation_error.get('message')}"
                )
            if hints:
                hint_lines = ["### Retrieved Fix Hints (Do not copy SQL verbatim)"]
                for idx, hint in enumerate(hints, start=1):
                    hint_lines.append(f"{idx}. {hint.text}")
                hint_lines.append("You MUST use only columns/tables from Current Schema.")
                extra_sections.append("\n".join(hint_lines))
            if extra_sections:
                feedback = feedback.rstrip() + "\n\n" + "\n\n".join(extra_sections)
            normalized_error_type = normalized.error_type
        else:
            normalized_error_type = None

        prompt: Any = REWRITE_QUERY_PROMPT.invoke(  # type: ignore
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execution(state["execution"]),
                "feedback": feedback,
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)  # type: ignore

        rewritten_query = self.parse_query(result)  # type: ignore

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],  # clear previous prompts
            "normalized_error_type": normalized_error_type,
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:  # type: ignore
        """Determine if the agent should continue based on the result."""
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):  # type: ignore
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:  # type: ignore
                if "THE QUERY IS INCORRECT" in last_message.content:  # type: ignore
                    # Both correct and incorrect messages found
                    # See which is the last one
                    correct_index = last_message.content.rfind("THE QUERY IS CORRECT")  # type: ignore
                    incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")  # type: ignore
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            return END

        return "rewrite_query"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)
        builder.add_node(self.write_query)  # type: ignore
        builder.add_node(self.execute_query)  # type: ignore
        builder.add_node(self.check_query)  # type: ignore
        builder.add_node(self.rewrite_query)  # type: ignore

        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,  # type: ignore
        )
        builder.add_edge("rewrite_query", "execute_query")

        return builder.compile()  # type: ignore


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    # TODO(yuge): Maybe we can evaluate intermediate queries and assign more precise rewards.

    # included in the original evaluation script
    # query = query.replace("value", "1")

    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # Parameters following the default setting
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        if exec_score == 1:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0


class LitSQLAgent(agl.LitAgent[Dict[str, Any]]):

    def __init__(
        self,
        trained_agents: Optional[str] = r"write",
        val_temperature: Optional[float] = None,
        max_turns: int = 3,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ) -> None:
        super().__init__(trained_agents=trained_agents)
        self.val_temperature = val_temperature
        self.spider_dir = os.environ.get("VERL_SPIDER_DATA_DIR", "data")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate

    def rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = task["question"]
        start_time = time.time()
        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        memento_config = read_memento_config()
        runtime_policy = None
        if memento_config.enable:
            runtime_policy = (
                memento_config.train_policy if rollout.mode == "train" else memento_config.eval_policy
            )
        rollout_id = rollout.rollout_id
        if memento_config.enable:
            logger.debug(
                "[Rollout %s] Memento enabled (policy=%s, mode=%s).",
                rollout_id,
                runtime_policy,
                rollout.mode,
            )

        if rollout.mode == "train":
            original_db_path = os.path.join(self.spider_dir, "database", task["db_id"], task["db_id"] + ".sqlite")
        else:
            original_db_path = os.path.join(self.spider_dir, "test_database", task["db_id"], task["db_id"] + ".sqlite")
        ground_truth = task["query"]

        if not os.path.exists(original_db_path):
            logger.error(f"Database {original_db_path} does not exist. Skipping.")
            return None

        schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = f.read()
        else:
            logger.error("Schema file not found: %s", schema_path)
            schema = "No schema available."

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            logger.info(f"[Rollout {rollout_id}] Question: {question}")
            logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

            # Run the agent
            sql_agent = SQLAgent(
                "sqlite:///" + db_path,
                max_turns=self.max_turns,
                table_info_truncate=self.table_info_truncate,
                execution_truncate=self.execution_truncate,
                debug=False,
                db_schema=schema,
                endpoint=llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),  # type: ignore
                verl_replacement=(
                    {"model": llm.model, **llm.sampling_parameters}
                    if rollout.mode == "train"
                    else {
                        "model": llm.model,
                        "temperature": (
                            self.val_temperature
                            if self.val_temperature is not None
                            else llm.sampling_parameters.get("temperature", 0.0)
                        ),
                    }
                ),
            )
            sql_agent.memento_runtime = None
            sql_agent.memento_policy = None
            sql_agent.memento_config = memento_config
            sql_agent.db_id = task.get("db_id")
            if memento_config.enable:
                sql_agent.memento_policy = runtime_policy
                sql_agent.memento_runtime = _maybe_init_memento_runtime(memento_config)
            agent = sql_agent.graph()
            try:
                # Required to make the langchain tracing work
                handler = self.tracer.get_langchain_handler()
                result = agent.invoke(  # type: ignore
                    {"question": question},  # type: ignore
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
            except Exception as e:
                logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
                return

            logger.info(f"[Rollout {rollout_id}] Generated Query: {result['query']}")

        end_time_rollout = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)

            reward = evaluate_query(result["query"], ground_truth, db_path, raise_on_error=False)
            logger.info("[Rollout %s] Reward: %s", rollout_id, reward)

        end_time_eval = time.time()

        logger.info("[Rollout %s] Time taken for rollout: %.2f seconds", rollout_id, end_time_rollout - start_time)
        logger.info(
            "[Rollout %s] Time taken for evaluation: %.2f seconds", rollout_id, end_time_eval - end_time_rollout
        )

        return reward


def debug_sql_agent():
    spider_dev_data_path = os.path.join(os.environ.get("VERL_SPIDER_DATA_DIR", "data"), "dev.parquet")
    if not os.path.exists(spider_dev_data_path):
        raise FileNotFoundError(f"Spider dev data file {spider_dev_data_path} does not exist.")
    df = pd.read_parquet(spider_dev_data_path).head(10)  # type: ignore
    df = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore
    print("Debug data:", df)

    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=os.environ["OPENAI_API_BASE"],
                model="gpt-4.1-nano",
                sampling_parameters={"temperature": 0.7},
            )
        },
    )
    trainer.dev(LitSQLAgent(), df)


if __name__ == "__main__":
    debug_sql_agent()
