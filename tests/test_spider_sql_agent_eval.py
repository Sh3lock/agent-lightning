from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class _FakePromptTemplate:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def invoke(self, values):
        return values


class _FakeStateGraph:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def add_node(self, *args, **kwargs) -> None:
        pass

    def add_edge(self, *args, **kwargs) -> None:
        pass

    def add_conditional_edges(self, *args, **kwargs) -> None:
        pass

    def compile(self):
        return object()


class _FakeGenericBase:
    def __class_getitem__(cls, item):
        return cls


def _load_spider_module(module_filename: str, module_name: str):
    spider_dir = Path(__file__).resolve().parents[1] / "examples/spider"
    module_path = spider_dir / module_filename
    sys.path.insert(0, str(spider_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        fake_pandas = types.ModuleType("pandas")
        fake_termcolor = types.ModuleType("termcolor")
        fake_termcolor.cprint = lambda *args, **kwargs: None

        fake_chat_models = types.ModuleType("langchain.chat_models")
        fake_chat_models.init_chat_model = lambda *args, **kwargs: object()

        fake_callbacks = types.ModuleType("langchain_core.callbacks")
        fake_callbacks.BaseCallbackHandler = object

        fake_sql_tool = types.ModuleType("langchain_community.tools.sql_database.tool")
        fake_sql_tool.QuerySQLDatabaseTool = object

        fake_sql_utils = types.ModuleType("langchain_community.utilities")
        fake_sql_utils.SQLDatabase = object

        fake_messages = types.ModuleType("langchain_core.messages")
        fake_messages.AnyMessage = object
        fake_messages.BaseMessage = object
        fake_messages.HumanMessage = object

        fake_prompts = types.ModuleType("langchain_core.prompts")
        fake_prompts.ChatPromptTemplate = _FakePromptTemplate

        fake_graph = types.ModuleType("langgraph.graph")
        fake_graph.END = "END"
        fake_graph.START = "START"
        fake_graph.MessagesState = dict
        fake_graph.StateGraph = _FakeStateGraph

        fake_graph_state = types.ModuleType("langgraph.graph.state")
        fake_graph_state.CompiledStateGraph = object

        fake_exec_eval = types.ModuleType("spider_eval.exec_eval")
        fake_exec_eval.eval_exec_match = lambda **kwargs: 0.0

        fake_agl = types.ModuleType("agentlightning")
        fake_agl.LitAgent = _FakeGenericBase
        fake_agl.NamedResources = object
        fake_agl.Rollout = object
        fake_agl.LLM = object
        fake_agl.setup_logging = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {
                "pandas": fake_pandas,
                "termcolor": fake_termcolor,
                "langchain.chat_models": fake_chat_models,
                "langchain_core.callbacks": fake_callbacks,
                "langchain_community.tools.sql_database.tool": fake_sql_tool,
                "langchain_community.utilities": fake_sql_utils,
                "langchain_core.messages": fake_messages,
                "langchain_core.prompts": fake_prompts,
                "langgraph.graph": fake_graph,
                "langgraph.graph.state": fake_graph_state,
                "spider_eval.exec_eval": fake_exec_eval,
                "agentlightning": fake_agl,
            },
            clear=False,
        ):
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class SqlAgentEvalHelpersTests(unittest.TestCase):
    def test_extract_safe_retry_max_tokens_from_context_window_error(self) -> None:
        module = _load_spider_module("sql_agent_eval.py", "spider_sql_agent_eval_for_test")
        error = (
            "ContextWindowExceededError: Hosted_vllmException - "
            "{\"error\":{\"message\":\"'max_tokens' or 'max_completion_tokens' is too large: 2048. "
            "This model's maximum context length is 6144 tokens and your request has 5359 input tokens "
            "(2048 > 6144 - 5359). None\"}}"
        )

        safe_max_tokens = module._extract_safe_retry_max_tokens(error)

        self.assertEqual(safe_max_tokens, 769)

    def test_extract_safe_retry_max_tokens_returns_none_for_unrelated_error(self) -> None:
        module = _load_spider_module("sql_agent_eval.py", "spider_sql_agent_eval_for_test")

        safe_max_tokens = module._extract_safe_retry_max_tokens("some other error")

        self.assertIsNone(safe_max_tokens)


class SqlAgentHelpersTests(unittest.TestCase):
    def test_extract_safe_retry_max_tokens_from_context_window_error(self) -> None:
        module = _load_spider_module("sql_agent.py", "spider_sql_agent_for_test")
        error = (
            "ContextWindowExceededError: Hosted_vllmException - "
            "{\"error\":{\"message\":\"'max_tokens' or 'max_completion_tokens' is too large: 2048. "
            "This model's maximum context length is 6144 tokens and your request has 5359 input tokens "
            "(2048 > 6144 - 5359). None\"}}"
        )

        safe_max_tokens = module._extract_safe_retry_max_tokens(error)

        self.assertEqual(safe_max_tokens, 769)


if __name__ == "__main__":
    unittest.main()
