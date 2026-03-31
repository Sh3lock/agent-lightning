from __future__ import annotations

import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path
from unittest import mock
import unittest


def _load_train_sql_agent_module():
    spider_dir = Path(__file__).resolve().parents[1] / "examples/spider"
    module_path = spider_dir / "train_sql_agent.py"
    sys.path.insert(0, str(spider_dir))
    try:
        spec = importlib.util.spec_from_file_location("spider_train_sql_agent_for_test", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        fake_sql_agent = types.ModuleType("sql_agent")
        fake_sql_agent.LitSQLAgent = object
        fake_agl = types.ModuleType("agentlightning")
        fake_agl.__file__ = "/tmp/fake_agentlightning.py"
        fake_pandas = types.ModuleType("pandas")
        with mock.patch.dict(
            sys.modules,
            {
                "sql_agent": fake_sql_agent,
                "agentlightning": fake_agl,
                "pandas": fake_pandas,
            },
            clear=False,
        ):
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class TrainSqlAgentConfigTests(unittest.TestCase):
    def test_main_sets_ray_num_cpus_from_runner_budget(self) -> None:
        module = _load_train_sql_agent_module()
        captured: dict[str, object] = {}
        tmp_path = Path("/tmp/train_sql_agent_test_run")
        tmp_path.mkdir(parents=True, exist_ok=True)

        def fake_train(config, active_agent) -> None:
            captured["config"] = deepcopy(config)
            captured["active_agent"] = active_agent

        with (
            mock.patch.dict(module.os.environ, {"SPIDER_N_RUNNERS": "20"}, clear=False),
            mock.patch.object(module, "prepare_run_outputs", side_effect=lambda config, run_label: tmp_path),
            mock.patch.object(module, "_install_warning_error_logger", side_effect=lambda run_dir: None),
            mock.patch.object(module, "_install_raw_logger", side_effect=lambda run_dir: None),
            mock.patch.object(module, "_write_final_status", side_effect=lambda run_dir, status, message="": None),
            mock.patch.object(module, "train", side_effect=fake_train),
            mock.patch.object(sys, "argv", ["train_sql_agent.py", "local_qwen05"]),
        ):
            module.main()

        config = captured["config"]
        assert isinstance(config, dict)
        self.assertEqual(config.get("ray_init", {}).get("num_cpus"), 20)


if __name__ == "__main__":
    unittest.main()
