from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock
import unittest


def _load_entrypoint_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "agentlightning/verl/entrypoint.py"
    spec = importlib.util.spec_from_file_location(
        "agentlightning.verl.entrypoint_for_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "agentlightning.verl"

    fake_hydra = types.ModuleType("hydra")
    fake_hydra.main = lambda **_: (lambda fn: fn)

    fake_ray = types.ModuleType("ray")
    fake_ray.is_initialized = lambda: False
    fake_ray.init = lambda **kwargs: kwargs
    fake_ray.get = lambda obj: obj
    fake_ray.remote = lambda *args, **kwargs: (lambda obj: obj)

    fake_ray_actor = types.ModuleType("ray.actor")
    fake_ray_actor.ActorClass = object

    fake_main_ppo = types.ModuleType("verl.trainer.main_ppo")
    fake_main_ppo.create_rl_sampler = lambda *args, **kwargs: None

    fake_reward = types.ModuleType("verl.trainer.ppo.reward")
    fake_reward.load_reward_manager = lambda *args, **kwargs: None

    fake_adapter = types.ModuleType("agentlightning.adapter")
    fake_adapter.TraceAdapter = object

    fake_llm_proxy = types.ModuleType("agentlightning.llm_proxy")
    fake_llm_proxy.LLMProxy = object

    fake_store = types.ModuleType("agentlightning.store.base")
    fake_store.LightningStore = object

    fake_types = types.ModuleType("agentlightning.types")
    fake_types.Dataset = list

    fake_dataset = types.ModuleType("agentlightning.verl.dataset")
    fake_dataset.AgentDataset = object
    fake_dataset.LoadedDataset = object

    sys.path.insert(0, str(project_root))
    try:
        with mock.patch.dict(
            sys.modules,
            {
                "hydra": fake_hydra,
                "ray": fake_ray,
                "ray.actor": fake_ray_actor,
                "verl.trainer.main_ppo": fake_main_ppo,
                "verl.trainer.ppo.reward": fake_reward,
                "agentlightning.adapter": fake_adapter,
                "agentlightning.llm_proxy": fake_llm_proxy,
                "agentlightning.store.base": fake_store,
                "agentlightning.types": fake_types,
                "agentlightning.verl.dataset": fake_dataset,
            },
            clear=False,
        ):
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class EntryPointRayOverrideTests(unittest.TestCase):
    def test_build_local_ray_param_overrides_defaults_dashboard_port_to_dynamic(self) -> None:
        module = _load_entrypoint_module()
        with mock.patch.dict(module.os.environ, {}, clear=False):
            overrides = module._build_local_ray_param_overrides()
        self.assertEqual(overrides["dashboard_agent_listen_port"], 0)

    def test_wrap_ray_params_cls_injects_override_without_clobbering_explicit_value(self) -> None:
        module = _load_entrypoint_module()
        calls: list[dict[str, int]] = []

        class FakeRayParams:
            def __init__(self, *args, **kwargs):
                calls.append(kwargs)

        wrapped = module._wrap_ray_params_cls_with_overrides(
            FakeRayParams,
            {"dashboard_agent_listen_port": 0, "metrics_agent_port": 41001},
        )

        wrapped()
        wrapped(dashboard_agent_listen_port=52365)

        self.assertEqual(calls[0]["dashboard_agent_listen_port"], 0)
        self.assertEqual(calls[0]["metrics_agent_port"], 41001)
        self.assertEqual(calls[1]["dashboard_agent_listen_port"], 52365)
        self.assertEqual(calls[1]["metrics_agent_port"], 41001)


if __name__ == "__main__":
    unittest.main()
