# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys
from types import ModuleType

import pytest

from resources_servers.osworld.eval_task import (
    _apply_task_reward_contract,
    _install_python_controller_compat,
    _register_rlvr_evaluators,
)


def test_python_controller_run_command_compat_proxies_execute(monkeypatch):
    controllers = ModuleType("desktop_env.controllers")
    python_module = ModuleType("desktop_env.controllers.python")

    class PythonController:
        http_server = "http://guest.test"
        retry_times = 1
        retry_interval = 0

    python_module.PythonController = PythonController
    monkeypatch.setitem(sys.modules, "desktop_env.controllers", controllers)
    monkeypatch.setitem(sys.modules, "desktop_env.controllers.python", python_module)

    class Response:
        status_code = 200
        text = '{"status":"success","output":"ok","error":"","returncode":0}'

        @staticmethod
        def json():
            return {
                "status": "success",
                "output": "ok",
                "error": "",
                "returncode": 0,
            }

    post_calls = []

    def post(url, *, json, timeout):
        post_calls.append((url, json, timeout))
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    _install_python_controller_compat()

    result = PythonController().run_command(["python3", "-c", "print('ok')"])

    assert result["returncode"] == 0
    assert result["output"] == "ok"
    assert post_calls == [
        (
            "http://guest.test/execute",
            {
                "command": ["python3", "-c", "print('ok')"],
                "shell": False,
            },
            120,
        )
    ]


def test_register_rlvr_evaluators_loads_only_current_frozen_task(monkeypatch, tmp_path):
    task_root = tmp_path / "tmp_funcs" / "task-1"
    for role in ("getters", "metrics"):
        role_root = task_root / role
        role_root.mkdir(parents=True)
        (role_root / "selftest_provenance.py").write_text(
            "SENTINEL = 'not an evaluator plugin'\n"
        )
        function_name = "get_custom_task_1" if role == "getters" else "check_custom_task_1"
        if role == "getters":
            (role_root / "helper.py").write_text(
                "def preserve(value):\n    return value\n"
            )
            body = (
                "from .helper import preserve\n"
                f"def {function_name}(value=None):\n    return preserve(value)\n"
            )
        else:
            body = f"def {function_name}(value=None):\n    return value\n"
        (role_root / "custom.py").write_text(body)

    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    getters = ModuleType("desktop_env.evaluators.getters")
    metrics = ModuleType("desktop_env.evaluators.metrics")
    evaluators.getters = getters
    evaluators.metrics = metrics
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.getters", getters)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.setenv("OSWORLD_RLVR_SNAPSHOT", str(tmp_path))

    task = {
        "id": "task-1",
        "evaluator": {
            "func": "check_custom_task_1",
            "result": {"type": "custom_task_1"},
        },
        "metadata": {
            "evaluator_routes": [
                {"route": "getters:get_custom_task_1"},
                {"route": "metrics:check_custom_task_1"},
            ]
        },
    }
    _register_rlvr_evaluators(task)

    assert getattr(getters, f"get_{task['evaluator']['result']['type']}")("getter") == "getter"
    assert getattr(metrics, task["evaluator"]["func"])("metric") == "metric"


def test_register_rlvr_evaluators_accepts_external_functions_root(
    monkeypatch, tmp_path
):
    snapshot = tmp_path / "snapshot"
    functions = tmp_path / "prepared-pool" / "tmp_funcs"
    snapshot.mkdir()
    task_id = "task-1"
    task_root = functions / task_id
    for role in ("getters", "metrics"):
        role_root = task_root / role
        role_root.mkdir(parents=True)
        (role_root / "custom.py").write_text(
            f"def custom_{role}(value=None):\n    return value\n"
        )

    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    getters = ModuleType("desktop_env.evaluators.getters")
    getters.__path__ = []
    metrics = ModuleType("desktop_env.evaluators.metrics")
    metrics.__path__ = []
    evaluators.getters = getters
    evaluators.metrics = metrics
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.getters", getters)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.setenv("OSWORLD_RLVR_SNAPSHOT", str(snapshot))
    monkeypatch.setenv("OSWORLD_RLVR_FUNCTIONS_ROOT", str(functions))
    monkeypatch.delenv("OSWORLD_EVALUATOR_REGISTRY_ROOT", raising=False)

    _register_rlvr_evaluators({"id": task_id})

    assert getters.custom_getters("getter") == "getter"
    assert metrics.custom_metrics("metric") == "metric"


def test_register_rlvr_evaluators_loads_shared_lazy_registry(monkeypatch, tmp_path):
    registry = tmp_path / "registry"
    for role, symbol, module in (
        ("getters", "get_custom_result__abc", "m_getter"),
        ("metrics", "check_custom_result__abc", "m_metric"),
    ):
        role_root = registry / role / "verigen"
        role_root.mkdir(parents=True)
        (role_root / "__init__.py").write_text(
            f"_MAP = {{{symbol!r}: {module!r}}}\n"
        )
        (role_root / f"{module}.py").write_text(
            f"def {symbol}(value=None):\n    return value\n"
        )

    desktop_env = ModuleType("desktop_env")
    evaluators = ModuleType("desktop_env.evaluators")
    getters = ModuleType("desktop_env.evaluators.getters")
    metrics = ModuleType("desktop_env.evaluators.metrics")
    evaluators.getters = getters
    evaluators.metrics = metrics
    monkeypatch.setitem(sys.modules, "desktop_env", desktop_env)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators", evaluators)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.getters", getters)
    monkeypatch.setitem(sys.modules, "desktop_env.evaluators.metrics", metrics)
    monkeypatch.delenv("OSWORLD_RLVR_SNAPSHOT", raising=False)
    monkeypatch.setenv("OSWORLD_EVALUATOR_REGISTRY_ROOT", str(registry))

    task = {
        "id": "task-1",
        "evaluator": {
            "func": "check_custom_result__abc",
            "result": {"type": "custom_result__abc"},
        },
    }
    _register_rlvr_evaluators(task)

    assert getters.get_custom_result__abc("getter") == "getter"
    assert metrics.check_custom_result__abc("metric") == "metric"


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, 0.0), (0.5, 0.0), (0.999999, 0.0), (1.0, 1.0)],
)
def test_binary_reward_requires_exact_full_success(score, expected):
    task = {"metadata": {"binary_final_reward": True}}
    assert _apply_task_reward_contract(task, score) == expected


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), "1"])
def test_binary_reward_rejects_invalid_scores(score):
    task = {"metadata": {"binary_final_reward": True}}
    with pytest.raises((TypeError, ValueError)):
        _apply_task_reward_contract(task, score)


def test_unmarked_task_preserves_fractional_score():
    assert _apply_task_reward_contract({"metadata": {}}, 0.25) == 0.25
