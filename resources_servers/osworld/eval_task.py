# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-task OSWorld setup/evaluate subprocess entry (imports the pinned ``osworld`` fork).

The resources server shells out to this script (with its own venv's ``sys.executable``)
because the fork's remote-provider addressing is env-var-global
(``OSWORLD_CONTROL_SERVER_URL`` / ``OSWORLD_REMOTE_ADDR``): concurrent rollouts against
different sandboxes cannot share a process. Each invocation is one phase of one task:

- ``--phase setup``    -> ``DesktopEnv(provider_name="remote").reset(task_config)`` — the
  official reset path (setup steps with official semantics, incl. proxied/CDP handling).
- ``--phase evaluate`` -> ``_set_task_info`` + ``action_history`` + ``env.evaluate()`` — the
  official getter->metric dispatch (evaluate-at-max-steps semantics live in the caller,
  which always invokes this phase).

Addressing modes:
- ``--control-url http://<pod-ip>:5000`` (direct): host:port-expressible; the remote
  provider's ``OSWORLD_REMOTE_ADDR`` carries all ports (5000/9222/vnc/vlc) directly.
- ``--control-url http://<domain>/sandboxes/<id>/proxy/5000 --proxied``: path-based proxy;
  local forwarders (``local_forwarder.py``) give the upstream code plain host:port targets
  and inject any required route ``--headers-json``.

Contract (stdout): exactly one line ``__NEMO_GYM_OSWORLD__ {"ok": bool, "score": float|null,
"error": str|null}``.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import numbers
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


SENTINEL = "__NEMO_GYM_OSWORLD__"

VNC_PORT = 8006
VLC_PORT = 8080


def _install_python_controller_compat() -> None:
    """Add the command API expected by accepted legacy RLVR evaluators.

    The pinned OSWorld ``PythonController`` exposes specialized helpers such as
    ``execute_python_command`` and ``run_bash_script``, while accepted Astra
    evaluators call the older generic ``run_command`` API.  Both APIs ultimately
    use the guest's ``/execute`` endpoint, so install a process-local adapter
    rather than changing hundreds of frozen evaluator modules.
    """
    import requests
    from desktop_env.controllers.python import PythonController

    if callable(getattr(PythonController, "run_command", None)):
        return

    def run_command(
        self: Any,
        command: str | list[str],
        shell: bool = False,
        timeout: int = 120,
    ) -> dict[str, Any]:
        payload = {"command": command, "shell": shell}
        last_error = "unknown error"
        retry_times = max(1, int(getattr(self, "retry_times", 3)))
        retry_interval = float(getattr(self, "retry_interval", 5))
        for attempt in range(retry_times):
            try:
                response = requests.post(
                    self.http_server + "/execute",
                    json=payload,
                    timeout=timeout,
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        return result
                    last_error = f"guest returned non-object JSON: {result!r}"
                else:
                    last_error = (
                        f"guest execute returned HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except (TypeError, ValueError) as exc:
                last_error = f"invalid guest execute response: {exc}"
            if attempt + 1 < retry_times:
                time.sleep(retry_interval)
        return {
            "status": "error",
            "output": "",
            "error": last_error,
            "returncode": -1,
        }

    PythonController.run_command = run_command


def _load_evaluator_module(
    path: Path, *, package: ModuleType, role: str, task_id: str
) -> set[str]:
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    module_name = f"{package.__name__}._rlvr_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {role} evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    exported: set[str] = set()
    for name, value in vars(module).items():
        if callable(value) and getattr(value, "__module__", None) == module_name:
            setattr(package, name, value)
            exported.add(name)
    if not exported:
        raise ImportError(f"{task_id}: {path} exports no evaluator functions")
    return exported


def _read_evaluator_registry(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_MAP" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict) or not all(
            isinstance(name, str) and isinstance(module, str)
            for name, module in value.items()
        ):
            break
        return value
    raise ValueError(f"invalid evaluator registry: {path}")


def _task_evaluator_symbols(task_config: dict[str, Any]) -> dict[str, set[str]]:
    evaluator = task_config.get("evaluator")
    if not isinstance(evaluator, dict):
        return {"getters": set(), "metrics": set()}
    functions = evaluator.get("func", [])
    if not isinstance(functions, list):
        functions = [functions]
    metrics = {name for name in functions if isinstance(name, str)}
    getters: set[str] = set()
    for key in ("result", "expected"):
        values = evaluator.get(key, [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            result_type = value.get("type") if isinstance(value, dict) else None
            if isinstance(result_type, str):
                getters.add(f"get_{result_type}")
    return {"getters": getters, "metrics": metrics}


def _uses_bookmark_title_rule(task_config: dict[str, Any]) -> bool:
    evaluator = task_config.get("evaluator")
    if not isinstance(evaluator, dict):
        return False
    return "bookmark_bar_websites_with_titles" in json.dumps(
        evaluator.get("expected"),
        sort_keys=True,
    )


def _register_shared_evaluator_registry(
    task_config: dict[str, Any],
    *,
    getters: ModuleType,
    metrics: ModuleType,
    task_id: str,
) -> None:
    root_value = os.environ.get("OSWORLD_EVALUATOR_REGISTRY_ROOT", "").strip()
    if not root_value:
        return
    root = Path(root_value).resolve()
    if _uses_bookmark_title_rule(task_config):
        chrome_metric = (root / "metrics" / "chrome.py").resolve()
        if not chrome_metric.is_relative_to(root) or not chrome_metric.is_file():
            raise FileNotFoundError(f"{task_id}: patched Chrome evaluator is missing")
        _load_evaluator_module(
            chrome_metric,
            package=metrics,
            role="metrics",
            task_id=task_id,
        )

    packages = {"getters": getters, "metrics": metrics}
    for role, symbols in _task_evaluator_symbols(task_config).items():
        package = packages[role]
        unresolved = {
            symbol for symbol in symbols if not callable(getattr(package, symbol, None))
        }
        if not unresolved:
            continue
        role_root = (root / role / "verigen").resolve()
        if not role_root.is_relative_to(root):
            raise ValueError(f"{task_id}: evaluator registry escapes its root")
        routes = _read_evaluator_registry(role_root / "__init__.py")
        modules = {routes.get(symbol) for symbol in unresolved}
        if None in modules:
            missing = sorted(symbol for symbol in unresolved if symbol not in routes)
            raise ImportError(f"{task_id}: evaluator registry routes are missing: {missing}")
        for module in sorted(modules):
            if not module.replace("_", "").isalnum():
                raise ValueError(f"{task_id}: invalid evaluator module route: {module}")
            path = (role_root / f"{module}.py").resolve()
            if not path.is_relative_to(role_root) or not path.is_file():
                raise FileNotFoundError(f"{task_id}: evaluator module is missing: {path}")
            _load_evaluator_module(path, package=package, role=role, task_id=task_id)
        missing = sorted(
            symbol for symbol in symbols if not callable(getattr(package, symbol, None))
        )
        if missing:
            raise ImportError(f"{task_id}: evaluator symbols failed to load: {missing}")


def _register_rlvr_evaluators(task_config: dict[str, Any]) -> None:
    """Load this task's frozen custom getters/metrics into desktop_env."""
    root_value = os.environ.get("OSWORLD_RLVR_SNAPSHOT", "").strip()
    functions_root_value = os.environ.get(
        "OSWORLD_RLVR_FUNCTIONS_ROOT",
        str(Path(root_value) / "tmp_funcs") if root_value else "",
    ).strip()
    task_id = task_config.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("RLVR task has no non-empty id")

    from desktop_env.evaluators import getters, metrics

    packages = {"getters": getters, "metrics": metrics}
    if functions_root_value:
        functions_root = Path(functions_root_value).resolve()
        task_root = (functions_root / task_id).resolve()
        if not task_root.is_relative_to(functions_root):
            raise ValueError(f"frozen evaluator path escapes function root: {task_root}")
        if task_root.is_dir():
            loaded = 0
            for role, package in (("getters", getters), ("metrics", metrics)):
                role_root = task_root / role
                # Accepted task modules can use either ``from .helper`` or
                # ``from helper`` for task-local support files. Each evaluation
                # runs in its own subprocess, so these paths cannot leak across
                # concurrent rollouts.
                package_path = list(getattr(package, "__path__", []))
                if str(role_root) not in package_path:
                    package.__path__ = [str(role_root), *package_path]
                if str(role_root) not in sys.path:
                    sys.path.insert(0, str(role_root))
                paths = sorted(
                    path
                    for path in role_root.glob("*.py")
                    # Verigen task bundles may include executable validation
                    # scripts alongside the actual evaluator modules.  They
                    # are provenance artifacts, not DesktopEnv evaluator
                    # plugins; some intentionally export no callables.
                    if path.name != "__init__.py"
                    and not path.name.startswith("selftest_")
                )
                for path in paths:
                    _load_evaluator_module(
                        path,
                        package=package,
                        role=role,
                        task_id=task_id,
                    )
                    loaded += 1
            if loaded == 0:
                raise FileNotFoundError(
                    f"{task_id}: frozen evaluator modules not found"
                )

    # Shared registries are fallbacks for symbols not supplied by this task.
    # They also provide the patched built-in Chrome bookmark evaluator.
    _register_shared_evaluator_registry(
        task_config,
        getters=getters,
        metrics=metrics,
        task_id=task_id,
    )

    metadata = task_config.get("metadata")
    route_records = metadata.get("evaluator_routes") if isinstance(metadata, dict) else None
    if isinstance(route_records, list):
        for record in route_records:
            route_name = record.get("route") if isinstance(record, dict) else None
            if not isinstance(route_name, str) or ":" not in route_name:
                raise ValueError(f"{task_id}: invalid frozen evaluator route record")
            role, symbol = route_name.split(":", 1)
            package = packages.get(role)
            if package is None or not callable(getattr(package, symbol, None)):
                raise ImportError(
                    f"{task_id}: selected evaluator route was not loaded: {route_name}"
                )


def _emit(ok: bool, score: float | None = None, error: str | None = None) -> None:
    sys.stdout.write(f"{SENTINEL} {json.dumps({'ok': ok, 'score': score, 'error': error})}\n")
    sys.stdout.flush()


def _apply_task_reward_contract(task_config: dict[str, Any], score: Any) -> float:
    """Apply accepted RLVR tasks' exact-full-success binary reward contract."""
    metadata = task_config.get("metadata")
    binary = isinstance(metadata, dict) and metadata.get("binary_final_reward") is True
    if not binary:
        return float(score)
    if not isinstance(score, numbers.Real):
        raise TypeError("binary evaluator must return a real scalar score")
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("binary evaluator score must be finite and in [0, 1]")
    return float(value == 1.0)


def _configure_remote_addressing(
    control_url: str,
    proxied: bool,
    headers: dict[str, str],
    service_endpoints: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Point the fork's remote provider at this task's sandbox (before importing desktop_env)."""
    control_url = control_url.rstrip("/")
    if not proxied:
        # http://<host>:5000 -> host; all guest ports are directly reachable on that host.
        hostport = control_url.split("://", 1)[-1]
        host, _, port = hostport.partition(":")
        server_port = port or "5000"
        os.environ["OSWORLD_CONTROL_SERVER_URL"] = control_url
        os.environ["OSWORLD_REMOTE_ADDR"] = f"{host}:{server_port}:9222:{VNC_PORT}:{VLC_PORT}"
        return

    from resources_servers.osworld.local_forwarder import start_forwarder

    if service_endpoints:
        local_ports: dict[int, int] = {}
        for port in (5000, 9222, 8080):
            route = service_endpoints.get(str(port))
            if not isinstance(route, dict) or not route.get("endpoint"):
                raise ValueError(f"proxied service endpoint map is missing port {port}")
            _, local_ports[port] = start_forwarder(
                str(route["endpoint"]),
                dict(route.get("headers") or {}),
            )
        os.environ["OSWORLD_CONTROL_SERVER_URL"] = f"http://127.0.0.1:{local_ports[5000]}"
        os.environ["OSWORLD_REMOTE_ADDR"] = (
            f"127.0.0.1:{local_ports[5000]}:{local_ports[9222]}:{VNC_PORT}:{local_ports[8080]}"
        )
        return

    # http://<domain>/sandboxes/<id>/proxy/5000 -> proxy root, then one forwarder per port.
    if not control_url.endswith("/5000"):
        raise ValueError(f"proxied control URL must end with /5000, got: {control_url}")
    proxy_root = control_url[: -len("/5000")]
    _, port_5000 = start_forwarder(f"{proxy_root}/5000", headers)
    _, port_9222 = start_forwarder(f"{proxy_root}/9222", headers)
    # VLC's HTTP interface (:8080) is queried directly by the vlc getters
    # (vlc_playing_info -> http://<ip>:<vlc_port>/requests/status.xml) — forward it too,
    # or those evaluations dial a dead local port and crash (vlc-gate-v1: 2x evaluate_exception).
    _, port_8080 = start_forwarder(f"{proxy_root}/8080", headers)
    os.environ["OSWORLD_CONTROL_SERVER_URL"] = f"http://127.0.0.1:{port_5000}"
    os.environ["OSWORLD_REMOTE_ADDR"] = f"127.0.0.1:{port_5000}:{port_9222}:{VNC_PORT}:{port_8080}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["setup", "evaluate"])
    parser.add_argument("--config", required=True, help="Path to the task config JSON (verifier_metadata)")
    parser.add_argument("--control-url", required=True, help="Guest :5000 control base URL")
    parser.add_argument("--proxied", action="store_true", help="control-url goes through a path-based proxy")
    parser.add_argument("--headers-json", default="{}", help="Route headers required by the proxy")
    parser.add_argument("--headers-file", help="Mode-0600 JSON file containing proxy route headers")
    parser.add_argument(
        "--service-endpoints-file",
        help="Mode-0600 JSON map of guest port to endpoint and route headers",
    )
    parser.add_argument("--cache", required=True, help="Setup download cache dir")
    parser.add_argument("--action-history", default="[]", help="JSON list; evaluate phase only")
    parser.add_argument("--screen-w", type=int, default=1920)
    parser.add_argument("--screen-h", type=int, default=1080)
    parser.add_argument("--client-password", default="password")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            task_config = json.load(f)
        if args.headers_file:
            with open(args.headers_file) as f:
                headers = json.load(f) or {}
        else:
            headers = json.loads(args.headers_json) or {}
        service_endpoints = None
        if args.service_endpoints_file:
            with open(args.service_endpoints_file) as f:
                service_endpoints = json.load(f) or {}
        action_history = json.loads(args.action_history)
        if not isinstance(action_history, list):
            action_history = []
    except Exception as e:  # noqa: BLE001 - bad inputs are a clean failure, not a crash
        _emit(False, error=f"bad inputs: {e!r}")
        return 0

    try:
        _configure_remote_addressing(
            args.control_url,
            args.proxied,
            headers,
            service_endpoints,
        )
        os.makedirs(args.cache, exist_ok=True)

        from desktop_env.desktop_env import DesktopEnv

        _install_python_controller_compat()
        _register_rlvr_evaluators(task_config)
        env = DesktopEnv(
            provider_name="remote",
            action_space="pyautogui",
            cache_dir=args.cache,
            screen_size=(args.screen_w, args.screen_h),
            client_password=args.client_password,
            require_a11y_tree=False,
            require_terminal=False,
        )

        if args.phase == "setup":
            env.reset(task_config=task_config)
            _emit(True)
        else:
            env._set_task_info(task_config)
            # _set_task_info points env.cache_dir at <base>/<task_id>, but the SetupController
            # (which executes evaluator postconfig steps and writes their `stdout:` captures)
            # still holds the base dir from __init__ — reset() is what normally syncs them, and
            # the evaluate phase never calls reset(). Without this, every cache_file getter
            # whose file comes from a postconfig capture raises FileNotFoundError
            # (= the thunderbird "evaluate_exception" infra crashes).
            env.setup_controller.reset_cache_dir(env.cache_dir)
            env.action_history = action_history
            # The guest control API occasionally answers a transient 500 mid-evaluation
            # (observed as "Internal Server Error for url: .../platform"); one such blip
            # otherwise scores a solved task 0 with `evaluate_exception`. Retry briefly.
            for attempt in range(3):
                try:
                    score = env.evaluate()
                    break
                except Exception as exc:  # noqa: BLE001 - inspect and re-raise if not transient
                    if "Internal Server Error" not in str(exc) or attempt == 2:
                        raise
                    sys.stderr.write(f"evaluate transient 5xx (attempt {attempt + 1}/3): {exc}\n")
                    time.sleep(10)
            _emit(True, score=_apply_task_reward_contract(task_config, score))
    except Exception as exc:  # noqa: BLE001 - report, don't crash: the caller decides the consequence
        sys.stderr.write(f"eval_task ({args.phase}) failed:\n" + traceback.format_exc())
        # Keep the machine-readable result diagnostic even when the traceback's
        # final exception line contains a very large evaluator-state repr and is
        # therefore truncated by the parent process's stderr-tail logging.
        details = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            details.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        detail = " <- ".join(details)
        _emit(
            False,
            score=0.0 if args.phase == "evaluate" else None,
            error=f"{args.phase}_exception: {detail[:1200]}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
