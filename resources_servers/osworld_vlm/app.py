from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


def _auto_detect_osworld_repo_root() -> str:
    env_value = os.environ.get("OSWORLD_REPO_ROOT", "").strip()
    if env_value:
        env_root = Path(env_value).expanduser()
        if env_root.exists():
            return str(env_root.resolve())

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidate = parent / "osworld"
        if (candidate / "third_party" / "OSWorld" / "desktop_env").exists():
            return str(candidate.resolve())
    return ""


@dataclass
class SessionState:
    env: Any
    task_config: Dict[str, Any]
    latest_observation: Dict[str, Any]
    max_steps: int
    step_count: int = 0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    screenshot_dir: Optional[Path] = None
    preserve_screenshots: bool = False


class OSWorldResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "osworld_vlm"

    # Repo root that contains third_party/OSWorld.
    osworld_repo_root: str = Field(default_factory=_auto_detect_osworld_repo_root)
    test_config_base_dir: str = "evaluation_examples"

    provider_name: str = "apptainer"
    region: Optional[str] = None
    path_to_vm: Optional[str] = None
    snapshot_name: str = "init_state"
    action_space: str = "pyautogui"
    cache_dir: str = "cache"
    headless: bool = True
    require_a11y_tree: bool = True
    require_terminal: bool = False
    os_type: str = "Ubuntu"
    enable_proxy: bool = False
    client_password: str = ""
    screen_width: int = 1920
    screen_height: int = 1080

    default_max_steps: int = 15
    step_pause_seconds: float = 1.5
    max_a11y_chars: int = 12_000
    max_terminal_chars: int = 8_000
    trace_dir: Optional[str] = None
    initial_observation_timeout_seconds: float = 60.0
    initial_observation_poll_seconds: float = 2.0
    initial_observation_min_nonblack_ratio: float = 0.001


class OSWorldSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    domain: Optional[str] = None
    example_id: Optional[str] = None
    task_config_path: Optional[str] = None
    task_config: Optional[Dict[str, Any]] = None
    max_steps: Optional[int] = None


class OSWorldToolRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class OSWorldActionRequest(OSWorldToolRequest):
    action: str
    pause_seconds: Optional[float] = None


class OSWorldVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class OSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    metadata: Dict[str, Any] = Field(default_factory=dict)


class OSWorldResourcesServer(SimpleResourcesServer):
    config: OSWorldResourcesServerConfig
    session_id_to_state: Dict[str, SessionState] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._osworld_repo_root = Path(self.config.osworld_repo_root).expanduser()
        if not self._osworld_repo_root.exists():
            raise ValueError(
                f"OSWorld repo root does not exist: {self._osworld_repo_root}. "
                "Set OSWORLD_REPO_ROOT to the directory containing third_party/OSWorld."
            )

        self._osworld_project_root = self._osworld_repo_root / "third_party" / "OSWorld"
        if not self._osworld_project_root.exists():
            raise ValueError(
                f"OSWorld project root not found under {self._osworld_repo_root}. "
                "Expected third_party/OSWorld."
            )

        if str(self._osworld_project_root) not in sys.path:
            sys.path.insert(0, str(self._osworld_project_root))

        self._task_base_dir = self._resolve_task_base_dir(self.config.test_config_base_dir)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_tool)
        return app

    def _screenshot_nonblack_ratio(self, screenshot: Optional[bytes]) -> float:
        if not screenshot:
            return 0.0
        try:
            image = Image.open(BytesIO(screenshot)).convert("L")
            image.thumbnail((160, 90))
            histogram = image.histogram()
            return sum(histogram[11:]) / max(1, image.width * image.height)
        except Exception:
            return 0.0

    def _task_requires_chromium(self, task_config: Dict[str, Any]) -> bool:
        if str(task_config.get("snapshot", "")).lower() == "chrome":
            return True
        return any(
            "chrome" in str(app).lower()
            for app in task_config.get("related_apps", [])
        )

    def _chromium_ready(self, env: Any) -> bool:
        try:
            response = requests.get(
                f"http://{env.vm_ip}:{env.chromium_port}/json/version",
                timeout=3,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _wait_for_initial_observation(
        self,
        env: Any,
        task_config: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        timeout = max(1.0, self.config.initial_observation_timeout_seconds)
        poll_seconds = max(0.1, self.config.initial_observation_poll_seconds)
        min_nonblack_ratio = max(
            0.0, self.config.initial_observation_min_nonblack_ratio
        )
        requires_chromium = self._task_requires_chromium(task_config)
        deadline = time.monotonic() + timeout
        latest_observation = observation
        nonblack_ratio = self._screenshot_nonblack_ratio(
            latest_observation.get("screenshot")
        )
        chromium_ready = not requires_chromium or self._chromium_ready(env)

        while (
            nonblack_ratio < min_nonblack_ratio or not chromium_ready
        ) and time.monotonic() < deadline:
            time.sleep(poll_seconds)
            screenshot = env.controller.get_screenshot()
            if screenshot:
                latest_observation = {
                    **latest_observation,
                    "screenshot": screenshot,
                }
            nonblack_ratio = self._screenshot_nonblack_ratio(
                latest_observation.get("screenshot")
            )
            chromium_ready = not requires_chromium or self._chromium_ready(env)

        if nonblack_ratio < min_nonblack_ratio or not chromium_ready:
            raise TimeoutError(
                "OSWorld task setup did not produce a usable initial observation "
                f"within {timeout:.1f}s "
                f"(nonblack_ratio={nonblack_ratio:.6f}, "
                f"required={min_nonblack_ratio:.6f}, "
                f"chromium_required={requires_chromium}, "
                f"chromium_ready={chromium_ready})."
            )

        # Refresh all modalities once after the task-specific app is ready.
        return env._get_obs()

    async def seed_session(
        self, request: Request, body: OSWorldSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self._cleanup_session(session_id)

        task_config = self._resolve_task_config(body)
        env = self._create_env()
        try:
            observation = env.reset(task_config=task_config)
            observation = self._wait_for_initial_observation(
                env,
                task_config,
                observation,
            )
        except Exception:
            env.close()
            raise
        max_steps = body.max_steps or self.config.default_max_steps

        preserve_screenshots = bool(self.config.trace_dir)
        if preserve_screenshots:
            safe_session_id = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in session_id
            )
            screenshot_dir = (
                Path(self.config.trace_dir).expanduser()
                / f"{safe_session_id}-{time.time_ns()}"
            ).resolve()
            screenshot_dir.mkdir(parents=True, exist_ok=False)
            (screenshot_dir / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "domain": body.domain,
                        "example_id": body.example_id,
                        "task_config_path": body.task_config_path,
                        "instruction": task_config.get("instruction"),
                        "max_steps": max_steps,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            screenshot_dir = Path(
                tempfile.mkdtemp(prefix=f"osworld_gym_{session_id}_")
            ).resolve()
        state = SessionState(
            env=env,
            task_config=task_config,
            latest_observation={},
            max_steps=max_steps,
            screenshot_dir=screenshot_dir,
            preserve_screenshots=preserve_screenshots,
        )
        state.latest_observation = self._snapshot_observation(state, observation)
        self.session_id_to_state[session_id] = state
        return BaseSeedSessionResponse()

    async def route_tool(
        self, path: str, body: OSWorldToolRequest, request: Request
    ) -> Dict[str, Any]:
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_id_to_state.get(session_id)
        if state is None:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Call seed_session first.",
            )

        try:
            if path == "osworld_get_observation":
                return {
                    "ok": True,
                    "observation": state.latest_observation,
                    "done": state.done,
                }

            if path == "osworld_finish":
                state.done = True
                return {
                    "ok": True,
                    "observation": state.latest_observation,
                    "done": True,
                }

            if path == "osworld_execute_action":
                action_body = OSWorldActionRequest.model_validate(
                    body.model_dump(exclude_unset=True)
                )
                if state.done:
                    return {
                        "ok": False,
                        "error": "Episode already marked done.",
                        "observation": state.latest_observation,
                        "done": True,
                    }

                pause = (
                    action_body.pause_seconds
                    if action_body.pause_seconds is not None
                    else self.config.step_pause_seconds
                )
                observation, reward, done, info = state.env.step(
                    action_body.action, pause=float(pause)
                )
                state.step_count += 1
                state.info = info or {}
                if state.step_count >= state.max_steps and not done:
                    done = True
                    state.info = {
                        **state.info,
                        "max_steps_reached": True,
                    }
                state.done = bool(done)
                state.latest_observation = self._snapshot_observation(
                    state, observation, done=state.done, info=state.info
                )
                return {
                    "ok": True,
                    "reward": float(reward),
                    "observation": state.latest_observation,
                    "done": state.done,
                }

            return {"ok": False, "error": f"Unknown tool: {path}"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def verify(
        self, request: Request, body: OSWorldVerifyRequest
    ) -> OSWorldVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_id_to_state.get(session_id)

        reward = 0.0
        metadata: Dict[str, Any] = {}
        if state is None:
            metadata["error"] = "No active session found during verify."
        else:
            try:
                reward = float(state.env.evaluate())
            except Exception as exc:
                metadata["error"] = f"evaluate failed: {type(exc).__name__}: {exc}"
            metadata["step_count"] = state.step_count
            metadata["done"] = state.done
            metadata["info"] = state.info
            if state.preserve_screenshots and state.screenshot_dir:
                (state.screenshot_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "reward": reward,
                            "step_count": state.step_count,
                            "done": state.done,
                            "info": state.info,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )

        self._cleanup_session(session_id)
        return OSWorldVerifyResponse(
            **body.model_dump(),
            reward=reward,
            metadata=metadata,
        )

    def _resolve_task_base_dir(self, configured_dir: str) -> Path:
        candidate = Path(configured_dir)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self._osworld_project_root / candidate).resolve()

    def _resolve_task_config(self, body: OSWorldSeedSessionRequest) -> Dict[str, Any]:
        if body.task_config is not None:
            return body.task_config

        if body.task_config_path:
            task_path = Path(body.task_config_path).expanduser()
            if not task_path.is_absolute():
                task_path = (self._osworld_project_root / task_path).resolve()
            return json.loads(task_path.read_text(encoding="utf-8"))

        if body.domain and body.example_id:
            task_path = (
                self._task_base_dir
                / "examples"
                / body.domain
                / f"{body.example_id}.json"
            )
            return json.loads(task_path.read_text(encoding="utf-8"))

        raise ValueError(
            "Missing task configuration. Provide task_config, task_config_path, "
            "or both domain and example_id."
        )

    def _create_env(self):
        from desktop_env.desktop_env import DesktopEnv

        env_kwargs: Dict[str, Any] = {
            "provider_name": self.config.provider_name,
            "region": self.config.region,
            "path_to_vm": self.config.path_to_vm,
            "snapshot_name": self.config.snapshot_name,
            "action_space": self.config.action_space,
            "cache_dir": self.config.cache_dir,
            "screen_size": (self.config.screen_width, self.config.screen_height),
            "headless": self.config.headless,
            "require_a11y_tree": self.config.require_a11y_tree,
            "require_terminal": self.config.require_terminal,
            "os_type": self.config.os_type,
            "enable_proxy": self.config.enable_proxy,
            "client_password": self.config.client_password,
        }
        return DesktopEnv(**env_kwargs)

    def _truncate(self, value: Optional[str], max_chars: int) -> Optional[str]:
        if value is None:
            return None
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "\n...[truncated]..."

    def _write_screenshot(
        self, state: SessionState, screenshot_bytes: Optional[bytes]
    ) -> Optional[str]:
        if not screenshot_bytes or state.screenshot_dir is None:
            return None
        filename = f"step_{state.step_count:04d}.png"
        path = state.screenshot_dir / filename
        path.write_bytes(screenshot_bytes)
        return str(path)

    def _snapshot_observation(
        self,
        state: SessionState,
        observation: Dict[str, Any],
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        screenshot_path = self._write_screenshot(state, observation.get("screenshot"))
        return {
            "instruction": observation.get("instruction"),
            "accessibility_tree": self._truncate(
                observation.get("accessibility_tree"),
                self.config.max_a11y_chars,
            ),
            "terminal": self._truncate(
                observation.get("terminal"),
                self.config.max_terminal_chars,
            ),
            "screenshot_path": screenshot_path,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
            "done": done,
            "info": info or {},
        }

    def _cleanup_session(self, session_id: str) -> None:
        state = self.session_id_to_state.pop(session_id, None)
        if state is None:
            return
        try:
            state.env.close()
        except Exception:
            pass
        if (
            not state.preserve_screenshots
            and state.screenshot_dir
            and state.screenshot_dir.exists()
        ):
            shutil.rmtree(state.screenshot_dir, ignore_errors=True)


if __name__ == "__main__":
    OSWorldResourcesServer.run_webserver()
