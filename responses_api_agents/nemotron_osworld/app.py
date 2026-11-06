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
"""Nemotron-Omni OSWorld agent — the validated Omni-Nano-v3 harness over the OSWorld
resources server.

``run()`` wires the three servers together: ``/seed_session`` (allocate + official task
setup) -> self ``/v1/responses`` (the agent loop) -> ``/verify`` (official evaluator +
release), threading the session cookie through every call and forwarding the faithful
OSWorld ``action_history`` to the verifier.

``responses()`` is the host-side observe -> act loop, driven by the REFERENCE agent —
``mm_agents.nvidia.nemotron_agent.NemotronAgent`` imported from the pinned ``osworld``
fork (prompt template, message/history serialization, free-text ``## Action/## Code``
parsing, 0-1 fractional coordinate projection, 5-retry parse recovery, and the internal
force-``FAIL`` at the step cap are all vendor-validated code). Its synchronous model seam
is replaced with a thread-safe bridge back to Gym's shared aiohttp ``ServerClient``.
Per step it fetches ``/screenshot``, calls ``agent.predict`` in a worker thread, and
executes the returned pyautogui snippets via ``/execute`` with the exact
``PythonController`` wrapping.
"""

import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import ClientTimeout
from fastapi import Request, Response
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status


logger = logging.getLogger("nemo_gym.osworld.nemotron_agent")

# Mirrors desktop_env.controllers.python.PythonController's pkgs_prefix + command shape:
# actions run as `python -c "<prefix + action>"` with shell=False in the guest.
_PYAUTOGUI_PKGS_PREFIX = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"


class NemotronOSWorldAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # The model id sent in the chat-completions payload (the model server's served name).
    model_name: str = "vllm_local"

    # Reference-run knobs (Omni-Nano-v3 47-48% runs; see the fork's reference launch scripts).
    max_steps: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 4096
    max_image_history_length: int = 3
    # Keep the number of samples per GRPO group independent from live VM pressure.
    # Deployments can raise this to match the available OpenSandbox VM pool.
    max_parallel_rollouts: int = 4
    thinking: bool = True
    coordinate_type: str = "relative"
    client_password: str = "password"
    screen_width: int = 1920
    screen_height: int = 1080
    # The reference runner's env.step pause (sleep_after_execution); WAIT sleeps the same.
    sleep_after_execution_s: float = 5.0
    llm_timeout_s: float = Field(default=1200.0, gt=0)
    rollout_timeout_s: float = Field(default=3600.0, gt=0)
    bound_model_calls_by_rollout_deadline: bool = False
    action_timeout_s: float = Field(default=60.0, gt=0)
    model_retries: int = Field(default=1, ge=1)
    model_retry_backoff_s: float = Field(default=2.0, ge=0)
    prompt_format_variant: Optional[str] = None
    prompt_mode: str = "hf_tools"
    execution_feedback: bool = False
    image_block_k: int = Field(default=0, ge=0)
    image_boundary_s: int = Field(default=0, ge=0)
    canonical_harness_root: str = ""
    canonical_agent_sha256: str = ""
    canonical_feedback_sha256: str = ""
    # Debug aid: when set, save each step's observed screenshot + the parsed actions under
    # <dir>/<task_id8>/ so failing trajectories can be replayed visually. OFF for real gates.
    debug_trajectory_dir: str = ""


class NemotronOSWorldRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NemotronOSWorldVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class NemotronOSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _extract_instruction(response_input) -> str:
    """Pull the task instruction text out of the incoming Responses-API input."""
    if isinstance(response_input, str):
        return response_input
    for message in response_input:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if role != "user":
            continue
        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                if text:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
    return ""


def _message_to_text(message: Any) -> str:
    """Flatten the reference agent's returned message (or error string) into rollout text."""
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        content = message.get("content") or ""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return str(content)
    return str(message)


def _cookie_values(cookies: Any) -> Dict[str, str]:
    return {
        str(key): str(getattr(value, "value", value))
        for key, value in (cookies.items() if cookies is not None else ())
    }


class NemotronOSWorldAgent(SimpleResponsesAPIAgent):
    config: NemotronOSWorldAgentConfig
    # Faithful OSWorld action history per rollout session; run() forwards it to /verify
    # (the evaluator inspects the LAST entry for the infeasible/FAIL contract). Keyed by
    # the session id as seen INSIDE the cookie-threaded requests, so run() retrieves it
    # through the cookie-scoped /action_history endpoint rather than its own session.
    session_id_to_action_history: Dict[str, List[str]] = Field(default_factory=dict)
    _rollout_semaphore: Optional[asyncio.Semaphore] = PrivateAttr(default=None)
    _canonical_agent_module: Any = PrivateAttr(default=None)
    _canonical_feedback_module: Any = PrivateAttr(default=None)

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/action_history")(self.action_history)
        return app

    async def action_history(self, request: Request) -> Dict[str, List[str]]:
        session_id = request.session[SESSION_ID_KEY]
        return {"action_history": self.session_id_to_action_history.pop(session_id, [])}

    @staticmethod
    def _load_pinned_module(path: Path, module_name: str, expected_sha256: str) -> Any:
        source = path.read_bytes()
        actual_sha256 = hashlib.sha256(source).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Canonical OSWorld source digest mismatch for {path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load canonical OSWorld source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        logger.info("Loaded canonical OSWorld source %s (sha256=%s)", path, actual_sha256)
        return module

    def _reference_agent_module(self) -> Any:
        if not self.config.canonical_harness_root:
            from mm_agents.nvidia import nemotron_agent  # noqa: PLC0415

            return nemotron_agent
        if self._canonical_agent_module is None:
            path = (
                Path(self.config.canonical_harness_root)
                / "mm_agents"
                / "nvidia"
                / "nemotron_agent.py"
            )
            self._canonical_agent_module = self._load_pinned_module(
                path,
                "_nemo_gym_canonical_nemotron_agent",
                self.config.canonical_agent_sha256,
            )
        return self._canonical_agent_module

    def _feedback_module(self) -> Any:
        if self._canonical_feedback_module is None:
            if not self.config.canonical_harness_root:
                raise RuntimeError("execution_feedback requires canonical_harness_root")
            path = Path(self.config.canonical_harness_root) / "desktop_env" / "execution_feedback.py"
            self._canonical_feedback_module = self._load_pinned_module(
                path,
                "_nemo_gym_canonical_execution_feedback",
                self.config.canonical_feedback_sha256,
            )
        return self._canonical_feedback_module

    def _make_reference_agent(self) -> Any:
        """Build the reference NemotronAgent from the pinned fork (lazy import: the fork's
        dependency set is heavy and Linux-only, and unit tests stub this seam)."""
        if self.config.prompt_format_variant:
            os.environ["NEMOTRON_OSWORLD_PROMPT_MODE"] = self.config.prompt_mode
        NemotronAgent = self._reference_agent_module().NemotronAgent

        return NemotronAgent(
            model=self.config.model_name,
            max_steps=self.config.max_steps,
            max_image_history_length=self.config.max_image_history_length,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            screen_size=(self.config.screen_width, self.config.screen_height),
            coordinate_type=self.config.coordinate_type,
            password=self.config.client_password,
            thinking=self.config.thinking,
            prompt_format_variant=self.config.prompt_format_variant,
        )

    def _bind_reference_model_transport(
        self,
        agent: Any,
        request: Request,
        model_cookies: Dict[str, str],
    ) -> None:
        """Replace the fork's sync network call with Gym's shared aiohttp transport."""

        if not hasattr(agent, "call_llm"):
            return
        event_loop = asyncio.get_running_loop()

        async def call(payload: Dict[str, Any]) -> Dict[str, Any]:
            last_error: Optional[BaseException] = None
            for attempt in range(self.config.model_retries):
                # The rollout deadline is otherwise only checked between steps, so
                # eight retried model calls could outlive it by up to two hours.
                deadline = getattr(agent, "_nemo_gym_rollout_deadline", None)
                request_timeout_s = self.config.llm_timeout_s
                if deadline is not None:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        break
                    request_timeout_s = min(request_timeout_s, remaining_s)
                try:
                    model_response = await self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path=self.url_path_for_request("/v1/chat/completions", request),
                        json=payload,
                        cookies=model_cookies,
                        timeout=ClientTimeout(total=request_timeout_s),
                        _retry=False,
                    )
                    await raise_for_status(model_response)
                    model_cookies.update(
                        {
                            str(key): str(getattr(value, "value", value))
                            for key, value in model_response.cookies.items()
                        }
                    )
                    response_payload = await get_response_json(model_response)
                    choices = response_payload.get("choices") or []
                    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                        raise RuntimeError("Nemotron model response did not finish cleanly")
                    message = choices[0].get("message")
                    if not isinstance(message, dict):
                        raise RuntimeError("Nemotron model response has no chat message")
                    # The NeMo RL endpoint can attach multi-megabyte token/routed-expert
                    # metadata to the OpenAI message.  The canonical OSWorld agent only
                    # consumes these three text fields; returning the transport metadata
                    # would also make its per-step response logger emit gigabytes.
                    return {
                        key: message[key]
                        for key in ("content", "reasoning_content", "reasoning")
                        if key in message
                    }
                except Exception as error:  # noqa: BLE001 - canonical runner retries transport failures
                    last_error = error
                    if attempt + 1 < self.config.model_retries:
                        await asyncio.sleep(
                            min(30.0, self.config.model_retry_backoff_s * (attempt + 1))
                        )
            raise RuntimeError(
                f"Nemotron model request failed after {self.config.model_retries} attempts"
            ) from last_error

        def call_from_worker(payload: Dict[str, Any], _model: str) -> Dict[str, Any]:
            future = asyncio.run_coroutine_threadsafe(call(payload), event_loop)
            try:
                return future.result(timeout=self.config.rollout_timeout_s + 5.0)
            except TimeoutError:
                future.cancel()
                raise

        agent.call_llm = call_from_worker

    def _set_image_window(self, agent: Any) -> None:
        """Apply RFC0040's exact K/S image rule before the next canonical-agent call."""
        if self.config.image_block_k <= 0 or self.config.image_boundary_s <= 0:
            return
        observations = getattr(agent, "observations", ())
        step_idx = len(observations)
        if step_idx < self.config.image_block_k:
            image_window_start = 0
        else:
            boundary = (step_idx // self.config.image_block_k) * self.config.image_block_k
            image_window_start = max(1, boundary - (self.config.image_boundary_s - 1))
        # The RFC0015 agent pins observation zero independently and retains this
        # many observations in the post-initial contiguous image window.
        agent.max_image_history_length = step_idx + 1 - image_window_start

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        instruction = _extract_instruction(body.input)
        session_id = request.session[SESSION_ID_KEY]
        resources_name = self.config.resources_server.name
        resources_cookies = _cookie_values(request.cookies)

        agent = self._make_reference_agent()
        agent.reset(None)
        model_cookies: Dict[str, str] = {}
        self._bind_reference_model_transport(agent, request, model_cookies)

        action_history: List[str] = []
        output_items: List[dict] = []
        terminal = False
        pending_feedback_text = ""
        rollout_deadline = time.monotonic() + self.config.rollout_timeout_s
        if self.config.bound_model_calls_by_rollout_deadline:
            agent._nemo_gym_rollout_deadline = rollout_deadline
        feedback_module = self._feedback_module() if self.config.execution_feedback else None

        debug_dir = None
        if self.config.debug_trajectory_dir:
            debug_dir = os.path.join(self.config.debug_trajectory_dir, session_id[:8])
            os.makedirs(debug_dir, exist_ok=True)

        for step_idx in range(self.config.max_steps):
            remaining_s = rollout_deadline - time.monotonic()
            if remaining_s <= 0:
                logger.warning("OSWorld rollout timed out before step %d", step_idx + 1)
                break
            # OBSERVE — fetch the current screenshot from the resources server.
            shot_resp = await self.server_client.post(
                server_name=resources_name,
                url_path="/screenshot",
                cookies=resources_cookies,
                timeout=ClientTimeout(total=remaining_s),
                _retry=False,
            )
            await raise_for_status(shot_resp)
            resources_cookies.update(_cookie_values(shot_resp.cookies))
            shot_json = await get_response_json(shot_resp)
            obs = {"screenshot": base64.b64decode(shot_json["image_base64"])}
            if pending_feedback_text:
                obs["execution_feedback_batch_text"] = pending_feedback_text

            # THINK — the reference agent builds messages, calls the model server's
            # chat-completions endpoint through the ServerClient bridge, and parses
            # the free-text `## Action/## Code` contract (including parse recovery
            # and the internal force-FAIL at the step cap).
            self._set_image_window(agent)
            try:
                message, actions, _cot = await asyncio.to_thread(
                    agent.predict, instruction, obs, step_idx=step_idx
                )
            except RuntimeError as error:
                if not str(error).startswith("Nemotron model request failed after"):
                    raise
                # Exhausting all model attempts (most commonly because the
                # trajectory filled max_model_len) is a terminal rollout
                # failure, not an infrastructure row to mask. Preserve a
                # concrete assistant output and FAIL action so the official
                # evaluator scores the current desktop state as the reference
                # runner would, while the rest of the shard continues.
                logger.warning(
                    "OSWorld model retries exhausted at step %d; terminating rollout with FAIL",
                    step_idx + 1,
                )
                action_history.append("FAIL")
                output_items.append(
                    {
                        "id": f"msg_{step_idx + 1}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "FAIL",
                                "annotations": [],
                            }
                        ],
                    }
                )
                break
            if debug_dir is not None:
                with open(os.path.join(debug_dir, f"step_{step_idx:03d}.png"), "wb") as f:
                    f.write(obs["screenshot"])
                with open(os.path.join(debug_dir, "trace.jsonl"), "a") as f:
                    f.write(
                        json.dumps({"step": step_idx, "actions": actions, "message": _message_to_text(message)[:4000]})
                        + "\n"
                    )
            output_item = {
                "id": f"msg_{step_idx + 1}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": _message_to_text(message), "annotations": []}],
            }
            output_items.append(output_item)

            # ACT — execute each returned action with the reference runner's semantics.
            execution_feedbacks = []
            for action in actions:
                action_history.append(action)
                if action == "WAIT":
                    await asyncio.sleep(self.config.sleep_after_execution_s)
                    if self.config.execution_feedback:
                        # DesktopEnv.step sleeps once for WAIT and once after every
                        # action; preserve that historical feedback_v2 behavior.
                        await asyncio.sleep(self.config.sleep_after_execution_s)
                    continue
                if action in ("FAIL", "DONE"):
                    terminal = True
                    break
                command = ["python", "-c", _PYAUTOGUI_PKGS_PREFIX.format(command=action)]
                exec_resp = await self.server_client.post(
                    server_name=resources_name,
                    url_path="/execute",
                    json={"command": command, "shell": False},
                    cookies=resources_cookies,
                    timeout=ClientTimeout(
                        total=min(
                            self.config.action_timeout_s,
                            max(rollout_deadline - time.monotonic(), 0.001),
                        )
                    ),
                    _retry=False,
                )
                await raise_for_status(exec_resp)
                resources_cookies.update(_cookie_values(exec_resp.cookies))
                if feedback_module is not None:
                    execution_feedbacks.append(
                        feedback_module.normalize_execution_result(
                            await get_response_json(exec_resp),
                            model_command=action,
                        )
                    )
                await asyncio.sleep(self.config.sleep_after_execution_s)
            pending_feedback_text = (
                feedback_module.render_feedback_batch(execution_feedbacks)
                if feedback_module is not None and execution_feedbacks
                else ""
            )
            if terminal:
                break

        self.session_id_to_action_history[session_id] = action_history

        for key, value in (*resources_cookies.items(), *model_cookies.items()):
            response.set_cookie(key, value)

        return NeMoGymResponse.model_validate(
            {
                "id": f"resp_osworld_{session_id[:16]}",
                "created_at": time.time(),
                "model": self.config.model_name,
                "object": "response",
                "output": output_items,
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            }
        )

    async def run(self, request: Request, body: NemotronOSWorldRunRequest) -> NemotronOSWorldVerifyResponse:
        """One rollout. NEVER raises: a raised exception propagates through gym's rollout
        TaskGroup and kills the ENTIRE run (observed live: one seed 500 after sandbox-pool
        starvation crashed a 24-task gate). Infrastructure failures instead return a
        zero-reward row marked with ``verify_error`` so a post-run pass can strip and
        re-run exactly those tasks."""
        if self._rollout_semaphore is None:
            self._rollout_semaphore = asyncio.Semaphore(max(1, self.config.max_parallel_rollouts))

        async with self._rollout_semaphore:
            resources_cookie_holder: Dict[str, str] = {}
            try:
                return await self._run_rollout(request, body, resources_cookie_holder)
            except Exception as e:  # noqa: BLE001 - one bad rollout must not kill the run
                logger.warning("OSWorld rollout failed (%r); emitting marked zero-reward row", e)
                body_dict = body.model_dump()
                instance_config = dict(body_dict.get("instance_config") or {})
                instance_config["mask_sample"] = True
                # BaseVerifyRequest requires a `response`; the rollout died before producing one.
                placeholder_response = {
                    "id": "resp_rollout_infra_failure",
                    "created_at": time.time(),
                    "model": self.config.model_name,
                    "object": "response",
                    # NeMo RL still requires at least one assistant generation when
                    # allow_token_free_results is enabled.  Keep this row explicitly
                    # masked, but provide a minimal token-free message so one transient
                    # sandbox/agent 500 cannot abort the entire evaluation shard.
                    "output": [
                        {
                            "id": "msg_rollout_infra_failure",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "rollout_infra_failure",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                }
                if resources_cookie_holder:
                    # Best-effort cleanup without replaying the official evaluator.
                    try:
                        await self.server_client.post(
                            server_name=self.config.resources_server.name,
                            url_path="/release",
                            cookies=dict(resources_cookie_holder),
                            _retry=False,
                        )
                    except Exception as release_error:  # noqa: BLE001 - release is best-effort
                        logger.warning("post-failure sandbox release also failed: %r", release_error)
                return NemotronOSWorldVerifyResponse.model_validate(
                    body_dict
                    | {
                        "response": placeholder_response,
                        "reward": 0.0,
                        "verify_error": f"rollout_infra_failure: {e!r}"[:400],
                        "instance_config": instance_config,
                    }
                )

    async def _run_rollout(
        self, request: Request, body: NemotronOSWorldRunRequest, resources_cookie_holder: Dict[str, str]
    ) -> NemotronOSWorldVerifyResponse:
        cookies = request.cookies

        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
            _retry=False,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies
        resources_cookie_holder.update(cookies)

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
            _retry=False,
        )
        await raise_for_status(response)
        cookies = response.cookies
        resources_cookie_holder.update(cookies)

        history_response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/action_history",
            cookies=cookies,
            _retry=False,
        )
        await raise_for_status(history_response)
        cookies = history_response.cookies
        action_history: Optional[List[str]] = (await get_response_json(history_response)).get("action_history")

        verify_request = NemotronOSWorldVerifyRequest.model_validate(
            body.model_dump()
            | {
                "response": await get_response_json(response),
                "action_history": action_history or [],
            }
        )

        # Merge every cookie seen this rollout (seed's resources session cookie included):
        # if an intermediate hop's response dropped the resources session cookie, verify
        # would otherwise land on a fresh session and score 0 with `no_seeded_sandbox`.
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies={**resources_cookie_holder, **dict(cookies)},
            _retry=False,
        )
        await raise_for_status(verify_response)
        return NemotronOSWorldVerifyResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    NemotronOSWorldAgent.run_webserver()
