from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from PIL import Image
from pydantic import Field, model_validator

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyResponse,
)


class OSWorldVLMAgentConfig(SimpleAgentConfig):
    max_image_side: int = 1024
    image_detail: str = "high"
    strip_images_from_output: bool = True

    @model_validator(mode="after")
    def _validate_image_side(self) -> "OSWorldVLMAgentConfig":
        if self.max_image_side < 0:
            raise ValueError("max_image_side must be >= 0")
        return self


def _extract_observation_from_tool_output(raw_output: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        if isinstance(payload.get("observation"), dict):
            return payload["observation"]
        if isinstance(payload.get("output"), dict) and isinstance(
            payload["output"].get("observation"), dict
        ):
            return payload["output"]["observation"]
    return None


def _compact_tool_output_with_observation(
    raw_output: str, observation: Optional[Dict[str, Any]]
) -> str:
    """Avoid sending the same large observation both as tool output and a VLM input."""
    if observation is None:
        return raw_output

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        payload = {}

    compact: Dict[str, Any] = {}
    if isinstance(payload, dict):
        compact.update(
            {
                key: value
                for key, value in payload.items()
                if key not in {"observation", "output"}
            }
        )
        nested_output = payload.get("output")
        if isinstance(nested_output, dict):
            compact_nested = {
                key: value
                for key, value in nested_output.items()
                if key != "observation"
            }
            if compact_nested:
                compact["output"] = compact_nested

    compact.update(
        {
            "observation_attached": True,
            "step_count": observation.get("step_count"),
            "done": observation.get("done"),
        }
    )
    return json.dumps(compact, ensure_ascii=False)


def _truncate_text(text: Optional[str], max_chars: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _to_data_url(image_path: Path, max_side: int) -> str:
    image_bytes = image_path.read_bytes()
    if max_side > 0:
        with Image.open(BytesIO(image_bytes)) as img:
            w, h = img.size
            largest_side = max(w, h)
            if largest_side > max_side:
                scale = max_side / float(largest_side)
                resized = img.resize((int(w * scale), int(h * scale)))
                buf = BytesIO()
                resized.save(buf, format="PNG")
                image_bytes = buf.getvalue()

    encoded = base64.standard_b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _observation_to_message(
    observation: Dict[str, Any], cfg: OSWorldVLMAgentConfig
) -> NeMoGymEasyInputMessage:
    content: List[Dict[str, str]] = []

    screenshot_path = observation.get("screenshot_path")
    if screenshot_path:
        image_path = Path(screenshot_path)
        if image_path.exists():
            content.append(
                {
                    "type": "input_image",
                    "image_url": _to_data_url(image_path, cfg.max_image_side),
                    "detail": cfg.image_detail,
                }
            )

    observation_lines = [
        "Updated desktop observation:",
        f"- step_count: {observation.get('step_count')}",
        f"- max_steps: {observation.get('max_steps')}",
        f"- done: {observation.get('done')}",
        f"- info: {json.dumps(observation.get('info', {}), ensure_ascii=False)}",
    ]

    instruction = observation.get("instruction")
    if instruction:
        observation_lines.append(f"\nInstruction:\n{instruction}")
    a11y_tree = _truncate_text(observation.get("accessibility_tree"), max_chars=5000)
    if a11y_tree:
        observation_lines.append(f"\nAccessibility tree:\n{a11y_tree}")
    terminal = _truncate_text(observation.get("terminal"), max_chars=3000)
    if terminal:
        observation_lines.append(f"\nTerminal output:\n{terminal}")

    observation_lines.append(
        "\nReturn the next action by calling one tool: "
        "`osworld_execute_action`, `osworld_get_observation`, or `osworld_finish`."
    )

    content.append({"type": "input_text", "text": "\n".join(observation_lines)})
    return NeMoGymEasyInputMessage(role="user", content=content)


def _strip_image_blocks(result: NeMoGymResponse) -> NeMoGymResponse:
    data = result.model_dump()
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get("type") == "input_image")
            ]
    return NeMoGymResponse.model_validate(data)


class OSWorldVLMAgent(SimpleAgent):
    config: OSWorldVLMAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None
        resources_server_cookies = request.cookies

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            model_response = NeMoGymResponse.model_validate(model_response_json)

            output = model_response.output
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage
                model_response.usage = None
            if usage and model_response.usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details:
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                item for item in output if item.type == "function_call"
            ]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                item
                for item in output
                if item.type == "message" and item.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    new_outputs.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=output_function_call.call_id,
                            output=json.dumps(
                                {"error": f"Invalid tool call arguments: {exc!r}"}
                            ),
                        )
                    )
                    continue

                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=parsed_arguments,
                    cookies=resources_server_cookies,
                )
                resources_server_cookies = api_response.cookies
                raw_tool_output = (await api_response.content.read()).decode()
                observation = _extract_observation_from_tool_output(raw_tool_output)

                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=_compact_tool_output_with_observation(
                            raw_tool_output, observation
                        ),
                    )
                )

                if observation:
                    new_outputs.append(
                        _observation_to_message(observation, self.config)
                    )

            if self.config.max_steps and step >= self.config.max_steps:
                break

        combined_cookies = []
        if resources_server_cookies:
            combined_cookies.extend(resources_server_cookies.items())
        if model_server_cookies:
            combined_cookies.extend(model_server_cookies.items())
        for key, value in combined_cookies:
            response.set_cookie(key, value)

        model_response.output = new_outputs
        model_response.usage = usage
        if self.config.strip_images_from_output:
            model_response = _strip_image_blocks(model_response)
        return model_response

    async def run(
        self, request: Request, body: SimpleAgentRunRequest
    ) -> SimpleAgentVerifyResponse:
        return await super().run(request, body)


if __name__ == "__main__":
    OSWorldVLMAgent.run_webserver()
