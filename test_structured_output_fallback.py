"""Tests for the opt-in structured-output fallback hook.

The hook has two halves, and both are covered here:

* ``agents_config`` parses ``AGENT_STRUCTURED_OUTPUT_MODE`` /
  ``AGENT_STRUCTURED_OUTPUT_MAX_TOKENS`` and refuses combinations the selected
  SDK route cannot serve.
* ``structured_output_fallback.install`` wraps ``OpenAIProvider.get_model`` so
  every Chat Completions model the SDK builds is handed back as a
  ``FallbackChatCompletionsModel``.

The second half patches a class attribute on the SDK, so every test restores
the pristine factory in ``tearDown``; without that, one test's wrapper would
leak into the rest of the suite.

A third half covers the DeepSeek JSON Output request adaptation itself, at the
HTTP level: an ``httpx2.MockTransport`` records the exact body the OpenAI client
serializes, because ``extra_body`` precedence is an OpenAI SDK merge behaviour
that only a real client proves.
"""

import asyncio
import importlib
import json
import os
import pathlib
import re
import sys
import unittest
from typing import Any, Literal
from unittest.mock import patch

import httpx2
from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.agent_output import AgentOutputSchema
from agents.exceptions import ModelBehaviorError
from agents.handoffs import handoff
from agents.items import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ToolCallItem,
)
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import FunctionTool
from agents.tracing import setup as tracing_setup
from openai import AsyncOpenAI
from pydantic import BaseModel, create_model

import structured_output_fallback
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider

# Captured before any test installs a wrapper, so it is the factory the Agents
# SDK itself defines.
_PRISTINE_GET_MODEL = OpenAIProvider.get_model

_BASE_URL = "https://provider.example.com/v1"

# Long enough to break PEP 8 when written inline twice below.
_MAX_TOKENS_ERROR = "AGENT_STRUCTURED_OUTPUT_MAX_TOKENS"

# Environment keys the configuration tests own; anything not named in a test is
# removed so the result cannot depend on the developer's own .env.
_CONFIG_ENV_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_AGENTS_CHAT_API",
    "AGENT_STRUCTURED_OUTPUT_MODE",
    "AGENT_STRUCTURED_OUTPUT_MAX_TOKENS",
)


def _reset_installation() -> None:
    """Undo :func:`structured_output_fallback.install` between tests."""
    OpenAIProvider.get_model = _PRISTINE_GET_MODEL
    structured_output_fallback._mode = None
    structured_output_fallback._fallback_max_tokens = None
    structured_output_fallback._installed = False
    structured_output_fallback._model_caches.clear()


def _reload_config(**values: str):
    """Re-import ``agents_config`` with only ``values`` set for these keys.

    The module reads its environment once, at import time, so reloading is the
    only way to exercise another configuration. Importing it also loads .env
    and exports ``OPENAI_DEFAULT_MODEL`` as a side effect, so it is imported
    inside the patched environment instead of at the top of this file: that
    keeps those variables out of the environment the other test modules see.
    """
    with patch.dict(os.environ, values):
        for key in _CONFIG_ENV_KEYS:
            if key not in values:
                os.environ.pop(key, None)
        module = sys.modules.get("agents_config")
        if module is None:
            return importlib.import_module("agents_config")
        return importlib.reload(module)


class ConfigurationTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_installation()
        _reload_config()

    def test_unset_mode_keeps_the_native_sdk_factory(self):
        with patch.object(structured_output_fallback, "install") as install:
            module = _reload_config(OPENAI_BASE_URL=_BASE_URL)

        self.assertIsNone(module.STRUCTURED_OUTPUT_MODE)
        self.assertIsNone(module.STRUCTURED_OUTPUT_MAX_TOKENS)
        install.assert_not_called()
        self.assertIs(OpenAIProvider.get_model, _PRISTINE_GET_MODEL)

    def test_deepseek_json_mode_is_accepted(self):
        with patch.object(structured_output_fallback, "install") as install:
            module = _reload_config(
                OPENAI_BASE_URL=_BASE_URL,
                AGENT_STRUCTURED_OUTPUT_MODE="deepseek_json",
            )

        self.assertEqual(module.STRUCTURED_OUTPUT_MODE, "deepseek_json")
        install.assert_called_once_with("deepseek_json", None)

    def test_minimax_function_mode_is_accepted(self):
        with patch.object(structured_output_fallback, "install") as install:
            module = _reload_config(
                OPENAI_BASE_URL=_BASE_URL,
                AGENT_STRUCTURED_OUTPUT_MODE="minimax_function",
            )

        self.assertEqual(module.STRUCTURED_OUTPUT_MODE, "minimax_function")
        install.assert_called_once_with("minimax_function", None)

    def test_token_budget_is_read_as_a_positive_integer(self):
        with patch.object(structured_output_fallback, "install") as install:
            module = _reload_config(
                OPENAI_BASE_URL=_BASE_URL,
                AGENT_STRUCTURED_OUTPUT_MODE="deepseek_json",
                AGENT_STRUCTURED_OUTPUT_MAX_TOKENS="2048",
            )

        self.assertEqual(module.STRUCTURED_OUTPUT_MAX_TOKENS, 2048)
        install.assert_called_once_with("deepseek_json", 2048)

    def test_unknown_mode_is_rejected(self):
        with patch.object(structured_output_fallback, "install") as install:
            with self.assertRaisesRegex(ValueError, "deepseek_json"):
                _reload_config(
                    OPENAI_BASE_URL=_BASE_URL,
                    AGENT_STRUCTURED_OUTPUT_MODE="json_schema",
                )

        install.assert_not_called()

    def test_non_integer_token_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, _MAX_TOKENS_ERROR):
            _reload_config(
                OPENAI_BASE_URL=_BASE_URL,
                AGENT_STRUCTURED_OUTPUT_MODE="deepseek_json",
                AGENT_STRUCTURED_OUTPUT_MAX_TOKENS="plenty",
            )

    def test_non_positive_token_budget_is_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, _MAX_TOKENS_ERROR
                ):
                    _reload_config(
                        OPENAI_BASE_URL=_BASE_URL,
                        AGENT_STRUCTURED_OUTPUT_MODE="deepseek_json",
                        AGENT_STRUCTURED_OUTPUT_MAX_TOKENS=value,
                    )

    def test_fallback_mode_on_the_responses_route_is_rejected(self):
        with patch.object(structured_output_fallback, "install") as install:
            with self.assertRaisesRegex(ValueError, "Chat Completions"):
                _reload_config(
                    OPENAI_BASE_URL=_BASE_URL,
                    OPENAI_AGENTS_CHAT_API="0",
                    AGENT_STRUCTURED_OUTPUT_MODE="deepseek_json",
                )

        install.assert_not_called()

    def test_import_hook_installs_the_adapter(self):
        _reload_config(
            OPENAI_BASE_URL=_BASE_URL,
            AGENT_STRUCTURED_OUTPUT_MODE="minimax_function",
        )

        self.assertIs(
            OpenAIProvider.get_model,
            structured_output_fallback._get_model_with_structured_fallback,
        )


class FallbackRegistrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_installation()

    def _provider(self, **kwargs) -> OpenAIProvider:
        return OpenAIProvider(
            api_key="test-key",
            base_url=_BASE_URL,
            use_responses=False,
            **kwargs,
        )

    def test_explicit_model_name_is_adapted(self):
        structured_output_fallback.install("deepseek_json", None)

        model = self._provider().get_model("deepseek-chat")

        self.assertIsInstance(
            model, structured_output_fallback.FallbackChatCompletionsModel
        )
        self.assertIsInstance(model, OpenAIChatCompletionsModel)
        self.assertEqual(model.model, "deepseek-chat")

    def test_default_model_name_is_adapted(self):
        from agents.models.default_models import get_default_model

        structured_output_fallback.install("minimax_function", 2048)

        model = self._provider().get_model(None)

        self.assertIsInstance(
            model, structured_output_fallback.FallbackChatCompletionsModel
        )
        self.assertIsInstance(model, OpenAIChatCompletionsModel)
        self.assertEqual(model.model, get_default_model())
        self.assertEqual(model._fallback_max_tokens, 2048)

    def test_adapter_keeps_the_client_and_feature_flags(self):
        provider = self._provider(
            strict_feature_validation=True, buffer_streamed_tool_calls=True
        )
        original = _PRISTINE_GET_MODEL(provider, "deepseek-chat")

        structured_output_fallback.install("deepseek_json", None)
        adapted = provider.get_model("deepseek-chat")

        self.assertIs(adapted._client, original._client)
        self.assertEqual(
            adapted._strict_feature_validation,
            original._strict_feature_validation,
        )
        self.assertEqual(
            adapted._buffer_streamed_tool_calls,
            original._buffer_streamed_tool_calls,
        )

    def test_adapter_is_cached_per_provider_and_model(self):
        structured_output_fallback.install("deepseek_json", None)
        provider = self._provider()

        first = provider.get_model("deepseek-chat")
        self.assertIs(first, provider.get_model("deepseek-chat"))
        self.assertIsNot(first, provider.get_model("glm-4-plus"))

    def test_mode_change_replaces_already_cached_adapters(self):
        structured_output_fallback.install("deepseek_json", None)
        provider = self._provider()
        cached = provider.get_model("deepseek-chat")
        self.assertEqual(cached._fallback_mode, "deepseek_json")

        structured_output_fallback.install("minimax_function", 2048)
        adapted = provider.get_model("deepseek-chat")

        self.assertIsNot(adapted, cached)
        self.assertEqual(adapted._fallback_mode, "minimax_function")
        self.assertEqual(adapted._fallback_max_tokens, 2048)

    def test_token_budget_change_replaces_already_cached_adapters(self):
        structured_output_fallback.install("deepseek_json", 1024)
        provider = self._provider()
        cached = provider.get_model("deepseek-chat")
        self.assertEqual(cached._fallback_max_tokens, 1024)

        structured_output_fallback.install("deepseek_json", 4096)
        adapted = provider.get_model("deepseek-chat")

        self.assertIsNot(adapted, cached)
        self.assertEqual(adapted._fallback_max_tokens, 4096)

    def test_reinstalling_the_same_settings_keeps_the_cached_adapter(self):
        structured_output_fallback.install("deepseek_json", 1024)
        provider = self._provider()
        cached = provider.get_model("deepseek-chat")

        structured_output_fallback.install("deepseek_json", 1024)

        self.assertIs(provider.get_model("deepseek-chat"), cached)

    def test_install_is_idempotent(self):
        structured_output_fallback.install("deepseek_json", None)
        installed = OpenAIProvider.get_model

        structured_output_fallback.install("minimax_function", 2048)

        self.assertIs(OpenAIProvider.get_model, installed)
        self.assertIs(
            structured_output_fallback._original_get_model, _PRISTINE_GET_MODEL
        )
        self.assertEqual(structured_output_fallback._mode, "minimax_function")

        model = self._provider().get_model("deepseek-chat")
        self.assertIs(
            type(model),
            structured_output_fallback.FallbackChatCompletionsModel,
        )

    def test_responses_models_are_returned_unchanged(self):
        from agents.models.openai_responses import OpenAIResponsesModel

        structured_output_fallback.install("deepseek_json", None)
        provider = OpenAIProvider(
            api_key="test-key", base_url=_BASE_URL, use_responses=True
        )

        model = provider.get_model("gpt-4o")

        self.assertIsInstance(model, OpenAIResponsesModel)
        self.assertNotIsInstance(
            model, structured_output_fallback.FallbackChatCompletionsModel
        )

    def test_unset_mode_returns_the_original_sdk_model(self):
        structured_output_fallback.install("deepseek_json", None)
        structured_output_fallback._mode = None

        model = self._provider().get_model("deepseek-chat")

        self.assertIs(type(model), OpenAIChatCompletionsModel)
        self.assertNotIsInstance(
            model, structured_output_fallback.FallbackChatCompletionsModel
        )


class ResearchStepModel(BaseModel):
    title: str
    estimated_minutes: int


class ResearchPlanModel(BaseModel):
    """A nested typed output: it exercises ``$defs``/``$ref`` and enums."""

    topic: str
    lead_step: ResearchStepModel
    status: Literal["draft", "review", "final"]
    risks: list[str]


class RecursiveNodeModel(BaseModel):
    """A typed output no finite JSON example can represent."""

    child: "RecursiveNodeModel"


RecursiveNodeModel.model_rebuild()


# A nesting depth no declared example schema reaches, but which must still be
# describable: it is finite, so only a real cycle may be refused.
_DEEP_LEVELS = 12


def _deep_output_type(levels: int = _DEEP_LEVELS):
    """Build ``levels`` nested Pydantic models; the innermost holds a string."""
    current = create_model("DeepLeaf", value=(str, ...))
    for index in range(levels - 1):
        current = create_model(f"DeepLevel{index}", child=(current, ...))
    return current


def _deep_payload(levels: int = _DEEP_LEVELS) -> dict[str, object]:
    payload: dict[str, object] = {"value": "bottom"}
    for _ in range(levels - 1):
        payload = {"child": payload}
    return payload


def _chain_schema(levels: int) -> dict[str, object]:
    """A ``$defs`` chain ``levels`` deep, with no reference repeated."""
    defs: dict[str, object] = {}
    for level in range(levels):
        if level == levels - 1:
            defs[f"L{level}"] = {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        else:
            defs[f"L{level}"] = {
                "type": "object",
                "properties": {"child": {"$ref": f"#/$defs/L{level + 1}"}},
                "required": ["child"],
            }
    return {"$defs": defs, "$ref": "#/$defs/L0"}


def _doubling_schema(levels: int) -> dict[str, object]:
    """An acyclic schema whose example still doubles at every level."""
    defs: dict[str, object] = {}
    for level in range(levels):
        if level == levels - 1:
            defs[f"D{level}"] = {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        else:
            next_ref = {"$ref": f"#/$defs/D{level + 1}"}
            defs[f"D{level}"] = {
                "type": "object",
                "properties": {"left": next_ref, "right": next_ref},
                "required": ["left", "right"],
            }
    return {"$defs": defs, "$ref": "#/$defs/D0"}


class DeepSeekFallbackTests(unittest.TestCase):
    """The DeepSeek JSON Output adaptation, checked on the serialized request.

    Every test builds the adapter directly instead of going through
    ``install``: what is under test is the request the adapter produces, not
    how the adapter is selected.
    """

    def setUp(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.reply = json.dumps(
            {
                "topic": "structured output",
                "lead_step": {
                    "title": "compare providers",
                    "estimated_minutes": 30,
                },
                "status": "draft",
                "risks": ["provider drift"],
            }
        )
        self.finish_reason = "stop"
        self.refusal: str | None = None
        # Assistant messages the provider returns, one per request. Empty means
        # every request is answered with the canned ``self.reply`` message.
        self.script: list[dict[str, object]] = []
        self._clients: list[AsyncOpenAI] = []

        # Tracing defaults to a processor that exports to api.openai.com, so it
        # is replaced by a disabled provider for the duration of each test.
        self._previous_trace_provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        tracing_setup.GLOBAL_TRACE_PROVIDER = None
        set_tracing_disabled(True)

    def tearDown(self) -> None:
        tracing_setup.GLOBAL_TRACE_PROVIDER = self._previous_trace_provider
        asyncio.run(self._close_clients())

    async def _close_clients(self) -> None:
        for client in self._clients:
            await client.close()

    def _handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.script:
            message = dict(self.script.pop(0))
        else:
            message = {"role": "assistant", "content": self.reply}
            if self.refusal is not None:
                message["refusal"] = self.refusal
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": self.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    def _model(self, mode: str = "deepseek_json", max_tokens: int | None = 2048):
        client = AsyncOpenAI(
            api_key="test-key",
            base_url=_BASE_URL,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(self._handle_request)
            ),
        )
        self._clients.append(client)
        return structured_output_fallback.FallbackChatCompletionsModel(
            model="deepseek-chat",
            openai_client=client,
            mode=mode,
            fallback_max_tokens=max_tokens,
        )

    def _run(
        self,
        *,
        model=None,
        output_type=ResearchPlanModel,
        model_settings=None,
        tools=None,
    ):
        kwargs = {}
        if model_settings is not None:
            kwargs["model_settings"] = model_settings
        if tools is not None:
            kwargs["tools"] = tools
        agent = Agent(
            name="planner",
            instructions="Plan the research.",
            model=model or self._model(),
            output_type=output_type,
            **kwargs,
        )
        return asyncio.run(Runner.run(agent, "Plan some research."))

    def _call(
        self,
        arguments: str,
        name: str = "emit_typed_output",
        call_id: str = "call_1",
    ) -> dict[str, object]:
        """One Chat Completions tool call, in the provider's wire shape."""
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def _one_request(self) -> dict[str, object]:
        self.assertEqual(len(self.requests), 1, "expected exactly one request")
        return self.requests[0]

    # --- the serialized request -------------------------------------------

    def test_request_asks_for_json_object_not_json_schema(self):
        self._run()

        self.assertEqual(
            self._one_request()["response_format"], {"type": "json_object"}
        )

    def test_json_schema_response_format_is_not_sent(self):
        self._run()

        body = self._one_request()
        self.assertNotIn("json_schema", body)
        self.assertNotIn("json_schema", body["response_format"])

    def test_system_instruction_carries_the_schema_and_an_example(self):
        self._run()

        system = self._one_request()["messages"][0]["content"]
        schema = AgentOutputSchema(
            ResearchPlanModel, strict_json_schema=True
        ).json_schema()

        # DeepSeek's JSON mode rejects a request whose prompt never says JSON.
        self.assertIn("JSON", system)
        self.assertIn(json.dumps(schema), system)
        # The example is what the model copies; it must be the real shape.
        self.assertIn(
            json.dumps(
                {
                    "topic": "",
                    "lead_step": {"title": "", "estimated_minutes": 0},
                    "status": "draft",
                    "risks": [""],
                }
            ),
            system,
        )

    def test_unrelated_extra_body_entries_are_preserved(self):
        self._run(
            model_settings=ModelSettings(
                extra_body={"thinking": {"type": "disabled"}}
            )
        )

        body = self._one_request()
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_typed_output_is_still_validated_by_the_original_schema(self):
        result = self._run()

        self.assertEqual(result.final_output.topic, "structured output")
        self.assertEqual(result.final_output.status, "draft")
        self.assertEqual(result.final_output.lead_step.estimated_minutes, 30)

    def test_agent_instructions_are_kept_behind_the_formatting_ones(self):
        self._run()

        system = self._one_request()["messages"][0]["content"]
        self.assertIn("Plan the research.", system)

    # --- the token budget --------------------------------------------------

    def test_agent_max_tokens_wins_over_the_fallback_budget(self):
        self._run(
            model=self._model(max_tokens=2048),
            model_settings=ModelSettings(max_tokens=123),
        )

        self.assertEqual(self._one_request()["max_tokens"], 123)

    def test_fallback_budget_is_applied_when_the_agent_sets_none(self):
        self._run(model=self._model(max_tokens=2048))

        self.assertEqual(self._one_request()["max_tokens"], 2048)

    def test_missing_token_budget_fails_before_the_request(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            self._run(
                model=self._model(max_tokens=None),
                model_settings=ModelSettings(),
            )

        self.assertEqual(self.requests, [])

    # --- provider output that is not the schema ----------------------------

    def test_truncated_json_fails_visibly(self):
        self.reply = '{"topic": "half'

        with self.assertRaisesRegex(ModelBehaviorError, "Invalid JSON"):
            self._run()

    def test_truncation_fails_before_schema_validation(self):
        # finish_reason="length" is the SDK's own guard: it never reaches the
        # schema, so this only proves the budget exhaustion is still visible.
        self.reply = ""
        self.finish_reason = "length"

        with self.assertRaises(ModelBehaviorError):
            self._run()

    def test_empty_json_content_fails_immediately(self):
        """The documented DeepSeek failure: HTTP 200, empty JSON content.

        The runner only validates output text it found, so an empty completion
        would otherwise be retried until MaxTurnsExceeded. One request and a
        ModelBehaviorError is the visible failure the design asks for.
        """
        self.reply = ""
        self.finish_reason = "stop"

        with self.assertRaisesRegex(ModelBehaviorError, "DeepSeek JSON mode"):
            self._run()

        self.assertEqual(len(self.requests), 1, "the run must not retry")

    def test_refusal_output_item_is_preserved(self):
        """A refusal is terminal too: it is output, not missing output."""
        self.reply = ""
        self.finish_reason = "stop"
        self.refusal = "I cannot help with that."
        schema = AgentOutputSchema(ResearchPlanModel, strict_json_schema=True)

        response = asyncio.run(
            self._model().get_response(
                "Plan the research.",
                "Plan some research.",
                ModelSettings(),
                [],
                schema,
                [],
                ModelTracing.DISABLED,
            )
        )

        refusals = [
            part
            for item in response.output
            if isinstance(item, ResponseOutputMessage)
            for part in item.content
            if isinstance(part, ResponseOutputRefusal)
        ]
        self.assertEqual(
            [part.refusal for part in refusals], ["I cannot help with that."]
        )

    # --- requests the adaptation must leave alone --------------------------

    def test_plain_text_agents_keep_the_native_request(self):
        self.reply = "Here is the plan."
        self._run(output_type=str, model=self._model())

        self.assertNotIn("response_format", self._one_request())

    def test_minimax_mode_tools_run_a_work_phase_then_a_formatter(self):
        """A MiniMax tool agent Task 3 does not cover takes the two-phase path."""

        @function_tool
        def lookup_library(name: str) -> str:
            """Look a library up in the catalogue."""
            return "found"

        self.script = [
            {"role": "assistant", "content": "Found httpx2."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._call(self.reply)],
            },
        ]

        self._run(
            model=self._model(mode="minimax_function"), tools=[lookup_library]
        )

        work, formatter = self.requests
        self.assertNotIn("response_format", work)
        self.assertTrue(work["tools"])
        self.assertNotIn(
            structured_output_fallback._FORMAT_TOOL_NAME,
            json.dumps(work["tools"]),
        )
        self.assertEqual(
            formatter["tool_choice"],
            {
                "type": "function",
                "function": {
                    "name": structured_output_fallback._FORMAT_TOOL_NAME
                },
            },
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in formatter["tools"]],
            [structured_output_fallback._FORMAT_TOOL_NAME],
        )

    def test_agents_with_tools_run_a_work_phase_then_a_formatter(self):
        @function_tool
        def lookup_library(name: str) -> str:
            """Look a library up in the catalogue."""
            return "found"

        self._run(tools=[lookup_library])

        work, formatter = self.requests
        # The work phase asks for no structured output at all, so the provider
        # cannot skip the tool call the way it does under a response format.
        self.assertNotIn("response_format", work)
        self.assertTrue(work["tools"])
        self.assertEqual(formatter["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", formatter)

    def test_recursive_output_schema_fails_before_the_request(self):
        with self.assertRaisesRegex(ValueError, "recursive"):
            self._run(output_type=RecursiveNodeModel)

        self.assertEqual(self.requests, [])

    def test_deeply_nested_output_type_is_illustrated_and_validated(self):
        """Finite nesting is not recursion: the request still goes out."""
        self.reply = json.dumps(_deep_payload())

        result = self._run(output_type=_deep_output_type())

        body = self._one_request()
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn('"value": ""', body["messages"][0]["content"])
        node = result.final_output
        while hasattr(node, "child"):
            node = node.child
        self.assertEqual(node.value, "bottom")

    # --- the JSON example helper -------------------------------------------

    def test_example_is_a_json_value_not_only_an_object(self):
        for schema, expected in (
            ({"type": "string"}, ""),
            ({"type": "integer"}, 0),
            ({"type": "number"}, 0.0),
            ({"type": "boolean"}, False),
            ({"type": "null"}, None),
            ({"type": "array", "items": {"type": "string"}}, [""]),
        ):
            with self.subTest(schema=schema):
                self.assertEqual(
                    structured_output_fallback._json_schema_example(schema),
                    expected,
                )

    def test_example_fills_required_properties_only(self):
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }

        self.assertEqual(
            structured_output_fallback._json_schema_example(schema),
            {"required_field": ""},
        )

    def test_example_prefers_const_enum_and_default(self):
        for schema, expected in (
            ({"const": 7}, 7),
            ({"enum": ["draft", "final"]}, "draft"),
            ({"type": "integer", "default": 42}, 42),
        ):
            with self.subTest(schema=schema):
                self.assertEqual(
                    structured_output_fallback._json_schema_example(schema),
                    expected,
                )

    def test_example_resolves_local_refs(self):
        schema = {
            "$defs": {
                "Step": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                }
            },
            "type": "object",
            "properties": {
                "step": {"$ref": "#/$defs/Step"},
                "steps": {"type": "array", "items": {"$ref": "#/$defs/Step"}},
            },
            "required": ["step", "steps"],
        }

        self.assertEqual(
            structured_output_fallback._json_schema_example(schema),
            {"step": {"title": ""}, "steps": [{"title": ""}]},
        )

    def test_example_skips_the_null_branch_of_a_union(self):
        self.assertEqual(
            structured_output_fallback._json_schema_example(
                {"anyOf": [{"type": "null"}, {"type": "integer"}]}
            ),
            0,
        )
        self.assertIsNone(
            structured_output_fallback._json_schema_example(
                {"oneOf": [{"type": "null"}]}
            )
        )

    def test_example_builds_a_deeply_nested_finite_schema(self):
        """Deep is allowed: only a repeated reference is recursion."""
        example = structured_output_fallback._json_schema_example(
            _chain_schema(_DEEP_LEVELS)
        )

        depth = 0
        node = example
        while isinstance(node, dict) and "child" in node:
            depth += 1
            node = node["child"]
        self.assertEqual(depth, _DEEP_LEVELS - 1)
        self.assertEqual(node, {"value": ""})

    def test_example_rejects_a_schema_too_large_to_illustrate(self):
        """Acyclic but exploding: a size error, not a recursion error."""
        with self.assertRaises(ValueError) as raised:
            structured_output_fallback._json_schema_example(_doubling_schema(12))

        self.assertIn("too deep or too wide", str(raised.exception))
        self.assertNotIn("recursive", str(raised.exception))

    def test_example_rejects_recursive_schemas(self):
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                    "required": ["child"],
                }
            },
            "$ref": "#/$defs/Node",
        }

        with self.assertRaises(ValueError) as raised:
            structured_output_fallback._json_schema_example(schema)

        self.assertIn("recursive", str(raised.exception))
        self.assertNotIn("too deep or too wide", str(raised.exception))

    def test_example_rejects_unsupported_shapes(self):
        for schema in (
            {},
            {"type": "unsupported"},
            {"$ref": "https://example.com/other.json"},
            {"$defs": {}, "$ref": "#/$defs/Missing"},
        ):
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError):
                    structured_output_fallback._json_schema_example(schema)


class MiniMaxFallbackTests(unittest.TestCase):
    """The MiniMax Function Calling adaptation at the HTTP level.

    The provider is driven with a mock Chat Completion that answers with a
    function call, so these tests cover both halves of the protocol: the
    request that forces the synthetic formatting function, and the conversion
    of its arguments back into the assistant text the typed-output validator
    consumes.
    """

    def setUp(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.arguments = json.dumps(
            {
                "topic": "structured output",
                "lead_step": {
                    "title": "compare providers",
                    "estimated_minutes": 30,
                },
                "status": "draft",
                "risks": ["provider drift"],
            }
        )
        self.tool_calls: list[dict[str, object]] | None = [
            self._call(self.arguments)
        ]
        # Ordinary assistant text: what the provider sends instead of a call.
        self.reply: str | None = None
        self.refusal: str | None = None
        # Assistant messages the provider returns, one per request. Empty means
        # every request is answered with the canned ``self.reply`` message.
        self.script: list[dict[str, object]] = []
        self._clients: list[AsyncOpenAI] = []

        self._previous_trace_provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        tracing_setup.GLOBAL_TRACE_PROVIDER = None
        set_tracing_disabled(True)

    def tearDown(self) -> None:
        tracing_setup.GLOBAL_TRACE_PROVIDER = self._previous_trace_provider
        asyncio.run(self._close_clients())

    async def _close_clients(self) -> None:
        for client in self._clients:
            await client.close()

    # --- fixtures ---------------------------------------------------------

    def _call(
        self,
        arguments: str,
        name: str = "emit_typed_output",
        call_id: str = "call_1",
    ) -> dict[str, object]:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def _schema(self) -> AgentOutputSchema:
        return AgentOutputSchema(ResearchPlanModel, strict_json_schema=True)

    def _handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.script:
            message = dict(self.script.pop(0))
        else:
            message: dict[str, object] = {
                "role": "assistant",
                "content": self.reply,
            }
            if self.refusal is not None:
                message["refusal"] = self.refusal
            if self.tool_calls is not None:
                message["tool_calls"] = self.tool_calls
        return httpx2.Response(
            200,
            headers={"x-request-id": "req-minimax"},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "MiniMax-Text-01",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    def _model(
        self, mode: str = "minimax_function", max_tokens: int | None = None
    ):
        client = AsyncOpenAI(
            api_key="test-key",
            base_url=_BASE_URL,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(self._handle_request)
            ),
        )
        self._clients.append(client)
        return structured_output_fallback.FallbackChatCompletionsModel(
            model="MiniMax-Text-01",
            openai_client=client,
            mode=mode,
            fallback_max_tokens=max_tokens,
        )

    def _run(self, *, model=None, output_type=ResearchPlanModel, **kwargs):
        agent = Agent(
            name="planner",
            instructions="Plan the research.",
            model=model or self._model(),
            output_type=output_type,
            **kwargs,
        )
        return asyncio.run(Runner.run(agent, "Plan some research."))

    def _get_response(self, *, model_settings=None, output_schema=None):
        return asyncio.run(
            self._model().get_response(
                "Plan the research.",
                "Plan some research.",
                ModelSettings() if model_settings is None else model_settings,
                [],
                self._schema() if output_schema is None else output_schema,
                [],
                ModelTracing.DISABLED,
            )
        )

    def _one_request(self) -> dict[str, object]:
        self.assertEqual(len(self.requests), 1, "expected exactly one request")
        return self.requests[0]

    def _only_function(self, body: dict[str, object]) -> dict[str, object]:
        tools = body["tools"]
        self.assertEqual(len(tools), 1, "expected exactly one tool")
        return tools[0]["function"]

    # --- the serialized request -------------------------------------------

    def test_request_forces_only_the_formatting_function(self):
        self._run()

        body = self._one_request()
        name = structured_output_fallback._FORMAT_TOOL_NAME
        self.assertEqual(
            body["tool_choice"],
            {"type": "function", "function": {"name": name}},
        )
        self.assertEqual(
            self._only_function(body)["name"],
            structured_output_fallback._FORMAT_TOOL_NAME,
        )

    def test_formatting_tool_schema_is_the_output_schema(self):
        self._run()

        function = self._only_function(self._one_request())
        self.assertEqual(function["parameters"], self._schema().json_schema())
        self.assertFalse(function["strict"])

    def test_native_json_schema_response_format_is_not_sent(self):
        self._run()

        body = self._one_request()
        self.assertNotIn("response_format", body)
        self.assertNotIn("json_schema", json.dumps(body))

    def test_caller_response_format_in_extra_body_is_dropped(self):
        self._run(
            model_settings=ModelSettings(
                extra_body={
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                }
            )
        )

        body = self._one_request()
        self.assertNotIn("response_format", body)
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_unrelated_model_settings_are_preserved(self):
        self._run(
            model_settings=ModelSettings(
                temperature=0.25, top_p=0.5, max_tokens=123, presence_penalty=0.75
            )
        )

        body = self._one_request()
        self.assertEqual(body["temperature"], 0.25)
        self.assertEqual(body["top_p"], 0.5)
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(body["presence_penalty"], 0.75)

    # --- the converted response -------------------------------------------

    def test_valid_arguments_become_the_typed_output(self):
        result = self._run()

        self.assertEqual(result.final_output.topic, "structured output")
        self.assertEqual(result.final_output.status, "draft")
        self.assertEqual(result.final_output.lead_step.estimated_minutes, 30)
        self.assertEqual(len(self.requests), 1)

    def test_function_call_is_converted_to_output_text(self):
        response = self._get_response()

        self.assertEqual(len(response.output), 1)
        message = response.output[0]
        self.assertIsInstance(message, ResponseOutputMessage)
        self.assertEqual(len(message.content), 1)
        text = message.content[0]
        self.assertIsInstance(text, ResponseOutputText)
        self.assertEqual(text.text, self.arguments)
        self.assertEqual(response.usage.requests, 1)
        self.assertEqual(response.usage.total_tokens, 18)

    def test_request_id_and_raw_usage_are_preserved(self):
        response = self._get_response(
            model_settings=ModelSettings(preserve_raw_usage=True)
        )

        self.assertEqual(response.request_id, "req-minimax")
        self.assertEqual(response.raw_usage["total_tokens"], 18)

    def test_missing_function_call_fails_visibly(self):
        self.tool_calls = None
        self.reply = "Here is the plan instead."

        with self.assertRaisesRegex(ModelBehaviorError, "emit_typed_output"):
            self._run()

    def test_unrelated_function_name_fails_visibly(self):
        self.tool_calls = [self._call(self.arguments, name="lookup_library")]

        with self.assertRaisesRegex(ModelBehaviorError, "emit_typed_output"):
            self._run()

    def test_multiple_formatting_calls_fail_visibly(self):
        self.tool_calls = [
            self._call(self.arguments),
            self._call(self.arguments, call_id="call_2"),
        ]

        with self.assertRaisesRegex(ModelBehaviorError, "emit_typed_output"):
            self._run()

    def test_invalid_argument_json_fails_visibly(self):
        self.tool_calls = [self._call('{"topic": "half')]

        with self.assertRaises(ModelBehaviorError):
            self._run()

    def test_schema_mismatch_fails_visibly(self):
        self.tool_calls = [self._call(json.dumps({"topic": "structured output"}))]

        with self.assertRaises(ModelBehaviorError):
            self._run()

    def test_refusal_fails_visibly(self):
        self.tool_calls = None
        self.refusal = "I cannot help with that."

        with self.assertRaisesRegex(ModelBehaviorError, "refusal"):
            self._run()

    def test_empty_output_fails_visibly(self):
        """HTTP 200 with neither a call nor text: not a typed-output pass."""
        self.tool_calls = None
        self.reply = None

        with self.assertRaisesRegex(ModelBehaviorError, "emit_typed_output"):
            self._run()

    def test_no_synthetic_tool_call_reaches_the_runner(self):
        result = self._run()

        tool_calls = [
            item for item in result.new_items if isinstance(item, ToolCallItem)
        ]
        self.assertEqual(tool_calls, [])
        self.assertIsInstance(result.final_output, ResearchPlanModel)

    def test_formatting_tool_invoker_raises_if_called(self):
        tool = structured_output_fallback._format_tool(self._schema())

        with self.assertRaises(ModelBehaviorError):
            asyncio.run(tool.on_invoke_tool(None, self.arguments))

    # --- requests the adaptation must leave alone --------------------------

    def test_plain_text_agents_keep_the_native_request(self):
        self.tool_calls = None
        self.reply = "Here is the plan."
        self._run(output_type=str)

        body = self._one_request()
        self.assertNotIn("tools", body)
        self.assertNotIn("response_format", body)

    def test_agents_with_tools_run_a_work_phase_then_a_formatter(self):
        @function_tool
        def lookup_library(name: str) -> str:
            """Look a library up in the catalogue."""
            return "found"

        self.script = [
            {"role": "assistant", "content": "Found httpx2."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._call(self.arguments)],
            },
        ]

        self._run(tools=[lookup_library])

        work, formatter = self.requests
        self.assertNotIn("response_format", work)
        self.assertEqual(
            [tool["function"]["name"] for tool in work["tools"]],
            ["lookup_library"],
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in formatter["tools"]],
            [structured_output_fallback._FORMAT_TOOL_NAME],
        )

    def test_agents_with_handoffs_run_a_work_phase_then_a_formatter(self):
        specialist = Agent(name="specialist", model=self._model())

        self.script = [
            {"role": "assistant", "content": "Handing over."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._call(self.arguments)],
            },
        ]

        self._run(handoffs=[specialist])

        work, formatter = self.requests
        self.assertNotIn("response_format", work)
        self.assertEqual(
            [tool["function"]["name"] for tool in work["tools"]],
            ["transfer_to_specialist"],
        )
        # A handoff is a tool in this phase only: the formatter must not be
        # able to hand the turn over.
        self.assertEqual(
            [tool["function"]["name"] for tool in formatter["tools"]],
            [structured_output_fallback._FORMAT_TOOL_NAME],
        )


# The request IDs the two-phase mock provider reports, one per physical call.
_REQUEST_IDS = {1: "req-work", 2: "req-format"}


class TwoPhaseFallbackTests(unittest.TestCase):
    """Tool-, MCP- and handoff-bearing typed agents under either fallback mode.

    A typed request that also carries tools or handoffs is split in two: the
    work phase runs the agent's own tools, and only a terminal assistant answer
    is formatted in a second, tool-less request. These tests drive both phases
    at the HTTP level, so they prove what each phase actually puts on the wire
    and that nothing the runner owns is rewritten on the way through.
    """

    def setUp(self) -> None:
        self.requests: list[dict[str, object]] = []
        # One assistant message per request, in order.
        self.script: list[dict[str, object]] = []
        self.arguments = json.dumps(
            {
                "topic": "structured output",
                "lead_step": {
                    "title": "compare providers",
                    "estimated_minutes": 30,
                },
                "status": "draft",
                "risks": ["provider drift"],
            }
        )
        self._clients: list[AsyncOpenAI] = []

        self._previous_trace_provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        tracing_setup.GLOBAL_TRACE_PROVIDER = None
        set_tracing_disabled(True)

    def tearDown(self) -> None:
        tracing_setup.GLOBAL_TRACE_PROVIDER = self._previous_trace_provider
        asyncio.run(self._close_clients())

    async def _close_clients(self) -> None:
        for client in self._clients:
            await client.close()

    # --- fixtures ---------------------------------------------------------

    def _call(
        self, name: str, arguments: str, call_id: str = "call_1"
    ) -> dict[str, object]:
        """One Chat Completions tool call, in the provider's wire shape."""
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def _text(self, content: str) -> dict[str, object]:
        return {"role": "assistant", "content": content}

    def _handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        # Distinct token counts and request IDs per call, so the aggregated
        # usage can be checked against the two individual calls rather than
        # against a total that one call could also produce.
        index = len(self.requests)
        return httpx2.Response(
            200,
            headers={"x-request-id": _REQUEST_IDS.get(index, f"req-{index}")},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": self.script.pop(0),
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10 + index,
                    "completion_tokens": 6 + index,
                    "total_tokens": 16 + 2 * index,
                    "prompt_tokens_details": {"cached_tokens": index},
                    "completion_tokens_details": {"reasoning_tokens": index + 1},
                },
            },
        )

    def _model(
        self, mode: str = "deepseek_json", max_tokens: int | None = 2048
    ):
        client = AsyncOpenAI(
            api_key="test-key",
            base_url=_BASE_URL,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(self._handle_request)
            ),
        )
        self._clients.append(client)
        return structured_output_fallback.FallbackChatCompletionsModel(
            model="deepseek-chat",
            openai_client=client,
            mode=mode,
            fallback_max_tokens=max_tokens,
        )

    def _schema(self) -> AgentOutputSchema:
        return AgentOutputSchema(ResearchPlanModel, strict_json_schema=True)

    def _tool(self) -> FunctionTool:
        @function_tool
        def lookup_library(name: str) -> str:
            """Look a library up in the catalogue."""
            return "found"

        return lookup_library

    def _mcp_tool(self) -> FunctionTool:
        """A tool as an MCP server's tool reaches the model.

        ``MCPUtil.to_function_tool`` converts an MCP tool into an ordinary
        ``FunctionTool``, so on the wire it is indistinguishable from a local
        one; what makes it an MCP call is the runner routing the resulting call
        back to its server. Preserving the call is therefore preserving MCP.
        """

        async def invoke(context: Any, arguments: str) -> str:
            return "document: structured output"

        return FunctionTool(
            name="search_docs",
            description="Search the documentation server for a query.",
            params_json_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            on_invoke_tool=invoke,
            strict_json_schema=False,
        )

    def _handoff(self):
        return handoff(Agent(name="specialist"))

    def _get(
        self,
        *,
        mode: str = "deepseek_json",
        tools: tuple = (),
        handoffs: tuple = (),
        input: str | list = "Plan some research.",
        model_settings: ModelSettings | None = None,
        max_tokens: int | None = 2048,
    ):
        return asyncio.run(
            self._model(mode, max_tokens=max_tokens).get_response(
                "Plan the research.",
                input,
                ModelSettings() if model_settings is None else model_settings,
                list(tools),
                self._schema(),
                list(handoffs),
                ModelTracing.DISABLED,
            )
        )

    def _calls(self, response) -> list[ResponseFunctionToolCall]:
        return [
            item
            for item in response.output
            if isinstance(item, ResponseFunctionToolCall)
        ]

    def _tool_names(self, body: dict[str, object]) -> list[str]:
        return [tool["function"]["name"] for tool in body.get("tools", [])]

    # --- phase one: the work phase ---------------------------------------

    def test_work_phase_returns_the_application_tool_call_unchanged(self):
        self.script = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._call(
                        "lookup_library", '{"name": "httpx2"}', "call_library"
                    )
                ],
            }
        ]

        response = self._get(tools=[self._tool()])

        self.assertEqual(len(self.requests), 1, "no formatting call yet")
        body = self.requests[0]
        self.assertNotIn("response_format", body)
        self.assertEqual(self._tool_names(body), ["lookup_library"])
        self.assertNotIn(
            structured_output_fallback._FORMAT_TOOL_NAME, json.dumps(body)
        )

        calls = self._calls(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "lookup_library")
        self.assertEqual(calls[0].arguments, '{"name": "httpx2"}')
        self.assertEqual(calls[0].call_id, "call_library")

    def test_mcp_tool_call_is_returned_unchanged(self):
        self.script = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._call(
                        "search_docs",
                        '{"query": "structured output"}',
                        "call_mcp",
                    )
                ],
            }
        ]

        response = self._get(tools=[self._mcp_tool()])

        self.assertEqual(len(self.requests), 1, "no formatting call yet")
        self.assertEqual(self._tool_names(self.requests[0]), ["search_docs"])

        calls = self._calls(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search_docs")
        self.assertEqual(calls[0].arguments, '{"query": "structured output"}')
        # The correlation the runner needs to route the call back to its
        # server is the call id, so it has to survive untouched.
        self.assertEqual(calls[0].call_id, "call_mcp")

    def test_handoff_call_is_returned_unchanged(self):
        self.script = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._call("transfer_to_specialist", "{}")],
            }
        ]

        response = self._get(handoffs=[self._handoff()])

        self.assertEqual(len(self.requests), 1, "no formatting call yet")
        self.assertEqual(
            self._tool_names(self.requests[0]), ["transfer_to_specialist"]
        )

        calls = self._calls(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "transfer_to_specialist")

    def test_refusal_in_the_work_phase_is_returned_unchanged(self):
        self.script = [
            {
                "role": "assistant",
                "content": None,
                "refusal": "I cannot help with that.",
            }
        ]

        response = self._get(tools=[self._tool()])

        self.assertEqual(len(self.requests), 1, "a refusal is not formatted")
        refusals = [
            part
            for item in response.output
            if isinstance(item, ResponseOutputMessage)
            for part in item.content
            if isinstance(part, ResponseOutputRefusal)
        ]
        self.assertEqual(
            [part.refusal for part in refusals], ["I cannot help with that."]
        )

    def test_work_phase_without_text_calls_or_refusal_fails_visibly(self):
        self.script = [self._text("")]

        with self.assertRaisesRegex(ModelBehaviorError, "nothing to format"):
            self._get(tools=[self._tool()])

        self.assertEqual(len(self.requests), 1, "no formatting call was made")

    # --- phase two: the formatting phase ----------------------------------

    def test_formatting_runs_only_after_a_terminal_text_response(self):
        self.script = [
            self._text("Found httpx2 in the catalogue."),
            self._text(self.arguments),
        ]

        response = self._get(tools=[self._tool()])

        self.assertEqual(len(self.requests), 2)
        work, formatter = self.requests
        self.assertNotIn("response_format", work)
        self.assertTrue(work["tools"])
        self.assertEqual(formatter["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", formatter)

        # The formatter sees the question and the answer it has to convert,
        # and nothing else: one user turn of history, then the work answer and
        # the formatting instruction together in one final user turn. No
        # assistant turn of our own, because a forced function call is ignored
        # after a replayed assistant turn on the real MiniMax endpoint.
        self.assertEqual(
            [message["role"] for message in formatter["messages"]],
            ["system", "user", "user"],
        )
        self.assertEqual(
            formatter["messages"][1]["content"], "Plan some research."
        )
        self.assertEqual(
            formatter["messages"][2]["content"],
            "Found httpx2 in the catalogue.\n\n"
            + structured_output_fallback._FORMAT_REQUEST_TEXT,
        )

        self.assertEqual(len(response.output), 1)
        message = response.output[0]
        self.assertIsInstance(message, ResponseOutputMessage)
        self.assertEqual(
            json.loads(message.content[0].text), json.loads(self.arguments)
        )

    def test_minimax_mode_formats_with_only_the_synthetic_function(self):
        self.script = [
            self._text("Found httpx2 in the catalogue."),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._call("emit_typed_output", self.arguments)],
            },
        ]

        response = self._get(mode="minimax_function", tools=[self._tool()])

        work, formatter = self.requests
        self.assertNotIn("response_format", work)
        self.assertEqual(self._tool_names(work), ["lookup_library"])
        self.assertEqual(
            formatter["tool_choice"],
            {
                "type": "function",
                "function": {"name": "emit_typed_output"},
            },
        )
        self.assertEqual(self._tool_names(formatter), ["emit_typed_output"])
        # The role shape the live endpoint needs: no assistant turn of our own
        # before the forced call, and the instruction still last.
        self.assertEqual(
            [message["role"] for message in formatter["messages"]],
            ["system", "user", "user"],
        )
        self.assertIn(
            "Found httpx2 in the catalogue.", formatter["messages"][2]["content"]
        )
        self.assertTrue(
            formatter["messages"][2]["content"].endswith(
                structured_output_fallback._FORMAT_REQUEST_TEXT
            )
        )
        self.assertEqual(
            json.loads(response.output[0].content[0].text),
            json.loads(self.arguments),
        )

    def test_formatter_wire_roles_are_system_then_user_turns(self):
        """The role shape the live MiniMax endpoint needs, per mode.

        MiniMax-M3 ignores a forced ``emit_typed_output`` call after replayed
        assistant turns, so a formatter request is system instructions, the
        original user turns, and one final user turn carrying the work answer
        and the output instruction. No assistant turn and no tool record.
        """
        for mode in ("deepseek_json", "minimax_function"):
            with self.subTest(mode=mode):
                self.requests = []
                self.script = [
                    self._text("Found httpx2."),
                    self._text(self.arguments)
                    if mode == "deepseek_json"
                    else {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            self._call("emit_typed_output", self.arguments)
                        ],
                    },
                ]

                self._get(mode=mode, tools=[self._tool()], input=self._mixed_history())

                work, formatter = self.requests
                self.assertEqual(self._tool_names(work), ["lookup_library"])
                self.assertEqual(
                    [message["role"] for message in formatter["messages"]],
                    ["system", "user", "user", "user"],
                )
                # The original user content and the terminal work answer are
                # both there, in order, and the instruction closes the request.
                self.assertEqual(
                    formatter["messages"][1]["content"], "Plan some research."
                )
                self.assertEqual(
                    formatter["messages"][2]["content"],
                    "Structured output, please.",
                )
                self.assertEqual(
                    formatter["messages"][3]["content"],
                    "Found httpx2.\n\n"
                    + structured_output_fallback._FORMAT_REQUEST_TEXT,
                )
                self.assertNotIn(
                    "tool", [message["role"] for message in formatter["messages"]]
                )
                self.assertNotIn(
                    "assistant",
                    [message["role"] for message in formatter["messages"]],
                )
                self.assertNotIn("call_missing", json.dumps(formatter))
                if mode == "deepseek_json":
                    self.assertNotIn("tools", formatter)
                else:
                    # The only tool a MiniMax formatter declares is its own.
                    self.assertEqual(
                        self._tool_names(formatter), ["emit_typed_output"]
                    )

    def test_handoffs_are_never_sent_to_the_formatter(self):
        self.script = [
            self._text("Handing over to a specialist."),
            self._text(self.arguments),
        ]

        self._get(handoffs=[self._handoff()])

        work, formatter = self.requests
        self.assertEqual(self._tool_names(work), ["transfer_to_specialist"])
        self.assertNotIn("tools", formatter)
        self.assertNotIn("transfer_to_specialist", json.dumps(formatter))

    def test_string_input_is_preserved_as_a_user_message(self):
        self.script = [self._text("Found httpx2."), self._text(self.arguments)]

        self._get(tools=[self._tool()], input="Plan some research.")

        formatter = self.requests[1]
        self.assertEqual(formatter["messages"][1]["role"], "user")
        self.assertEqual(
            formatter["messages"][1]["content"], "Plan some research."
        )

    def test_formatter_history_drops_replayed_tool_records(self):
        """The formatter declares no tools, so no tool record may travel."""
        history = self._mixed_history()
        self.script = [self._text("Found httpx2."), self._text(self.arguments)]

        self._get(tools=[self._tool()], input=history)

        formatter = self.requests[1]
        # Only the original user turns survive, and the work answer is merged
        # into the final user turn instead of standing as an assistant turn.
        self.assertEqual(
            [message["role"] for message in formatter["messages"]],
            ["system", "user", "user", "user"],
        )
        self.assertEqual(
            [message["content"] for message in formatter["messages"][1:3]],
            ["Plan some research.", "Structured output, please."],
        )
        self.assertEqual(
            formatter["messages"][3]["content"],
            "Found httpx2.\n\n" + structured_output_fallback._FORMAT_REQUEST_TEXT,
        )

        rendered = json.dumps(formatter)
        self.assertNotIn("tool", [message["role"] for message in formatter["messages"]])
        self.assertNotIn(
            "assistant", [message["role"] for message in formatter["messages"]]
        )
        # Every replay-only record is gone, including the answer to a call the
        # history never opened: a dangling id is exactly what a tool-free
        # formatter request cannot carry.
        for replayed in (
            "lookup_library",
            "CATALOGUE_RESULT_42",
            "ORPHANED_RESULT_7",
            "call_1",
            "call_missing",
            "Let me check the catalogue.",
            "Found it, now planning.",
        ):
            with self.subTest(replayed=replayed):
                self.assertNotIn(replayed, rendered)

    def _mixed_history(self) -> list[dict[str, object]]:
        """A work history with assistant turns and tool records to replay.

        ``call_missing`` is deliberately dangling: it names a call this
        request does not declare and the history never opened, which is the
        shape the real MiniMax endpoint rejected.
        """
        return [
            {"role": "user", "content": "Plan some research."},
            {"role": "assistant", "content": "Let me check the catalogue."},
            # What an executed turn leaves in the runner's input: the
            # invocation, then its result.
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_library",
                "arguments": '{"name": "httpx2"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "CATALOGUE_RESULT_42",
            },
            {
                "type": "function_call_output",
                "call_id": "call_missing",
                "output": "ORPHANED_RESULT_7",
            },
            {"role": "assistant", "content": "Found it, now planning."},
            {"role": "user", "content": "Structured output, please."},
        ]

    def test_format_request_filters_records_the_transport_cannot_carry(self):
        """A direct check of the filter, for shapes the wire cannot show.

        A raw ``role="tool"`` message and an ``mcp_call`` are both rejected by
        the SDK's converter, so they can never survive a work phase to be
        observed in a request body. They are still history an input list may
        carry, so the filter is asserted on directly.
        """
        history = self._mixed_history() + [
            {
                "role": "tool",
                "tool_call_id": "call_missing",
                "content": "CATALOGUE_RESULT_42",
            },
            {
                "type": "mcp_call",
                "id": "mcp_1",
                "name": "search_docs",
                "arguments": '{"query": "structured output"}',
                "server_label": "docs",
            },
            {
                "type": "mcp_approval_request",
                "id": "mcp_2",
                "name": "search_docs",
                "arguments": "{}",
                "server_label": "docs",
            },
        ]

        messages = (
            structured_output_fallback.append_assistant_text_and_format_request(
                history, "Found httpx2.", self._schema()
            )
        )

        # Only the user turns stay; every assistant turn, function-call record
        # and tool-role or MCP record is dropped, and the work answer travels
        # inside the final user turn rather than as an assistant turn.
        self.assertEqual(
            [(message["role"], message["content"]) for message in messages],
            [
                ("user", "Plan some research."),
                ("user", "Structured output, please."),
                (
                    "user",
                    "Found httpx2.\n\n"
                    + structured_output_fallback._FORMAT_REQUEST_TEXT,
                ),
            ],
        )

    def test_format_request_does_not_mutate_the_caller_history(self):
        """The input list is the runner's; the formatter may not rewrite it."""
        history = self._mixed_history()
        before = [dict(item) for item in history]

        structured_output_fallback.append_assistant_text_and_format_request(
            history, "Found httpx2.", self._schema()
        )

        self.assertEqual(history, before)

    def test_list_input_history_is_preserved(self):
        history = [
            {"role": "user", "content": "Plan some research."},
            {"role": "assistant", "content": "Which area?"},
            {"role": "user", "content": "Structured output."},
        ]
        self.script = [self._text("Found httpx2."), self._text(self.arguments)]

        self._get(tools=[self._tool()], input=history)

        formatter = self.requests[1]
        # The original user turns survive in order; the assistant turn between
        # them does not, because two of our own turns in a row is the shape the
        # live endpoint refused to answer with a forced call.
        self.assertEqual(
            [message["content"] for message in formatter["messages"][1:3]],
            ["Plan some research.", "Structured output."],
        )
        self.assertEqual(
            [message["role"] for message in formatter["messages"]],
            ["system", "user", "user", "user"],
        )
        self.assertEqual(history[1]["content"], "Which area?", "input untouched")

    def test_formatter_output_that_is_not_the_schema_fails_visibly(self):
        self.script = [
            self._text("Found httpx2."),
            self._text("Here is the plan instead."),
        ]

        with self.assertRaises(ModelBehaviorError):
            self._get(tools=[self._tool()])

        self.assertEqual(len(self.requests), 2, "both phases were attempted")

    # --- configuration -----------------------------------------------------

    def test_missing_deepseek_budget_fails_before_the_work_request(self):
        """The budget is a configuration error, so nothing may be paid for.

        Phase two cannot run without it, and discovering that after phase one
        would charge for a work request whose answer can never be formatted.
        """
        self.script = [self._text("Found httpx2."), self._text(self.arguments)]

        with self.assertRaisesRegex(ValueError, "token budget"):
            self._get(
                mode="deepseek_json",
                tools=[self._tool()],
                max_tokens=None,
            )

        self.assertEqual(self.requests, [], "no request may be paid for")

    def test_minimax_two_phase_needs_no_token_budget(self):
        """Only DeepSeek JSON mode needs a budget: MiniMax formats anyway."""
        self.script = [
            self._text("Found httpx2."),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [self._call("emit_typed_output", self.arguments)],
            },
        ]

        response = self._get(
            mode="minimax_function", tools=[self._tool()], max_tokens=None
        )

        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            json.loads(response.output[0].content[0].text),
            json.loads(self.arguments),
        )

    # --- accounting --------------------------------------------------------

    def test_usage_is_aggregated_across_both_phase_requests(self):
        self.script = [self._text("Found httpx2."), self._text(self.arguments)]

        response = self._get(
            tools=[self._tool()],
            model_settings=ModelSettings(preserve_raw_usage=True),
        )

        usage = response.usage
        self.assertEqual(usage.requests, 2)
        self.assertEqual(usage.input_tokens, 11 + 12)
        self.assertEqual(usage.output_tokens, 7 + 8)
        self.assertEqual(usage.total_tokens, 18 + 20)
        self.assertEqual(usage.input_tokens_details.cached_tokens, 1 + 2)
        self.assertEqual(usage.output_tokens_details.reasoning_tokens, 2 + 3)
        self.assertEqual(len(usage.request_usage_entries), 2)
        # One response cannot carry the request id or raw usage payload of two
        # physical provider calls, so neither is reported as if it were both.
        self.assertIsNone(response.request_id)
        self.assertIsNone(response.raw_usage)

    def test_two_phase_run_logs_only_safe_metadata(self):
        self.script = [self._text("Found httpx2."), self._text(self.arguments)]

        with self.assertLogs("openai.agents", level="DEBUG") as logs:
            self._get(tools=[self._tool()])

        records = [
            record
            for record in logs.records
            if getattr(record, "phase_count", None) == 2
        ]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.fallback_mode, "deepseek_json")
        self.assertEqual(record.work_request_id, "req-work")
        self.assertEqual(record.format_request_id, "req-format")

        # The record is the adapter's own, so only its message is under test:
        # the SDK's surrounding debug logs do contain model data.
        rendered = record.getMessage()
        for secret in (
            "Plan some research.",
            self.arguments,
            "Found httpx2.",
            "test-key",
            "Bearer",
        ):
            with self.subTest(secret=secret):
                self.assertNotIn(secret, rendered)

    # --- the whole run -----------------------------------------------------

    def test_runner_executes_the_tool_then_formats_the_final_answer(self):
        """The work the runner asked for happens, and only then is it formatted."""
        self.script = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    self._call("lookup_library", '{"name": "httpx2"}', "call_1")
                ],
            },
            self._text("Found httpx2 in the catalogue."),
            self._text(self.arguments),
        ]
        agent = Agent(
            name="planner",
            instructions="Plan the research.",
            model=self._model(),
            output_type=ResearchPlanModel,
            tools=[self._tool()],
        )

        result = asyncio.run(Runner.run(agent, "Plan some research."))

        self.assertEqual(result.final_output.topic, "structured output")
        self.assertEqual(result.final_output.lead_step.estimated_minutes, 30)
        # One work call that produced the tool call, one that produced the
        # terminal answer, and one formatting call.
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(self.requests[0]["tools"]), 1)
        self.assertNotIn("tools", self.requests[2])

        # The formatter carries the question and the answer it has to format,
        # and none of the tool records behind them: it declares no tools, so a
        # tool call id it cannot resolve would be rejected upstream. The answer
        # and the instruction share one final user turn, so the request has no
        # assistant turn of its own - the shape the live MiniMax endpoint needs
        # before it honours the forced call.
        formatter = self.requests[2]
        self.assertEqual(
            [message["role"] for message in formatter["messages"]],
            ["system", "user", "user"],
        )
        self.assertEqual(formatter["messages"][1]["content"], "Plan some research.")
        self.assertEqual(
            formatter["messages"][2]["content"],
            "Found httpx2 in the catalogue.\n\n"
            + structured_output_fallback._FORMAT_REQUEST_TEXT,
        )
        self.assertNotIn(
            "tool", [message["role"] for message in formatter["messages"]]
        )
        rendered = json.dumps(formatter)
        for replayed in ("lookup_library", "call_1", "tool_call_id"):
            with self.subTest(replayed=replayed):
                self.assertNotIn(replayed, rendered)


class StreamingFallbackTests(unittest.TestCase):
    """Typed-output streaming is refused before a request is built."""

    def setUp(self) -> None:
        self.requests: list[dict[str, object]] = []
        self._clients: list[AsyncOpenAI] = []

        self._previous_trace_provider = tracing_setup.GLOBAL_TRACE_PROVIDER
        tracing_setup.GLOBAL_TRACE_PROVIDER = None
        set_tracing_disabled(True)

    def tearDown(self) -> None:
        tracing_setup.GLOBAL_TRACE_PROVIDER = self._previous_trace_provider
        asyncio.run(self._close_clients())

    async def _close_clients(self) -> None:
        for client in self._clients:
            await client.close()

    def _handle_request(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        chunks = (
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Here."},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "deepseek-chat",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        )
        body = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )
        return httpx2.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    def _model(self, mode: str = "deepseek_json"):
        client = AsyncOpenAI(
            api_key="test-key",
            base_url=_BASE_URL,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(self._handle_request)
            ),
        )
        self._clients.append(client)
        return structured_output_fallback.FallbackChatCompletionsModel(
            model="deepseek-chat",
            openai_client=client,
            mode=mode,
            fallback_max_tokens=2048,
        )

    def test_typed_stream_is_rejected_before_any_request(self):
        for mode in ("deepseek_json", "minimax_function"):
            with self.subTest(mode=mode):
                model = self._model(mode)

                with self.assertRaisesRegex(ModelBehaviorError, "stream"):
                    model.stream_response(
                        "Plan the research.",
                        "Plan some research.",
                        ModelSettings(),
                        [],
                        AgentOutputSchema(ResearchPlanModel, strict_json_schema=True),
                        [],
                        ModelTracing.DISABLED,
                    )

                self.assertEqual(self.requests, [], "no request may be sent")

    def test_plain_text_stream_delegates_unchanged(self):
        model = self._model()

        events = asyncio.run(
            self._consume(
                model.stream_response(
                    "Plan the research.",
                    "Plan some research.",
                    ModelSettings(),
                    [],
                    None,
                    [],
                    ModelTracing.DISABLED,
                )
            )
        )

        self.assertEqual(len(self.requests), 1)
        self.assertTrue(events)

    async def _consume(self, stream) -> list[object]:
        return [event async for event in stream]


_ENV_EXAMPLE = pathlib.Path(__file__).resolve().parent / ".env.example"
_README = pathlib.Path(__file__).resolve().parent / "README.md"

# The exact wording the documentation is required to carry, kept here rather
# than inline in each test so a reworded sentence fails in one place.
_DOC_PHRASES = {
    "endpoint_wide": "endpoint-wide",
    "not_inferred": "never inferred from the model id",
    "local_validation": "validated locally",
    "two_phase_cost": r"[Tt]wo phases.{0,120}extra request",
    "no_streaming": "[Ss]treaming typed output is not supported",
    "syntax_only": "syntactically valid JSON",
    "not_schema_enforcement": "not schema compliance",
    "do_not_rely": r"[Dd]o\s+not\s+rely",
    # Acceptance is endpoint-specific and unequal across the two modes, so a
    # reader must be told where each mode actually stopped being verified.
    "endpoint_specific": "endpoint-specific",
}

# The one number both docs have to carry: the minimax_function two-phase path
# is not a pass, it is a pass rate. The mode name has to sit next to it, or a
# reader cannot tell which mode the caveat belongs to.
_MINIMAX_TOOL_PASS_RATE = r"minimax_function.{0,300}3 of 5"


def _documented_modes(text: str) -> set[str]:
    """Every mode value ``text`` offers for AGENT_STRUCTURED_OUTPUT_MODE."""
    return set(re.findall(r"AGENT_STRUCTURED_OUTPUT_MODE=(\w+)", text))


class DocumentationTests(unittest.TestCase):
    """The fallback is opt-in, so the two docs are the only way to find it.

    These read the real files instead of restating them: a mode the adapter
    supports but nobody documented is a mode nobody will turn on, and a mode
    the docs promise but the adapter rejects is worse than either.
    """

    def setUp(self) -> None:
        self.env_example = _ENV_EXAMPLE.read_text(encoding="utf-8")
        self.readme = _README.read_text(encoding="utf-8")

    def _assert_documents(self, text: str, pattern: str, what: str) -> None:
        """Assert ``pattern`` matches, without dumping the whole file.

        assertRegex prints the subject on failure, which for README.md is a
        few kilobytes of Markdown between the reader and the one line that
        matters.
        """
        self.assertTrue(
            re.search(pattern, text, re.DOTALL) is not None,
            f"not documented: {what}",
        )

    # --- .env.example ---------------------------------------------------

    def test_env_example_offers_every_supported_mode(self):
        self.assertEqual(
            _documented_modes(self.env_example),
            set(structured_output_fallback._FALLBACK_MODES),
        )

    def test_env_example_leaves_every_mode_commented_out(self):
        # Opt-in: a template that enabled a mode by default would silently
        # switch every typed example onto a non-native protocol.
        for mode in structured_output_fallback._FALLBACK_MODES:
            with self.subTest(mode=mode):
                self.assertRegex(
                    self.env_example,
                    rf"(?m)^#AGENT_STRUCTURED_OUTPUT_MODE={mode}$",
                )

    def test_env_example_says_the_mode_is_endpoint_wide(self):
        self.assertIn(_DOC_PHRASES["endpoint_wide"], self.env_example)
        self.assertIn(_DOC_PHRASES["not_inferred"], self.env_example)

    def test_env_example_documents_the_token_budget(self):
        self.assertIn("AGENT_STRUCTURED_OUTPUT_MAX_TOKENS", self.env_example)

    def test_env_example_documents_the_two_phase_cost(self):
        self._assert_documents(
            self.env_example,
            _DOC_PHRASES["two_phase_cost"],
            "the extra request tool-bearing agents cost",
        )

    def test_env_example_documents_that_validation_stays_local(self):
        self.assertIn(_DOC_PHRASES["local_validation"], self.env_example)

    def test_env_example_documents_the_stream_limitation(self):
        self._assert_documents(
            self.env_example,
            _DOC_PHRASES["no_streaming"],
            "the streamed typed-output limitation",
        )

    def test_env_example_documents_the_tool_bearing_limitation(self):
        self._assert_documents(
            self.env_example,
            _MINIMAX_TOOL_PASS_RATE,
            "the measured minimax_function tool-bearing pass rate",
        )
        self._assert_documents(
            self.env_example,
            _DOC_PHRASES["do_not_rely"],
            "that the minimax_function tool-bearing path is not to be relied on",
        )

    # --- README ---------------------------------------------------------

    def test_readme_offers_every_supported_mode(self):
        self.assertEqual(
            _documented_modes(self.readme),
            set(structured_output_fallback._FALLBACK_MODES),
        )

    def test_readme_shows_how_to_select_each_mode(self):
        for mode in structured_output_fallback._FALLBACK_MODES:
            with self.subTest(mode=mode):
                self._assert_documents(
                    self.readme,
                    rf"AGENT_STRUCTURED_OUTPUT_MODE={mode}\s+python",
                    f"a command that selects {mode}",
                )

    def test_readme_explains_deepseek_json_is_syntax_only(self):
        self.assertIn(_DOC_PHRASES["syntax_only"], self.readme)
        self.assertIn(_DOC_PHRASES["not_schema_enforcement"], self.readme)

    def test_readme_explains_minimax_uses_a_forced_synthetic_function(self):
        format_tool = structured_output_fallback._FORMAT_TOOL_NAME
        self.assertIn(format_tool, self.readme)
        self._assert_documents(
            self.readme,
            r"[Ff]orces?.{0,80}tool_choice",
            "that the synthetic function is forced with tool_choice",
        )

    def test_readme_documents_that_validation_stays_local(self):
        self.assertIn(_DOC_PHRASES["local_validation"], self.readme)

    def test_readme_documents_the_second_request_for_tool_bearing_agents(self):
        self.assertIn("second request", self.readme)

    def test_readme_documents_the_stream_limitation(self):
        self._assert_documents(
            self.readme,
            r"Runner\.run_streamed.{0,120}not supported",
            "that streamed typed output is unsupported",
        )

    def test_readme_documents_that_unset_keeps_the_native_json_schema(self):
        self._assert_documents(
            self.readme,
            r"[Uu]nset.{0,200}json_schema",
            "that leaving the mode unset keeps the native json_schema request",
        )

    def test_readme_documents_the_tool_bearing_limitation(self):
        # deepseek_json was accepted for both agent shapes; minimax_function
        # only for plain typed ones. Naming the mode and the measured rate is
        # what stops a reader from turning on the wrong one and calling the
        # result a bug.
        self.assertIn(_DOC_PHRASES["endpoint_specific"], self.readme)
        self._assert_documents(
            self.readme,
            _MINIMAX_TOOL_PASS_RATE,
            "the measured minimax_function tool-bearing pass rate",
        )
        self._assert_documents(
            self.readme,
            _DOC_PHRASES["do_not_rely"],
            "that the minimax_function tool-bearing path is not to be relied on",
        )


if __name__ == "__main__":
    unittest.main()
