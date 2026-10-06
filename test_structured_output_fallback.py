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
import sys
import unittest
from typing import Literal
from unittest.mock import patch

import httpx2
from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.agent_output import AgentOutputSchema
from agents.exceptions import ModelBehaviorError
from agents.items import ResponseOutputMessage, ResponseOutputRefusal
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
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
        message: dict[str, object] = {"role": "assistant", "content": self.reply}
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

    def test_minimax_mode_still_delegates_to_the_sdk(self):
        self._run(model=self._model(mode="minimax_function"))

        self.assertEqual(
            self._one_request()["response_format"]["type"], "json_schema"
        )

    def test_agents_with_tools_keep_the_native_request(self):
        @function_tool
        def lookup_library(name: str) -> str:
            """Look a library up in the catalogue."""
            return "found"

        self._run(tools=[lookup_library])

        body = self._one_request()
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["tools"])

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


if __name__ == "__main__":
    unittest.main()
