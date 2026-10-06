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
"""

import importlib
import os
import sys
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
