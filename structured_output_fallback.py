"""Provider-aware structured-output adapter for the Agents SDK.

Importing this module changes nothing. :func:`install`, called from
``agents_config._configure_agents_sdk()`` when the operator opts in, wraps the
public ``OpenAIProvider.get_model`` factory so that every Chat Completions
model the SDK builds is handed back as a :class:`FallbackChatCompletionsModel`.

The wrapper is a seam onto SDK behaviour the version in ``requirements.txt``
defines, not a documented extension point: ``OpenAIChatCompletionsModel``'s
constructor and ``OpenAIProvider.get_model`` are both read here, so an SDK
upgrade has to re-check both before widening that pin.
"""

import weakref

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider

# Captured at import time, before any wrapper replaces the factory.
_original_get_model = OpenAIProvider.get_model

_mode: str | None = None
_fallback_max_tokens: int | None = None
_installed = False

# Keyed weakly on the provider so a discarded provider does not keep its
# adapted models alive.
_model_caches: "weakref.WeakKeyDictionary[OpenAIProvider, dict]" = (
    weakref.WeakKeyDictionary()
)


class FallbackChatCompletionsModel(OpenAIChatCompletionsModel):
    """A Chat Completions model that can rewrite typed-output requests.

    It stays a subclass of ``OpenAIChatCompletionsModel`` on purpose: SDK code
    guards behaviour on ``isinstance(..., OpenAIChatCompletionsModel)``, and a
    plain delegating ``Model`` would quietly lose those guardrails.

    Registering it changes no request yet: ``get_response`` is still the SDK's
    own, so an installed adapter with no mode behaves exactly like the model it
    replaced.
    """

    def __init__(
        self,
        *,
        model: str,
        openai_client,
        mode: str,
        fallback_max_tokens: int | None = None,
        strict_feature_validation: bool = False,
        buffer_streamed_tool_calls: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            openai_client=openai_client,
            strict_feature_validation=strict_feature_validation,
            buffer_streamed_tool_calls=buffer_streamed_tool_calls,
        )
        self._fallback_mode = mode
        self._fallback_max_tokens = fallback_max_tokens


def install(mode: str, max_tokens: int | None) -> None:
    """Install the provider factory adapter once.

    Calling it again only updates the selected mode: the factory is wrapped a
    single time, so importing ``agents_config`` twice cannot stack wrappers.
    Changing the settings drops the cached adapters, because a model built
    under the previous settings is still carrying them.
    """
    global _mode, _fallback_max_tokens, _installed

    if _mode != mode or _fallback_max_tokens != max_tokens:
        _model_caches.clear()

    _mode = mode
    _fallback_max_tokens = max_tokens

    if _installed:
        return

    OpenAIProvider.get_model = _get_model_with_structured_fallback
    _installed = True


def _cache_for_provider(provider: OpenAIProvider) -> dict:
    cache = _model_caches.get(provider)
    if cache is None:
        cache = {}
        _model_caches[provider] = cache
    return cache


def _get_model_with_structured_fallback(provider, model_name):
    model = _original_get_model(provider, model_name)
    if _mode is None or not isinstance(model, OpenAIChatCompletionsModel):
        return model

    cache = _cache_for_provider(provider)
    key = model.model
    if key not in cache:
        cache[key] = FallbackChatCompletionsModel(
            model=model.model,
            openai_client=model._client,
            strict_feature_validation=model._strict_feature_validation,
            buffer_streamed_tool_calls=model._buffer_streamed_tool_calls,
            mode=_mode,
            fallback_max_tokens=_fallback_max_tokens,
        )
    return cache[key]
