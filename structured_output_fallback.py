"""Provider-aware structured-output adapter for the Agents SDK.

Importing this module changes nothing. :func:`install`, called from
``agents_config._configure_agents_sdk()`` when the operator opts in, wraps the
public ``OpenAIProvider.get_model`` factory so that every Chat Completions
model the SDK builds is handed back as a :class:`FallbackChatCompletionsModel`.

The wrapper is a seam onto SDK behaviour the version in ``requirements.txt``
defines, not a documented extension point: ``OpenAIChatCompletionsModel``'s
constructor and ``OpenAIProvider.get_model`` are both read here, so an SDK
upgrade has to re-check both before widening that pin.

Two provider protocols are selected by :func:`install`, and both are
implemented here: ``deepseek_json`` asks for JSON mode in the request, and
``minimax_function`` routes the typed output through a synthetic function call
instead of a JSON Schema response format.
"""

import dataclasses
import json
import uuid
import weakref
from typing import Any

from agents.agent_output import AgentOutputSchemaBase
from agents.exceptions import ModelBehaviorError
from agents.handoffs import Handoff
from agents.items import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    TResponseInputItem,
)
from agents.model_settings import ModelSettings
from agents.models.interface import ModelResponse, ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider
from agents.tool import FunctionTool, Tool
from agents.tool_context import ToolContext
from openai.types.responses.response_prompt_param import ResponsePromptParam

# Captured at import time, before any wrapper replaces the factory.
_original_get_model = OpenAIProvider.get_model

# DeepSeek's JSON mode only guarantees JSON syntax, and it needs a budget: its
# documented failure mode is a truncated or empty completion.
_DEEPSEEK_JSON_MODE = "deepseek_json"

# MiniMax's guide documents Function Calling but not JSON Schema
# ``response_format``, so the typed output is collected as a function call.
_MINIMAX_FUNCTION_MODE = "minimax_function"

# Reserved for the adapter's own formatting call. An application tool must
# never be given this name: the two would be indistinguishable in a response.
_FORMAT_TOOL_NAME = "emit_typed_output"

# How many schema nodes one example traversal may expand. Generous next to the
# declared example schemas, and small enough that an example stays readable in
# a prompt. It bounds size, not recursion: cycles are detected separately.
_MAX_EXAMPLE_NODES = 256

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

    Registering it changes no request on its own. With no mode selected, or on
    a request the selected mode does not cover, ``get_response`` is still the
    SDK's own.
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

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        """Rewrite typed-output requests the selected mode covers.

        Both protocols apply only to a typed request that is not competing
        with application tools or handoffs; everything else is left to the SDK.
        The original ``output_schema`` is always passed through on the DeepSeek
        path, so the SDK still parses and validates the final content against
        it; MiniMax validates the arguments here and hands the SDK plain text.
        """
        if self._uses_deepseek_json(output_schema, tools, handoffs):
            # The budget is a configuration problem, so it is checked before a
            # schema the adapter cannot describe.
            settings = self._json_mode_settings(model_settings)
            instructions = _json_instructions(output_schema) + (
                system_instructions or ""
            )
            response = await self._sdk_get_response(
                instructions,
                input,
                settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            _reject_silent_empty_output(response)
            return response

        if self._uses_minimax_function(output_schema, tools, handoffs):
            # The schema travels as the function's parameter schema instead of
            # a response format, so the SDK is given no output schema at all:
            # this adapter validates the arguments against the real one.
            response = await self._sdk_get_response(
                system_instructions,
                input,
                self._function_call_settings(model_settings),
                [_format_tool(output_schema)],
                None,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )
            return _function_call_to_text(response, output_schema)

        return await self._sdk_get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )

    async def _sdk_get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        previous_response_id,
        conversation_id,
        prompt,
    ) -> ModelResponse:
        return await super().get_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

    def _uses_deepseek_json(self, output_schema, tools, handoffs) -> bool:
        return (
            self._fallback_mode == _DEEPSEEK_JSON_MODE
            and _is_typed_output(output_schema)
            and not tools
            and not handoffs
        )

    def _uses_minimax_function(self, output_schema, tools, handoffs) -> bool:
        return (
            self._fallback_mode == _MINIMAX_FUNCTION_MODE
            and _is_typed_output(output_schema)
            and not tools
            and not handoffs
        )

    def _json_mode_settings(self, model_settings: ModelSettings) -> ModelSettings:
        """Add JSON mode to the settings, keeping every other caller value.

        ``response_format`` has to travel in ``extra_body``: the SDK passes its
        own ``response_format`` from the output schema, and ``extra_body`` is
        merged over it by the OpenAI client. ``extra_args`` is not an
        alternative here - it collides with that keyword and raises instead of
        overriding it.
        """
        max_tokens = (
            model_settings.max_tokens
            if model_settings.max_tokens is not None
            else self._fallback_max_tokens
        )
        if max_tokens is None:
            raise ValueError(
                "DeepSeek JSON mode needs a token budget: set max_tokens on the "
                "agent, or AGENT_STRUCTURED_OUTPUT_MAX_TOKENS for the examples "
                "that set none. DeepSeek truncates JSON output without one."
            )
        return dataclasses.replace(
            model_settings,
            extra_body={
                **(model_settings.extra_body or {}),
                "response_format": {"type": "json_object"},
            },
            max_tokens=max_tokens,
        )

    def _function_call_settings(
        self, model_settings: ModelSettings
    ) -> ModelSettings:
        """Force the formatting function, keeping every other caller value.

        Withholding ``output_schema`` already removes the SDK's own
        ``response_format``, but a caller may have set one in ``extra_body``,
        and the OpenAI client merges that over the keyword. This protocol
        speaks in tool calls, so the entry is dropped rather than replaced.
        """
        extra_body = dict(model_settings.extra_body or {})
        extra_body.pop("response_format", None)
        return dataclasses.replace(
            model_settings,
            tool_choice=_FORMAT_TOOL_NAME,
            extra_body=extra_body or None,
        )


def _is_typed_output(output_schema: AgentOutputSchemaBase | None) -> bool:
    return output_schema is not None and not output_schema.is_plain_text()


def _reject_silent_empty_output(response: ModelResponse) -> None:
    """Fail a JSON-mode response that carries no assistant text or refusal.

    Passing such a response through is not the visible failure the design
    asks for: the runner only validates output text it found, so an empty
    completion is retried until the turn limit and surfaces as ten requests
    and a ``MaxTurnsExceeded``, with nothing pointing at the provider. DeepSeek
    documents JSON mode returning empty content, so the adapter reports it as
    the typed-output failure it is, after the request and its usage have been
    recorded. A refusal is a terminal answer too, so it is left to the SDK's
    own refusal handling.
    """
    # ModelResponse.output holds raw Responses items, not the run-item
    # wrappers the runner builds from them later.
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for part in item.content:
            if isinstance(part, ResponseOutputRefusal):
                return
            if isinstance(part, ResponseOutputText) and part.text:
                return

    raise ModelBehaviorError(
        "DeepSeek JSON mode returned a completed response with no assistant "
        "text or refusal, so there is no JSON output to validate. This is the "
        "documented DeepSeek empty-content failure: raise max_tokens, or check "
        "that the endpoint really supports JSON mode."
    )


def _format_tool(output_schema: AgentOutputSchemaBase) -> FunctionTool:
    """Build the formatting-only function that carries the output schema.

    The parameter schema is the declared output schema itself, so the shape is
    described through the one mechanism the provider's guide documents.
    Strictness is off: it is an OpenAI Responses guarantee, not one this
    endpoint's Function Calling is documented to honour.

    The invoker only raises. The adapter turns the call into text and never
    leaves it in the response, so a call reaching the tool runtime means the
    adapter lost its own call - which would turn a typed output into a tool
    result.
    """
    return FunctionTool(
        name=_FORMAT_TOOL_NAME,
        description="Emit the final answer as JSON matching the schema.",
        params_json_schema=output_schema.json_schema(),
        on_invoke_tool=_reject_format_tool_invocation,
        strict_json_schema=False,
    )


async def _reject_format_tool_invocation(
    context: ToolContext[Any], arguments: str
) -> Any:
    """Fail if the formatting call ever escapes the adapter."""
    raise ModelBehaviorError(
        f"{_FORMAT_TOOL_NAME!r} is an adapter formatting call: it is converted "
        "into the typed output and must never be executed. Reaching the tool "
        "runtime means the MiniMax function-call fallback failed to intercept "
        "its own call."
    )


def _function_call_to_text(
    response: ModelResponse, output_schema: AgentOutputSchemaBase
) -> ModelResponse:
    """Replace the formatting call with the assistant text it carries.

    The call is an adapter detail: the runner knows nothing about the tool and
    would either fail on an unknown name or execute it. Exactly one correctly
    named call is accepted, its arguments are validated against the schema the
    caller declared, and the usage and request ID of that single request are
    kept.
    """
    calls = [
        item
        for item in response.output
        if isinstance(item, ResponseFunctionToolCall)
    ]
    if len(calls) != 1 or calls[0].name != _FORMAT_TOOL_NAME:
        raise ModelBehaviorError(_missing_format_call_message(response, calls))

    arguments = calls[0].arguments
    # Raises ModelBehaviorError on any argument JSON the schema does not accept.
    output_schema.validate_json(arguments)

    message = ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex}",
        content=[
            ResponseOutputText(
                text=arguments,
                type="output_text",
                annotations=[],
                logprobs=[],
            )
        ],
        role="assistant",
        type="message",
        status="completed",
    )
    return ModelResponse(
        output=[message],
        usage=response.usage,
        response_id=None,
        request_id=response.request_id,
        raw_usage=response.raw_usage,
    )


def _missing_format_call_message(
    response: ModelResponse, calls: list[ResponseFunctionToolCall]
) -> str:
    """Say why a function-call response carries no typed output.

    A refusal and a missing call are both terminal answers with no arguments,
    but only one of them tells the operator what the provider decided, so they
    are reported differently.
    """
    refusal = next(
        (
            part.refusal
            for item in response.output
            if isinstance(item, ResponseOutputMessage)
            for part in item.content
            if isinstance(part, ResponseOutputRefusal)
        ),
        None,
    )
    if refusal is not None:
        reason = f"the provider returned a refusal: {refusal}"
    elif calls:
        reason = f"it returned calls to {[call.name for call in calls]}"
    else:
        reason = "it returned no function call"

    return (
        f"MiniMax function calling expected exactly one {_FORMAT_TOOL_NAME!r} "
        f"call, but {reason}. A function-call response that does not carry the "
        "typed output is a typed-output failure, not an empty result: check "
        "that the endpoint really supports function calling."
    )


def _json_instructions(output_schema: AgentOutputSchemaBase) -> str:
    """Describe the expected JSON in a system instruction.

    DeepSeek's JSON mode guarantees JSON syntax and nothing else, and it
    rejects a request whose prompt never mentions JSON, so the schema and a
    filled-in example both have to be in the prompt. The example is labelled
    illustrative: it shows the shape, not the answer.
    """
    schema = output_schema.json_schema()
    return (
        "Respond with JSON only: no prose, no Markdown fences, no extra keys.\n"
        f"JSON schema:\n{json.dumps(schema)}\n"
        f"Example (illustrative):\n{json.dumps(_json_schema_example(schema))}\n"
    )


class _NodeBudget:
    """Caps how many schema nodes one example traversal may expand.

    A schema can be acyclic and still explode: twelve ``$defs`` that each
    reference the next one twice describe a four-thousand-node example. Nothing
    about that is recursive, so it gets its own error rather than the
    recursion one.
    """

    def __init__(self, limit: int = _MAX_EXAMPLE_NODES) -> None:
        self.remaining = limit

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError(
                "Cannot build a JSON example: the output schema needs more "
                f"than {_MAX_EXAMPLE_NODES} nodes to illustrate, so it is too "
                "deep or too wide for a prompt example."
            )


def _json_schema_example(schema: dict[str, object]) -> object:
    """Build a representative JSON value for ``schema``.

    The result is any JSON value, not only an object: a typed output may be a
    list or a scalar. Anything the traversal cannot represent - a remote
    ``$ref``, a schema with no usable type, a cycle, a schema too large to
    illustrate - raises instead of producing an example that would mislead the
    model.
    """
    return _example_for(schema, schema)


def _example_for(
    schema: dict[str, object],
    root: dict[str, object],
    active: tuple[str, ...] = (),
    budget: _NodeBudget | None = None,
):
    """Build an example for ``schema``.

    ``active`` holds the ``$ref`` values whose expansion is in progress right
    now, so a reference that leads back into itself is reported as recursion.
    Depth is not a proxy for that: a chain of distinct references is deep but
    finite, and calling it recursive would refuse schemas that are perfectly
    describable. ``budget`` bounds total size instead, which depth cannot: a
    schema that references the next definition twice is acyclic and still
    doubles at every level.
    """
    budget = _NodeBudget() if budget is None else budget
    budget.spend()

    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in active:
            raise ValueError(
                f"Cannot build a JSON example: output schema reference {ref!r} "
                "is recursive, so it has no finite example."
            )
        return _example_for(
            _resolve_ref(ref, root), root, active + (ref,), budget
        )
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]

    # Optional values arrive as a union with null; show the real type.
    for union_key in ("anyOf", "oneOf"):
        if union_key in schema:
            branches = [
                branch
                for branch in schema[union_key]
                if branch.get("type") != "null"
            ]
            if not branches:
                return None
            return _example_for(branches[0], root, active, budget)

    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        # Only required properties: an example has to satisfy the schema, and
        # omitting an optional property always does.
        required = set(schema.get("required", []))
        return {
            name: _example_for(value, root, active, budget)
            for name, value in schema.get("properties", {}).items()
            if name in required
        }
    if kind == "array":
        return [_example_for(schema["items"], root, active, budget)]
    if kind == "string":
        return ""
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    raise ValueError(f"Cannot build a JSON example for schema type {kind!r}")


def _resolve_ref(ref: str, root: dict[str, object]) -> dict[str, object]:
    """Resolve a ``#/$defs/...`` reference against the schema it came from."""
    if not ref.startswith("#/"):
        raise ValueError(f"Cannot build a JSON example for remote $ref {ref!r}")

    target: object = root
    for part in ref.removeprefix("#/").split("/"):
        try:
            target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            raise ValueError(
                f"Cannot resolve $ref {ref!r} in the output schema"
            ) from None
    return target


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
