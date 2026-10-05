# Provider-Aware Structured-Output Fallback

**Status:** Design for review

**Date:** 2026-10-05

## Problem

The Agents SDK sends `response_format: {"type":"json_schema", ...}` for every non-plain `Agent(output_type=...)` on the current Chat Completions route. This works for some TokenHub models but not all:

- Seven tested DeepSeek IDs were rejected with HTTP 400 / `400006` because that route/model does not accept the requested JSON Schema response format.
- Three GLM and four MiniMax IDs accepted the request but returned ordinary prose/Markdown. The SDK then raised `ModelBehaviorError: Invalid JSON when parsing model output`.
- A simpler `tasks: List[str]` type did not fix those failures. One additional model, `glm-5-1`, passed the simpler example after timing out on the nested schema.

The failures are therefore provider-protocol and model-output differences, not only schema complexity. The existing 35 example files using typed output should keep their current default behavior and should not each grow provider-specific request code.

## Goals

- Preserve the Agents SDK's native JSON Schema output as the default.
- Offer explicit, provider-compatible fallback modes for DeepSeek JSON Output and MiniMax Function Calling.
- Apply the selected mode to existing typed-output examples through the shared `agents_config.py` import path, without editing each example.
- Keep the existing `output_type` schema as the local source of truth and validate every final value through the SDK's existing typed-output validator.
- Preserve ordinary tools, MCP tools, and handoffs; never make a formatting function compete with application tools in the same model turn.
- Make extra requests, validation failures, and unsupported combinations visible rather than silently claiming typed success.

## Non-goals

- Do not change the teaching examples' declared Pydantic/TypedDict schemas or their intended output types.
- Do not infer the provider from the model ID. Model aliases and relays are not reliable provider identifiers.
- Do not automatically retry every native failure with a different format. Fallback is selected explicitly by configuration.
- Do not promise that prompt instructions alone enforce a schema.
- Do not support typed-output fallback in streamed runs in the first version. The repository currently has no `output_type` agent run through `Runner.run_streamed`.
- Do not change native OpenAI Responses behavior or plain-text agents.

## Configuration and routing

Add an opt-in environment setting:

```dotenv
# Unset: keep native Agents SDK JSON Schema behavior.
#AGENT_STRUCTURED_OUTPUT_MODE=deepseek_json
#AGENT_STRUCTURED_OUTPUT_MODE=minimax_function
# Required for DeepSeek JSON mode when the agent has no max_tokens setting.
#AGENT_STRUCTURED_OUTPUT_MAX_TOKENS=2048
```

The allowed values are `deepseek_json` and `minimax_function`; unset means native mode. Invalid values fail during `agents_config.py` initialization with a clear configuration error. These adapters require the Chat Completions route. Selecting either adapter while `OPENAI_AGENTS_CHAT_API=0` must fail early rather than send a Chat Completions-specific request through Responses.

The setting is explicit and endpoint-wide: the operator chooses it to match the provider/relay configured by `OPENAI_BASE_URL`. This first version does not auto-switch modes per model ID. The mode must be documented as suitable only for endpoints that implement its protocol.

For DeepSeek JSON Output, preserve `ModelSettings.max_tokens` when the example sets it. If it is absent, require `AGENT_STRUCTURED_OUTPUT_MAX_TOKENS` and apply that positive integer budget; fail before sending the request if neither source supplies a budget. This addresses DeepSeek's documented truncation/empty-content caveat without imposing an arbitrary limit on all examples.

## Shared adapter integration

Install one `OpenAIChatCompletionsModel` subclass adapter from the existing `_configure_agents_sdk()` import-time setup in `agents_config.py`. It delegates ordinary model execution to the SDK implementation and intervenes only when `output_schema` is present and an explicit fallback mode is selected.

Use an import-time wrapper around the public `OpenAIProvider.get_model` factory to return a cached adapter instance for each provider/model pair. The adapter must remain an `OpenAIChatCompletionsModel` subclass, not a generic delegating `Model`, so SDK code that uses `isinstance(..., OpenAIChatCompletionsModel)` retains its existing guardrail behavior. Do not patch the private response-format converter. Because the factory wrapper is an SDK extension seam rather than a documented registration API, pin/test the supported Agents SDK version and add a smoke test that confirms the adapter is selected for both explicit `Agent(model=...)` names and the SDK default model path.

No adapter behavior is applied when mode is unset, the request is plain text, or the agent has no typed output. Native mode remains byte-for-byte equivalent in request semantics.

## Provider-specific protocols

### DeepSeek JSON Output

DeepSeek's documented JSON Output mode uses Chat Completions `response_format: {"type":"json_object"}`. The system/user prompt must contain the word “JSON”, describe the expected structure, and include a JSON-shaped example. The SDK continues to receive the original `output_schema`, so `Runner` still parses and validates the result against the original type after generation.

For typed requests with no application tools or handoffs:

1. Prepend a short formatting instruction to `system_instructions`, including `output_schema.json_schema()` and a representative JSON example for the declared shape.
2. Merge into `ModelSettings.extra_body`, preserving any existing entries while overriding only `response_format` with `{"type":"json_object"}`. Do not use `extra_args`; it collides with the SDK's explicit `response_format` keyword.
3. Pass the response through the normal SDK output conversion and strict `output_schema.validate_json` path.

DeepSeek JSON Output guarantees JSON syntax, not schema compliance. Empty content, malformed JSON, or schema mismatch remains a typed-output failure. Do not fabricate `{}`, strip arbitrary prose, or report success based on a prompt-only response.

### MiniMax Function Calling

MiniMax TokenHub's guide documents Chat Completions and Function Calling, but does not establish that JSON Schema `response_format` is supported by the relevant model/endpoint. The fallback therefore uses a synthetic, formatting-only function whose parameter schema is derived from `output_schema.json_schema()`; set the synthetic tool's strict flag to false unless TokenHub's current documentation and live validation establish strict tool-schema support.

For typed requests with no application tools or handoffs:

1. Add only the synthetic formatting function to the model request and force that function via `tool_choice`.
2. Convert the returned function arguments into the assistant JSON text consumed by the existing typed-output validator. Remove the synthetic `ResponseFunctionToolCall` from the `ModelResponse`; it is an adapter detail and must not reach the runner as an application tool call.
3. Validate arguments locally against the original `output_schema`. Missing calls, invalid argument JSON, or schema mismatch fail clearly.

### Agents with tools, MCP, or handoffs

Never combine the synthetic formatting function with an agent's ordinary tools, MCP tools, or handoffs: observed provider behavior shows tool invocation can silently be skipped when structured response constraints are present.

For either fallback mode, use two phases when `tools` or `handoffs` are present:

1. **Work phase:** delegate to the original Chat Completions model with the real tools/handoffs intact and `output_schema=None`. Return any function-tool call or handoff unchanged so the Agents SDK executes its normal loop. Repeat naturally on later runner turns.
2. **Formatting phase:** only after the work phase returns a final assistant text response, issue a separate formatter request with no application tools/handoffs and the chosen provider protocol. Include the final work result and schema instructions, but do not expose prior tool calls as formatter tools. Return a normal assistant JSON message for the SDK's original output schema to validate.

Each formatting phase adds a billable model request. Usage accounting must include both provider calls; request IDs and raw usage diagnostics must not imply a single call if the SDK surface cannot represent the pair. Report a clear two-stage indicator in logs when this path is used.

## Unsupported combinations and errors

- If fallback mode is selected for Responses routing, fail before an API call with a configuration error.
- If a typed agent is executed through `Runner.run_streamed` while a fallback mode is enabled, raise a clear unsupported-combination error. Do not silently fall back to native schema or emit an unvalidated stream.
- If a provider rejects JSON mode or function calling, preserve the provider error and identify the selected fallback mode; do not retry as another provider protocol.
- If the formatter returns no content/function call, malformed JSON, or a value that fails the original schema, surface an error. A successful HTTP response is not itself a typed-output pass.
- Keep API keys out of prompts, diagnostics, and logs.

## Validation and acceptance criteria

1. **Default compatibility:** with `AGENT_STRUCTURED_OUTPUT_MODE` unset, request bodies and outcomes remain the current native JSON Schema behavior.
2. **DeepSeek request:** an HTTP-client-level test verifies `response_format.type == "json_object"`, the schema is present in the system instruction and includes a JSON example, the max-token budget comes from the agent or required fallback setting, and the final parsed object is validated by the original schema. Test that a truncated/empty response fails visibly.
3. **MiniMax request:** tests verify the formatter function schema comes from `output_schema`, the forced `tool_choice` names only that function, function arguments are converted to output text, and invalid/missing arguments fail.
4. **Tool safety:** tests with application tools, MCP servers, and handoffs verify phase one sends no output schema/formatter function and preserves original tools/handoffs; phase two contains no application tools and runs only after a terminal text response.
5. **Accounting:** tests confirm two-stage usage includes both model requests/tokens and records or labels both request diagnostics.
6. **Mode validation:** tests cover unset/native mode, each supported mode, unknown values, incompatible Responses routing, and streamed typed-output rejection.
7. **Real endpoint acceptance:** run representative actual `output_type=` examples against the configured TokenHub endpoint: one plain typed agent and one tool-bearing typed agent for each enabled fallback. A flat raw endpoint probe alone is insufficient. Record actual result status and sanitized provider request/response evidence.
8. **Documentation:** update `.env.example` and README with mode selection, protocol limits, extra-call cost for tool-bearing agents, and the fact that local schema validation remains authoritative.

## Known implementation risk

The current shared-hook candidate wraps `OpenAIProvider.get_model` at import time. This preserves the existing examples' import-side-effect architecture, but is sensitive to Agents SDK internals and must be guarded by an SDK-version smoke test. Pin the initial supported `openai-agents` runtime to the inspected 0.23.1 version; an SDK upgrade must update the compatibility test and revalidate the adapter before widening that pin. `extra_body` precedence is also an OpenAI SDK merge behavior; test the final serialized HTTP body rather than only testing `ModelSettings` construction.

## Vendor references

- [DeepSeek API JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode/) documents `response_format: {"type":"json_object"}`, a JSON prompt/example requirement, an appropriate `max_tokens` budget, and possible empty content. This is JSON mode, not JSON Schema enforcement.
- [Tencent Cloud TokenHub MiniMax guide](https://cloud.tencent.com/document/product/1823/132246) documents Chat Completions and MiniMax Function Calling. It does not establish JSON Schema `response_format` support for the tested models.
- [Tencent Cloud TokenHub DeepSeek guide](https://cloud.tencent.com/document/product/1823/132248) is the DeepSeek TokenHub guide (not a second MiniMax guide); its Chat Completions route is the deployment route to validate against.
