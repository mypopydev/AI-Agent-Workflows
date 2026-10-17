import os
import pathlib as _pathlib
import sys as _sys

_root = next(p for p in _pathlib.Path(__file__).resolve().parents
             if (p / "agents_config.py").is_file())
_sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

if os.getenv("OPENAI_BASE_URL") and not os.getenv("AGENT_STRUCTURED_OUTPUT_MODE", "").strip():
    os.environ["AGENT_STRUCTURED_OUTPUT_MODE"] = "deepseek_json"

if (
    os.getenv("AGENT_STRUCTURED_OUTPUT_MODE", "").strip() == "deepseek_json"
    and not os.getenv("AGENT_STRUCTURED_OUTPUT_MAX_TOKENS", "").strip()
):
    os.environ["AGENT_STRUCTURED_OUTPUT_MAX_TOKENS"] = "2048"

import agents_config  # noqa: E402,F401

from agents import Agent, Runner, function_tool
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

# Agent Instructions
instructions = """
You are a research planning assistant.

**TASK INSTRUCTIONS**
- You will be given a research topic.
- Begin by using the tool get_research_sources() 
to get a list of available research sources. 
- Constrain your research plan 
only to use the available research sources.
- Your task is to provide a plan for researching this topic.
- Output 5 concise tasks and specify which of the 
available research sources will be used for each task.
"""

class Task(TypedDict):
    step: int
    """Task Step."""
    research_source: str
    """The source to search."""
    description: str
    """Task description."""

class ResearchPlanModel(BaseModel):
    tasks: list[Task]
    """Numbered tasks for research."""
    model_config = ConfigDict(extra='forbid')
    

@function_tool
def get_research_sources() -> list[str]:
    """Provides a list of research sources."""
    search_sources = [
        "Wikipedia",
        "Google",
        "YouTube",
    ]
    return search_sources    
    
agent = Agent(
    name="Research Planner", 
    instructions=instructions,
    output_type=ResearchPlanModel,
    tools=[get_research_sources],  # Register the tool with the agent
    )


def print_run_usage(result):
    usage = result.context_wrapper.usage
    configured_model = os.getenv("AGENT_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL")

    print("\nRun details:")
    if configured_model:
        print(f"Model (configured): {configured_model}")
    else:
        print("Model (configured): Agents SDK default (not explicitly configured)")
    print("The actual provider-served model may differ from this configuration.")
    print(f"Requests: {usage.requests}")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    def token_detail(details, name):
        value = getattr(details, name, None)
        return value if value is not None else "not reported"

    print(
        "Cached input tokens: "
        f"{token_detail(usage.input_tokens_details, 'cached_tokens')}"
    )
    print(
        "Cache write input tokens: "
        f"{token_detail(usage.input_tokens_details, 'cache_write_tokens')}"
    )
    print(
        "Reasoning output tokens: "
        f"{token_detail(usage.output_tokens_details, 'reasoning_tokens')}"
    )
    if usage.request_usage_entries:
        for index, request in enumerate(usage.request_usage_entries, start=1):
            print(
                f"Request {index}: input={request.input_tokens}, "
                f"output={request.output_tokens}, total={request.total_tokens}"
            )
            print(
                "  Cached input tokens: "
                f"{token_detail(request.input_tokens_details, 'cached_tokens')}"
            )
            print(
                "  Cache write input tokens: "
                f"{token_detail(request.input_tokens_details, 'cache_write_tokens')}"
            )
            print(
                "  Reasoning output tokens: "
                f"{token_detail(request.output_tokens_details, 'reasoning_tokens')}"
            )
    else:
        print("Per-request token details: not available")
    print("Note: the SDK may normalize omitted provider detail fields to 0.")


input = "learn about AI agents"

result = Runner.run_sync(
    agent, 
    input=input,
    )

print(result.final_output)
print_run_usage(result)