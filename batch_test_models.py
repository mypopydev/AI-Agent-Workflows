import asyncio
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "chapter_02" / "01_first_agent.py"
TIMEOUT_SECONDS = 300
MODEL_NAME_TIMEOUT_SECONDS = 60
WEB_SEARCH_TIMEOUT_SECONDS = 180
WEB_SEARCH_RESULT_PREFIX = "WEB_SEARCH_RESULT\t"


def filter_available_models(catalog):
    excluded_models = {
        model.strip()
        for model in os.getenv("BATCH_TEST_EXCLUDED_MODELS", "").split(",")
        if model.strip()
    }
    return sorted(
        {
            model["id"]
            for model in catalog.get("data", [])
            if isinstance(model, dict)
            and isinstance(model.get("id"), str)
            and model["id"] not in excluded_models
        }
    )


def format_output(output):
    output = re.sub(
        r"<think>.*?</think>", " ", output, flags=re.IGNORECASE | re.DOTALL
    )
    output = re.sub(
        r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x1b\x07\x9c]*(?:\x07|\x1b\\|\x9c))",
        " ",
        output,
    )
    output = "".join(
        char
        for char in output
        if unicodedata.category(char) != "Cc" or char in "\r\n\t"
    )
    lines = (
        " ".join(line.split())
        for line in output.splitlines()
        if line.strip()
    )
    return "\n".join(lines)


def run_model(model_id, api_key=None):
    env = os.environ.copy()
    env["AGENT_MODEL"] = model_id
    try:
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"Exceeded {TIMEOUT_SECONDS} seconds"

    if result.returncode == 0:
        return "PASS", format_output(result.stdout)

    detail = (result.stderr or result.stdout).strip()
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    return (
        "FAIL",
        detail.splitlines()[-1] if detail else "Process exited unsuccessfully",
    )


def identify_model_name(model_id, api_key=None):
    env = os.environ.copy()
    env["AGENT_MODEL"] = model_id
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--report-model-name",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=MODEL_NAME_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"Exceeded {MODEL_NAME_TIMEOUT_SECONDS} seconds"
    except OSError as exc:
        detail = str(exc)
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        return "FAIL", detail

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        return "FAIL", detail.splitlines()[-1] if detail else "Probe failed"

    name = format_output(result.stdout)
    if api_key:
        name = name.replace(api_key, "[REDACTED]")
    return ("REPORTED", name) if name else ("EMPTY", "No name returned")


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def native_search_result(result):
    output_items = [
        item
        for response in result.raw_responses
        for item in _field(response, "output", []) or []
    ]
    calls = [
        item
        for item in output_items
        if _field(item, "type") == "web_search_call"
    ]
    if not calls:
        return "NO_CALL", "Model did not invoke native web search"
    if any(_field(call, "status") != "completed" for call in calls):
        return "FAIL", "Native web_search_call did not complete"

    citations = [
        annotation
        for message in output_items
        if _field(message, "type") == "message"
        for content in _field(message, "content", []) or []
        for annotation in _field(content, "annotations", []) or []
        if _field(annotation, "type") == "url_citation"
        and _is_http_url(_field(annotation, "url", ""))
    ]
    if not citations:
        return "FAIL", "Native web search returned no cited sources"
    if not result.final_output:
        return "FAIL", "Native web search returned no final answer"
    return "PASS", "Native web_search_call completed with cited sources"


def _is_http_url(value):
    parsed = urlparse(str(value))
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _has_brave_search_result(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            lines = [line for line in value.splitlines() if line.strip()]
            if len(lines) == 1 and lines[0] == value:
                return False
            return any(_has_brave_search_result(line) for line in lines)
        if not isinstance(parsed, (Mapping, list, tuple)):
            return False
        value = parsed
    if isinstance(value, (list, tuple)):
        return any(_has_brave_search_result(item) for item in value)
    if not isinstance(value, Mapping):
        return False
    if (
        isinstance(_field(value, "title"), str)
        and _field(value, "title").strip()
        and _is_http_url(_field(value, "url", ""))
    ):
        return True
    web_results = _field(_field(value, "web", {}), "results", []) or []
    if any(_has_brave_search_result(item) for item in web_results):
        return True
    return _has_brave_search_result(_field(value, "text", ""))


def _is_search_error(item):
    raw_item = _field(item, "raw_item")
    if _field(
        item,
        "is_error",
        _field(raw_item, "is_error", _field(raw_item, "isError", False)),
    ):
        return True
    value = _field(item, "output", item)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, Mapping):
        if value.get("isError") or value.get("is_error"):
            return True
        if any(key in value for key in ("error", "errors")) and not any(
            key in value for key in ("web", "title", "url")
        ):
            return True
    text = str(value).lstrip().lower()
    return text.startswith(
        ("error:", "error ", "failed:", "failure:", "search failed")
    )


def brave_search_result(result):
    calls = [
        _field(item, "raw_item")
        for item in result.new_items
        if _field(item, "type") == "tool_call_item"
        and _field(_field(item, "raw_item"), "name") == "brave_web_search"
    ]
    if not calls:
        return "NO_CALL", "Model did not invoke brave_web_search"

    call_ids = {_field(call, "call_id") for call in calls}
    if None in call_ids:
        return "FAIL", "Could not correlate brave_web_search tool output"
    outputs = [
        item
        for item in result.new_items
        if _field(item, "type") == "tool_call_output_item"
        and _field(_field(item, "raw_item"), "call_id") in call_ids
    ]
    if not outputs:
        return "FAIL", "brave_web_search returned no results"
    if any(_is_search_error(item) for item in outputs):
        return "FAIL", "brave_web_search returned an error"
    if not any(
        _has_brave_search_result(_field(item, "output", ""))
        for item in outputs
    ):
        return "FAIL", "brave_web_search returned no usable result"
    if not result.final_output:
        return (
            "FAIL",
            "Search results returned but model produced no final answer",
        )
    return "PASS", "brave_web_search returned structured results"


def _env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _redact_search_secrets(text, api_key):
    for secret in (api_key, os.getenv("BRAVE_API_KEY")):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def probe_web_search(model_id, mode, api_key=None):
    if mode == "native" and os.getenv("OPENAI_BASE_URL") and _env_flag(
        "OPENAI_AGENTS_CHAT_API", True
    ):
        return "SKIP", "Native web search requires the Responses API"
    if mode == "brave" and not os.getenv("BRAVE_API_KEY"):
        return "SKIP", "BRAVE_API_KEY is not set"

    env = os.environ.copy()
    env["AGENT_MODEL"] = model_id
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--probe-web-search",
                mode,
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=WEB_SEARCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"Exceeded {WEB_SEARCH_TIMEOUT_SECONDS} seconds"
    except OSError as exc:
        return "FAIL", _redact_search_secrets(str(exc), api_key)

    if result.returncode != 0:
        detail = _redact_search_secrets(
            (result.stderr or result.stdout).strip(), api_key
        )
        return "FAIL", detail.splitlines()[-1] if detail else "Probe failed"

    marker = next(
        (
            line
            for line in reversed(result.stdout.splitlines())
            if line.startswith(WEB_SEARCH_RESULT_PREFIX)
        ),
        None,
    )
    if marker is None:
        return "FAIL", "Probe returned no result"
    _, status, detail = marker.split("\t", 2)
    return status, _redact_search_secrets(detail, api_key)


async def report_web_search(model_id, mode):
    import agents_config
    from agents import Agent, MultiProvider, RunConfig, Runner
    from agents.model_settings import ModelSettings

    run_config = RunConfig(
        model_provider=MultiProvider(unknown_prefix_mode="model_id")
    )

    if mode == "native":
        from agents import WebSearchTool

        agent = Agent(
            name="Native Web Search Probe",
            model=agents_config.model(model_id),
            instructions=(
                "Use the native web search tool and cite the source "
                "in your answer."
            ),
            tools=[WebSearchTool()],
            model_settings=ModelSettings(tool_choice="required"),
        )
        result = await Runner.run(
            agent,
            "Search the web for the official OpenAI Agents SDK tools "
            "documentation. Return one result title and its cited URL.",
            run_config=run_config,
        )
        return native_search_result(result)

    if mode == "brave":
        from agents.mcp import MCPServerStdio

        search_server = MCPServerStdio(
            name="Brave Search",
            params={
                "command": "npx",
                "args": [
                    "-y",
                    "@brave/brave-search-mcp-server@2.1.4",
                    "--transport",
                    "stdio",
                ],
                "env": {"BRAVE_API_KEY": os.environ["BRAVE_API_KEY"]},
            },
            use_structured_content=True,
        )
        async with search_server:
            agent = Agent(
                name="Brave MCP Search Probe",
                model=agents_config.model(model_id),
                instructions=(
                    "Use brave_web_search to search the web. Report a result "
                    "title and URL from the tool output."
                ),
                mcp_servers=[search_server],
                model_settings=ModelSettings(tool_choice="required"),
            )
            result = await Runner.run(
                agent,
                "Use brave_web_search with query "
                "'OpenAI Agents SDK tools documentation'. Return one result "
                "title and its URL.",
                run_config=run_config,
            )
        return brave_search_result(result)

    raise ValueError(f"Unknown web-search probe mode: {mode}")


def report_model_name(model_id):
    import agents_config  # noqa: F401
    from agents import Agent, MultiProvider, RunConfig, Runner

    agent = Agent(
        name="Model Name Reporter",
        model=agents_config.model(model_id),
        instructions=(
            "Answer based only on what you can confirm about the model "
            "serving this request. Do not infer or guess. If you cannot "
            "confirm, say so."
        ),
    )
    run_config = RunConfig(
        model_provider=MultiProvider(unknown_prefix_mode="model_id")
    )
    self_report = Runner.run_sync(
        agent,
        input=(
            "What exact model ID, name, and version are you running? "
            "Do not guess; say if you cannot confirm."
        ),
        run_config=run_config,
    )
    id_check = Runner.run_sync(
        agent,
        input=(
            f"The configured model ID is {model_id}. Does it exactly "
            "match the model serving this request? Reply Yes, No, or "
            "Unable to confirm."
        ),
        run_config=run_config,
    )
    print(f"Self-report: {self_report.final_output}")
    print(f"Configured ID check: {id_check.final_output}")


def fetch_models(base_url, api_key):
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return filter_available_models(json.load(response))


def main():
    load_dotenv(ROOT / ".env")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "OPENAI_BASE_URL and OPENAI_API_KEY must be set in .env"
        )

    models = fetch_models(base_url, api_key)
    if not models:
        raise SystemExit("No available models returned by the provider")

    report_name = os.getenv("BATCH_TEST_REPORT_MODEL_NAME") == "1"
    search_modes = (
        ("native", "brave")
        if os.getenv("BATCH_TEST_WEB_SEARCH") == "1"
        else ()
    )
    suffix = (
        "; model-name probe enabled (two extra requests per model)"
        if report_name
        else ""
    )
    if search_modes:
        suffix += "; native Responses and Brave MCP web-search probes enabled"
    print(
        f"Testing {len(models)} models sequentially; "
        f"per-model timeout={TIMEOUT_SECONDS}s{suffix}",
        flush=True,
    )
    counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    search_counts = {mode: {} for mode in search_modes}
    for index, model_id in enumerate(models, 1):
        status, detail = run_model(model_id, api_key)
        counts[status] += 1
        lines = [f"[{index}/{len(models)}] {status} {model_id} — {detail}"]
        if report_name:
            name_status, reported_name = identify_model_name(model_id, api_key)
            lines.append(
                "  self-reported name "
                f"({name_status}, unverified): {reported_name}"
            )
        for mode in search_modes:
            search_status, search_detail = probe_web_search(
                model_id, mode, api_key
            )
            search_counts[mode][search_status] = (
                search_counts[mode].get(search_status, 0) + 1
            )
            label = "Native Responses" if mode == "native" else "Brave MCP"
            lines.append(
                f"  {label} web search ({search_status}): {search_detail}"
            )
        print("\n".join(lines), flush=True)

    print(f"SUMMARY {counts}")
    for mode, mode_counts in search_counts.items():
        print(f"WEB_SEARCH {mode} {mode_counts}")
    return int(counts["FAIL"] > 0 or counts["TIMEOUT"] > 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--report-model-name"]:
        report_model_name(os.environ["AGENT_MODEL"])
    elif len(sys.argv) == 3 and sys.argv[1] == "--probe-web-search":
        status, detail = asyncio.run(
            report_web_search(os.environ["AGENT_MODEL"], sys.argv[2])
        )
        print(f"{WEB_SEARCH_RESULT_PREFIX}{status}\t{detail}")
    else:
        raise SystemExit(main())
