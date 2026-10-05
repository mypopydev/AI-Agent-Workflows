import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "chapter_02" / "01_first_agent.py"
TIMEOUT_SECONDS = 300
MODEL_NAME_TIMEOUT_SECONDS = 60


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
    output = re.sub(r"<think>.*?</think>", " ", output, flags=re.IGNORECASE | re.DOTALL)
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
    return " ".join(output.split())


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
    return "FAIL", detail.splitlines()[-1] if detail else "Process exited unsuccessfully"


def identify_model_name(model_id, api_key=None):
    env = os.environ.copy()
    env["AGENT_MODEL"] = model_id
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--report-model-name"],
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
    name = name[:200]
    return ("REPORTED", name) if name else ("EMPTY", "No name returned")


def report_model_name(model_id):
    import agents_config  # noqa: F401
    from agents import Agent, MultiProvider, RunConfig, Runner

    agent = Agent(
        name="Model Name Reporter",
        model=agents_config.model(model_id),
        instructions=(
            "Report the exact model identifier, name, and version you can confirm "
            "you are running. Do not infer or guess. If you cannot confirm the "
            "exact identity, reply exactly: Unable to confirm. Reply with no "
            "other text."
        ),
    )
    result = Runner.run_sync(
        agent,
        input="What exact model are you running?",
        run_config=RunConfig(
            model_provider=MultiProvider(unknown_prefix_mode="model_id")
        ),
    )
    print(result.final_output)


def fetch_models(base_url, api_key):
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return filter_available_models(json.load(response))


def main():
    load_dotenv(ROOT / ".env")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("OPENAI_BASE_URL and OPENAI_API_KEY must be set in .env")

    models = fetch_models(base_url, api_key)
    if not models:
        raise SystemExit("No available models returned by the provider")

    report_name = os.getenv("BATCH_TEST_REPORT_MODEL_NAME") == "1"
    suffix = (
        "; model-name probe enabled (one extra request per model)"
        if report_name
        else ""
    )
    print(
        f"Testing {len(models)} models sequentially; "
        f"per-model timeout={TIMEOUT_SECONDS}s{suffix}",
        flush=True,
    )
    counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    for index, model_id in enumerate(models, 1):
        status, detail = run_model(model_id, api_key)
        counts[status] += 1
        line = f"[{index}/{len(models)}] {status} {model_id} — {detail}"
        if report_name:
            name_status, reported_name = identify_model_name(model_id, api_key)
            line += (
                f" | self-reported name ({name_status}, unverified): "
                f"{reported_name}"
            )
        print(line, flush=True)

    print(f"SUMMARY {counts}")
    return int(counts["FAIL"] > 0 or counts["TIMEOUT"] > 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--report-model-name"]:
        report_model_name(os.environ["AGENT_MODEL"])
    else:
        raise SystemExit(main())
