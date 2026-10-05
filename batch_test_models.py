import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "chapter_02" / "01_first_agent.py"
TIMEOUT_SECONDS = 300


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
        return "PASS", result.stdout.strip()

    detail = (result.stderr or result.stdout).strip()
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    return "FAIL", detail.splitlines()[-1] if detail else "Process exited unsuccessfully"


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

    print(
        f"Testing {len(models)} models sequentially; "
        f"per-model timeout={TIMEOUT_SECONDS}s",
        flush=True,
    )
    counts = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    for index, model_id in enumerate(models, 1):
        status, detail = run_model(model_id, api_key)
        counts[status] += 1
        print(f"[{index}/{len(models)}] {status} {model_id} — {detail}", flush=True)

    print(f"SUMMARY {counts}")
    return int(counts["FAIL"] > 0 or counts["TIMEOUT"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
