import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "chapter_02" / "07_agent_with_tool.py"
TYPED_OUTPUT_SCRIPTS = tuple(
    ROOT / "chapter_02" / name
    for name in (
        "03_output_types_basic.py",
        "04_output_types.py",
        "05_output_types_fixed.py",
        "06_agent_with_tracing.py",
        "07_agent_with_tool.py",
        "07x_agent_with_tool_extended.py",
        "08_agent_with_tools_tracing.py",
    )
)


class AgentWithToolRegressionTests(unittest.TestCase):
    def _run_mocked_script(self, overrides, script=SCRIPT, expected_calls=1):
        env = os.environ.copy()
        env.pop("AGENT_STRUCTURED_OUTPUT_MODE", None)
        env.pop("AGENT_STRUCTURED_OUTPUT_MAX_TOKENS", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_AGENTS_CHAT_API", None)
        env.update(overrides)
        harness = """
import contextlib
import io
import json
import os
import runpy
import sys
from types import SimpleNamespace

import dotenv
import structured_output_fallback
from agents import Runner

install_calls = []
runner_calls = []
structured_output_fallback.install = (
    lambda mode, budget: install_calls.append([mode, budget])
)
dotenv.load_dotenv = lambda *args, **kwargs: False
usage = SimpleNamespace(
    requests=1,
    input_tokens=12,
    output_tokens=4,
    total_tokens=16,
    input_tokens_details=SimpleNamespace(
        cached_tokens=0, cache_write_tokens=0
    ),
    output_tokens_details=SimpleNamespace(reasoning_tokens=0),
    request_usage_entries=[],
)
Runner.run_sync = lambda *args, **kwargs: (
    runner_calls.append(args),
    SimpleNamespace(
        final_output="mock plan",
        context_wrapper=SimpleNamespace(usage=usage),
    ),
)[1]
output = io.StringIO()
with contextlib.redirect_stdout(output):
    runpy.run_path(sys.argv[1], run_name="__main__")
config = sys.modules["agents_config"]
print(json.dumps({
    "install_calls": install_calls,
    "mode": config.STRUCTURED_OUTPUT_MODE,
    "max_tokens": config.STRUCTURED_OUTPUT_MAX_TOKENS,
    "runner_calls": len(runner_calls),
    "output": output.getvalue(),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", harness, str(script)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["runner_calls"], expected_calls)
        return result

    def test_typed_examples_keep_native_mode_unless_configured(self):
        for script in TYPED_OUTPUT_SCRIPTS:
            with self.subTest(script=script.name):
                result = self._run_mocked_script(
                    {
                        "OPENAI_BASE_URL": "https://provider.invalid/v1",
                        "OPENAI_API_KEY": "test-key",
                        "AGENT_MODEL": "deepseek/deepseek-v4-flash",
                    },
                    script=script,
                )

                self.assertEqual(result["install_calls"], [])
                self.assertIsNone(result["mode"])
                self.assertIsNone(result["max_tokens"])
                self.assertIn("mock plan", result["output"])

    def test_typed_examples_use_explicit_shared_fallback_settings(self):
        for script in TYPED_OUTPUT_SCRIPTS:
            with self.subTest(script=script.name):
                result = self._run_mocked_script(
                    {
                        "OPENAI_BASE_URL": "https://provider.invalid/v1",
                        "OPENAI_API_KEY": "test-key",
                        "AGENT_MODEL": "deepseek/deepseek-v4-flash",
                        "AGENT_STRUCTURED_OUTPUT_MODE": "deepseek_json",
                        "AGENT_STRUCTURED_OUTPUT_MAX_TOKENS": "2048",
                    },
                    script=script,
                )

                self.assertEqual(
                    result["install_calls"], [["deepseek_json", 2048]]
                )
                self.assertEqual(result["mode"], "deepseek_json")
                self.assertEqual(result["max_tokens"], 2048)

    def test_explicit_fallback_settings_are_preserved(self):
        result = self._run_mocked_script(
            {
                "OPENAI_BASE_URL": "https://provider.invalid/v1",
                "OPENAI_API_KEY": "test-key",
                "AGENT_MODEL": "deepseek/deepseek-v4-flash",
                "AGENT_STRUCTURED_OUTPUT_MODE": "minimax_function",
                "AGENT_STRUCTURED_OUTPUT_MAX_TOKENS": "4096",
            }
        )

        self.assertEqual(result["install_calls"], [["minimax_function", 4096]])
        self.assertEqual(result["mode"], "minimax_function")
        self.assertEqual(result["max_tokens"], 4096)

    @unittest.skipUnless(
        os.getenv("RUN_DEEPSEEK_LIVE_TEST") == "1",
        "set RUN_DEEPSEEK_LIVE_TEST=1 to make billable provider calls",
    )
    def test_deepseek_structured_output_live(self):
        load_dotenv(ROOT / ".env")
        if not os.getenv("OPENAI_BASE_URL") or not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_BASE_URL and OPENAI_API_KEY are required")

        env = os.environ.copy()
        env["AGENT_MODEL"] = "deepseek/deepseek-v4-flash"
        env["AGENT_STRUCTURED_OUTPUT_MODE"] = "deepseek_json"
        env["AGENT_STRUCTURED_OUTPUT_MAX_TOKENS"] = "2048"
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan_output, usage_output = completed.stdout.split("\nRun details:", 1)
        self.assertTrue(plan_output.startswith("tasks="), plan_output)
        tasks = ast.literal_eval(plan_output.removeprefix("tasks="))
        self.assertEqual(len(tasks), 5)
        available_sources = {"Wikipedia", "Google", "YouTube"}
        for task in tasks:
            self.assertIsInstance(task["step"], int)
            self.assertIsInstance(task["description"], str)
            sources = {
                source.strip()
                for source in task["research_source"].split(",")
            }
            self.assertTrue(sources)
            self.assertLessEqual(sources, available_sources)
        self.assertIn(
            "Model (configured): deepseek/deepseek-v4-flash", usage_output
        )
        self.assertRegex(usage_output, r"Requests: [1-9]\d*")
        self.assertRegex(usage_output, r"Input tokens: [1-9]\d*")
        self.assertRegex(usage_output, r"Output tokens: [1-9]\d*")
        self.assertRegex(usage_output, r"Total tokens: [1-9]\d*")


if __name__ == "__main__":
    unittest.main()
