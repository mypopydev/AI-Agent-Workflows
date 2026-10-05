import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

from batch_test_models import filter_available_models, format_output, run_model


class FilterAvailableModelsTests(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {"BATCH_TEST_EXCLUDED_MODELS": "kimi-k2.7-code, deepseek-v4-flash"},
    )
    def test_excludes_models_configured_in_environment(self):
        catalog = {
            "data": [
                {"id": "kimi-k2.7-code"},
                {"id": "kimi-k2.5"},
                {"id": "minimax-m-2-5"},
                {"id": "deepseek-v4-flash"},
                {"id": "kimi-k-2-5"},
                {"id": "minimax-m2.5"},
            ]
        }

        self.assertEqual(
            filter_available_models(catalog),
            ["kimi-k-2-5", "kimi-k2.5", "minimax-m-2-5", "minimax-m2.5"],
        )

    def test_example_config_lists_models_denied_by_the_key(self):
        expected = "kimi-k-2-5,kimi-k2.5,minimax-m-2-5,minimax-m2.5"
        config_path = Path(__file__).resolve().parent / ".env.example"

        config = dotenv_values(config_path)

        self.assertEqual(config["BATCH_TEST_EXCLUDED_MODELS"], expected)

    def test_format_output_flattens_lines_and_removes_multiline_think_blocks(self):
        output = """Plan begins
<think>internal reasoning
spans multiple lines</think>
1. Define agents

2. Explore architectures"""

        self.assertEqual(
            format_output(output),
            "Plan begins 1. Define agents 2. Explore architectures",
        )

    @patch("batch_test_models.subprocess.run")
    def test_model_run_uses_a_300_second_timeout(self, subprocess_run):
        subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="done", stderr=""
        )

        result = run_model("kimi-k2.7-code")

        self.assertEqual(result, ("PASS", "done"))
        self.assertEqual(subprocess_run.call_args.kwargs["timeout"], 300)


if __name__ == "__main__":
    unittest.main()
