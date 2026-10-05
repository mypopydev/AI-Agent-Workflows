import io
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from dotenv import dotenv_values

from batch_test_models import (
    filter_available_models,
    format_output,
    identify_model_name,
    main,
    report_model_name,
    run_model,
)


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

    def test_format_output_removes_terminal_control_sequences(self):
        output = "\x1b[31mModel\x1b[0m \x1b]0;title\x07Name"

        self.assertEqual(format_output(output), "Model Name")

    def test_format_output_preserves_text_between_osc_hyperlink_markers(self):
        output = "\x1b]8;;https://example.invalid\x1b\\link here\x1b]8;;\x1b\\"

        self.assertEqual(format_output(output), "link here")

    @patch("batch_test_models.subprocess.run")
    def test_model_name_probe_returns_sanitized_self_report(self, subprocess_run):
        subprocess_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="<think>guessing</think>\nModel X v2\n",
            stderr="",
        )

        result = identify_model_name("model-x", "secret")

        self.assertEqual(result, ("REPORTED", "Model X v2"))
        self.assertEqual(subprocess_run.call_args.kwargs["timeout"], 60)
        self.assertEqual(
            subprocess_run.call_args.kwargs["env"]["AGENT_MODEL"], "model-x"
        )

    @patch("batch_test_models.subprocess.run", side_effect=OSError("launch failed"))
    def test_model_name_probe_records_launch_failure(self, subprocess_run):
        self.assertEqual(
            identify_model_name("model-x"), ("FAIL", "launch failed")
        )

    def test_name_reporter_passes_case_sensitive_model_id_explicitly(self):
        agent_module = types.ModuleType("agents")
        agent_module.Agent = Mock()
        agent_module.Runner = types.SimpleNamespace(
            run_sync=Mock(
                return_value=types.SimpleNamespace(final_output="Model name")
            )
        )
        config_module = types.ModuleType("agents_config")
        config_module.model = lambda model_id: model_id
        stdout = io.StringIO()

        with patch.dict(
            sys.modules,
            {"agents": agent_module, "agents_config": config_module},
        ), redirect_stdout(stdout):
            report_model_name("CaseSensitive-Model")

        self.assertEqual(
            agent_module.Agent.call_args.kwargs["model"], "CaseSensitive-Model"
        )

    @patch("batch_test_models.subprocess.run")
    def test_model_name_probe_records_timeout(self, subprocess_run):
        subprocess_run.side_effect = subprocess.TimeoutExpired("probe", 60)

        self.assertEqual(
            identify_model_name("model-x"), ("TIMEOUT", "Exceeded 60 seconds")
        )

    @patch("batch_test_models.identify_model_name")
    @patch("batch_test_models.run_model")
    @patch("batch_test_models.fetch_models", return_value=["model-x"])
    @patch("batch_test_models.load_dotenv")
    def test_model_name_probe_is_opt_in(
        self, load_dotenv, fetch_models, run_model, identify_model_name
    ):
        output = io.StringIO()
        with patch.dict(
            "os.environ",
            {
                "OPENAI_BASE_URL": "https://provider.example/v1",
                "OPENAI_API_KEY": "secret",
                "BATCH_TEST_REPORT_MODEL_NAME": "1",
            },
            clear=True,
        ), redirect_stdout(output):
            run_model.return_value = ("PASS", "example output")
            identify_model_name.return_value = ("REPORTED", "Model X")
            result = main()

        identify_model_name.assert_called_once_with("model-x", "secret")
        self.assertIn(
            "self-reported name (REPORTED, unverified): Model X", output.getvalue()
        )
        self.assertEqual(result, 0)

    @patch("batch_test_models.identify_model_name")
    @patch("batch_test_models.run_model", return_value=("PASS", "example output"))
    @patch("batch_test_models.fetch_models", return_value=["model-x"])
    @patch("batch_test_models.load_dotenv")
    def test_probe_timeout_does_not_change_example_result(
        self, load_dotenv, fetch_models, run_model, identify_model_name
    ):
        output = io.StringIO()
        identify_model_name.return_value = ("TIMEOUT", "Exceeded 60 seconds")
        with patch.dict(
            "os.environ",
            {
                "OPENAI_BASE_URL": "https://provider.example/v1",
                "OPENAI_API_KEY": "secret",
                "BATCH_TEST_REPORT_MODEL_NAME": "1",
            },
            clear=True,
        ), redirect_stdout(output):
            result = main()

        self.assertEqual(result, 0)
        self.assertIn("PASS model-x — example output", output.getvalue())
        self.assertIn("self-reported name (TIMEOUT, unverified)", output.getvalue())

    @patch("batch_test_models.identify_model_name")
    @patch("batch_test_models.run_model", return_value=("PASS", "example output"))
    @patch("batch_test_models.fetch_models", return_value=["model-x"])
    @patch("batch_test_models.load_dotenv")
    def test_model_name_probe_is_disabled_by_default(
        self, load_dotenv, fetch_models, run_model, identify_model_name
    ):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_BASE_URL": "https://provider.example/v1",
                "OPENAI_API_KEY": "secret",
            },
            clear=True,
        ), redirect_stdout(io.StringIO()):
            main()

        identify_model_name.assert_not_called()

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
