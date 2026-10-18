import asyncio
import io
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from dotenv import dotenv_values

from batch_test_models import (
    filter_available_models,
    format_output,
    identify_model_name,
    main,
    native_search_result,
    brave_search_result,
    probe_web_search,
    report_model_name,
    report_web_search,
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

    def test_format_output_preserves_numbered_lines_and_removes_think_blocks(
        self,
    ):
        output = """Plan begins
<think>internal reasoning
spans multiple lines</think>
1. Define agents

2. Explore architectures"""

        self.assertEqual(
            format_output(output),
            "Plan begins\n1. Define agents\n2. Explore architectures",
        )

    def test_native_search_result_requires_completed_search_call(self):
        result = SimpleNamespace(
            raw_responses=[
                SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="web_search_call", status="completed"
                        ),
                        SimpleNamespace(
                            type="message",
                            content=[
                                SimpleNamespace(
                                    annotations=[
                                        SimpleNamespace(
                                            type="url_citation",
                                            url="https://example.com/source",
                                        )
                                    ]
                                )
                            ],
                        ),
                    ]
                )
            ],
            final_output="Found current information.",
        )

        self.assertEqual(
            native_search_result(result),
            ("PASS", "Native web_search_call completed with cited sources"),
        )

    def test_native_search_result_requires_a_cited_source(self):
        result = SimpleNamespace(
            raw_responses=[
                SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="web_search_call", status="completed"
                        ),
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(annotations=[])],
                        ),
                    ]
                )
            ],
            final_output="Plausible answer without a citation",
        )

        self.assertEqual(
            native_search_result(result),
            ("FAIL", "Native web search returned no cited sources"),
        )

    def test_native_search_result_rejects_non_http_citations(self):
        result = SimpleNamespace(
            raw_responses=[
                SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="web_search_call", status="completed"
                        ),
                        SimpleNamespace(
                            type="message",
                            content=[
                                SimpleNamespace(
                                    annotations=[
                                        SimpleNamespace(
                                            type="url_citation",
                                            url="file:source",
                                        )
                                    ]
                                )
                            ],
                        ),
                    ]
                )
            ],
            final_output="Answer.",
        )

        self.assertEqual(
            native_search_result(result),
            ("FAIL", "Native web search returned no cited sources"),
        )

    def test_native_search_result_reports_when_model_skips_search(self):
        result = SimpleNamespace(
            raw_responses=[SimpleNamespace(output=[])],
            final_output="I think the answer is...",
        )

        self.assertEqual(
            native_search_result(result),
            ("NO_CALL", "Model did not invoke native web search"),
        )

    def test_brave_search_result_requires_tool_call_and_nonempty_output(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        type="function_call",
                        name="brave_web_search",
                        call_id="call-1",
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output=(
                        '{"title":"Example result",'
                        '"url":"https://example.com/result",'
                        '"description":"A web result"}'
                    ),
                    raw_item=SimpleNamespace(call_id="call-1", is_error=False),
                ),
            ],
            final_output="Here is the search summary.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("PASS", "brave_web_search returned structured results"),
        )

    def test_brave_search_result_accepts_multiple_json_content_blocks(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output=(
                        '{"title":"First result", '
                        '"url":"https://example.com/1"}\n'
                        '{"title":"Second result", '
                        '"url":"https://example.com/2"}'
                    ),
                    raw_item=SimpleNamespace(call_id="call-1", is_error=False),
                ),
            ],
            final_output="Search summary.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("PASS", "brave_web_search returned structured results"),
        )

    def test_brave_search_result_parses_agents_sdk_text_output_blocks(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output=[
                        {
                            "type": "text",
                            "text": (
                                '{"title":"Example result",'
                                '"url":"https://example.com/result"}'
                            ),
                        }
                    ],
                    raw_item=SimpleNamespace(call_id="call-1", is_error=False),
                ),
            ],
            final_output="Search summary.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("PASS", "brave_web_search returned structured results"),
        )

    def test_brave_search_result_requires_a_search_result_url(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output="No search results found",
                    raw_item=SimpleNamespace(call_id="call-1", is_error=False),
                ),
            ],
            final_output="No results found.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("FAIL", "brave_web_search returned no usable result"),
        )

    def test_brave_search_result_does_not_pass_on_unrelated_tool_output(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output=(
                        '{"help":{"title":"Documentation",'
                        '"url":"https://example.com/help"}}'
                    ),
                    raw_item=SimpleNamespace(
                        call_id="different-call", is_error=False
                    ),
                ),
            ],
            final_output="Unrelated answer.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("FAIL", "brave_web_search returned no results"),
        )

    def test_brave_search_result_does_not_pass_on_error_text_with_url(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output="Error: see https://docs.example.com/auth",
                    raw_item=SimpleNamespace(call_id="call-1", is_error=False),
                ),
            ],
            final_output="Error.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("FAIL", "brave_web_search returned an error"),
        )

    def test_brave_search_result_rejects_structured_error_output(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output=(
                        '{"error":"Unauthorized",'
                        '"documentation_url":"https://example.com/auth"}'
                    ),
                    raw_item=SimpleNamespace(call_id="call-1", is_error=False),
                ),
            ],
            final_output="Error.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("FAIL", "brave_web_search returned an error"),
        )

    def test_brave_search_result_does_not_pass_on_error_output(self):
        result = SimpleNamespace(
            new_items=[
                SimpleNamespace(
                    type="tool_call_item",
                    raw_item=SimpleNamespace(
                        name="brave_web_search", call_id="call-1"
                    ),
                ),
                SimpleNamespace(
                    type="tool_call_output_item",
                    output="Search failed",
                    raw_item=SimpleNamespace(call_id="call-1", is_error=True),
                ),
            ],
            final_output="Search failed.",
        )

        self.assertEqual(
            brave_search_result(result),
            ("FAIL", "brave_web_search returned an error"),
        )

    def test_format_output_removes_terminal_control_sequences(self):
        output = "\x1b[31mModel\x1b[0m \x1b]0;title\x07Name"

        self.assertEqual(format_output(output), "Model Name")

    def test_format_output_preserves_text_between_osc_hyperlink_markers(self):
        output = "\x1b]8;;https://example.invalid\x1b\\link here\x1b]8;;\x1b\\"

        self.assertEqual(format_output(output), "link here")

    @patch("batch_test_models.subprocess.run")
    def test_model_name_probe_returns_sanitized_self_report(
        self, subprocess_run
    ):
        subprocess_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="<think>guessing</think>\nModel X v2 secret\n",
            stderr="",
        )

        result = identify_model_name("model-x", "secret")

        self.assertEqual(result, ("REPORTED", "Model X v2 [REDACTED]"))
        self.assertEqual(subprocess_run.call_args.kwargs["timeout"], 60)
        self.assertEqual(
            subprocess_run.call_args.kwargs["env"]["AGENT_MODEL"], "model-x"
        )

    @patch("batch_test_models.subprocess.run")
    def test_model_name_probe_preserves_output_longer_than_200_characters(
        self, subprocess_run
    ):
        output = "Self-report: " + "x" * 200 + "\nConfigured ID check: Yes"
        subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=output, stderr=""
        )

        result = identify_model_name("model-x")

        self.assertEqual(result, ("REPORTED", output))

    @patch(
        "batch_test_models.subprocess.run",
        side_effect=OSError("launch failed"),
    )
    def test_model_name_probe_records_launch_failure(self, subprocess_run):
        self.assertEqual(
            identify_model_name("model-x"), ("FAIL", "launch failed")
        )

    @patch("batch_test_models.subprocess.run")
    def test_web_search_probe_skips_native_mode_on_chat_completions(
        self, subprocess_run
    ):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_BASE_URL": "https://provider.example/v1",
                "OPENAI_AGENTS_CHAT_API": "1",
            },
            clear=True,
        ):
            result = probe_web_search("model-x", "native")

        self.assertEqual(
            result, ("SKIP", "Native web search requires the Responses API")
        )
        subprocess_run.assert_not_called()

    @patch("batch_test_models.subprocess.run")
    def test_web_search_probe_skips_brave_when_api_key_is_missing(
        self, subprocess_run
    ):
        with patch.dict("os.environ", {}, clear=True):
            result = probe_web_search("model-x", "brave")

        self.assertEqual(result, ("SKIP", "BRAVE_API_KEY is not set"))
        subprocess_run.assert_not_called()

    @patch("batch_test_models.subprocess.run")
    def test_web_search_probe_reads_child_status(self, subprocess_run):
        subprocess_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "WEB_SEARCH_RESULT\tPASS\tNative web_search_call completed\n"
            ),
            stderr="",
        )

        result = probe_web_search("model-x", "native")

        self.assertEqual(result, ("PASS", "Native web_search_call completed"))
        self.assertEqual(
            subprocess_run.call_args.kwargs["env"]["AGENT_MODEL"], "model-x"
        )

    def test_native_web_search_probe_attaches_hosted_tool(self):
        agents_module = types.ModuleType("agents")
        agents_module.__path__ = []
        agents_module.Agent = Mock()
        agents_module.RunConfig = Mock(
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        agents_module.MultiProvider = Mock()
        agents_module.Runner = types.SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(
                    raw_responses=[
                        SimpleNamespace(
                            output=[
                                SimpleNamespace(
                                    type="web_search_call", status="completed"
                                ),
                                SimpleNamespace(
                                    type="message",
                                    content=[
                                        SimpleNamespace(
                                            annotations=[
                                                SimpleNamespace(
                                                    type="url_citation",
                                                    url=(
                                                        "https://example.com/"
                                                        "source"
                                                    ),
                                                )
                                            ]
                                        )
                                    ],
                                ),
                            ]
                        )
                    ],
                    final_output="Search result",
                )
            )
        )
        agents_module.WebSearchTool = Mock(return_value="native-web-search")
        settings_module = types.ModuleType("agents.model_settings")
        settings_module.ModelSettings = Mock(return_value="settings")
        config_module = types.ModuleType("agents_config")
        config_module.model = lambda model_id: model_id
        model_id = "DeepSeek/deepseek-V4-Flash"

        with patch.dict(
            sys.modules,
            {
                "agents": agents_module,
                "agents.model_settings": settings_module,
                "agents_config": config_module,
            },
        ):
            result = asyncio.run(report_web_search(model_id, "native"))

        self.assertEqual(
            result,
            ("PASS", "Native web_search_call completed with cited sources"),
        )
        self.assertEqual(
            agents_module.Agent.call_args.kwargs["model"], model_id
        )
        self.assertEqual(
            agents_module.Agent.call_args.kwargs["tools"],
            ["native-web-search"],
        )
        run_config = agents_module.Runner.run.call_args.kwargs["run_config"]
        self.assertEqual(
            run_config.model_provider,
            agents_module.MultiProvider.return_value,
        )
        agents_module.MultiProvider.assert_called_once_with(
            unknown_prefix_mode="model_id"
        )
        self.assertEqual(
            agents_module.Agent.call_args.kwargs["model_settings"], "settings"
        )

    def test_brave_web_search_probe_launches_stdio_mcp_tool(self):
        agents_module = types.ModuleType("agents")
        agents_module.__path__ = []
        agents_module.Agent = Mock()
        agents_module.RunConfig = Mock(
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        agents_module.MultiProvider = Mock()
        agents_module.Runner = types.SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(
                    new_items=[
                        SimpleNamespace(
                            type="tool_call_item",
                            raw_item=SimpleNamespace(
                                name="brave_web_search", call_id="call-1"
                            ),
                        ),
                        SimpleNamespace(
                            type="tool_call_output_item",
                            output=(
                                '{"web":{"results":[{"title":"Example result",'
                                '"url":"https://example.com/result"}]}}'
                            ),
                            raw_item=SimpleNamespace(
                                call_id="call-1", is_error=False
                            ),
                        ),
                    ],
                    final_output="Search result",
                )
            )
        )
        settings_module = types.ModuleType("agents.model_settings")
        settings_module.ModelSettings = Mock(return_value="settings")
        server = MagicMock()
        server.__aenter__ = AsyncMock(return_value=server)
        server.__aexit__ = AsyncMock(return_value=False)
        mcp_module = types.ModuleType("agents.mcp")
        mcp_module.MCPServerStdio = Mock(return_value=server)
        config_module = types.ModuleType("agents_config")
        config_module.model = lambda model_id: model_id
        model_id = "DeepSeek/deepseek-V4-Flash"

        with patch.dict(
            "os.environ", {"BRAVE_API_KEY": "brave-secret"}, clear=True
        ), patch.dict(
            sys.modules,
            {
                "agents": agents_module,
                "agents.model_settings": settings_module,
                "agents.mcp": mcp_module,
                "agents_config": config_module,
            },
        ):
            result = asyncio.run(report_web_search(model_id, "brave"))

        self.assertEqual(
            result, ("PASS", "brave_web_search returned structured results")
        )
        self.assertEqual(
            agents_module.Agent.call_args.kwargs["model"], model_id
        )
        run_config = agents_module.Runner.run.call_args.kwargs["run_config"]
        self.assertEqual(
            run_config.model_provider,
            agents_module.MultiProvider.return_value,
        )
        agents_module.MultiProvider.assert_called_once_with(
            unknown_prefix_mode="model_id"
        )
        server_kwargs = mcp_module.MCPServerStdio.call_args.kwargs
        self.assertTrue(server_kwargs["use_structured_content"])
        params = server_kwargs["params"]
        self.assertEqual(
            params["args"],
            [
                "-y",
                "@brave/brave-search-mcp-server@2.1.4",
                "--transport",
                "stdio",
            ],
        )
        self.assertEqual(params["env"]["BRAVE_API_KEY"], "brave-secret")

    def test_name_reporter_passes_case_sensitive_model_id_explicitly(self):
        agent_module = types.ModuleType("agents")
        agent_module.Agent = Mock()
        agent_module.Runner = types.SimpleNamespace(
            run_sync=Mock(
                side_effect=[
                    types.SimpleNamespace(final_output="Model X v2"),
                    types.SimpleNamespace(final_output="Yes, that matches."),
                ]
            )
        )
        agent_module.RunConfig = Mock(
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        agent_module.MultiProvider = Mock()
        config_module = types.ModuleType("agents_config")
        config_module.model = lambda model_id: model_id
        stdout = io.StringIO()

        with patch.dict(
            sys.modules,
            {"agents": agent_module, "agents_config": config_module},
        ), redirect_stdout(stdout):
            report_model_name("DeepSeek/deepseek-V4-Flash")

        self.assertEqual(
            agent_module.Agent.call_args.kwargs["model"],
            "DeepSeek/deepseek-V4-Flash",
        )
        self.assertNotIn(
            "DeepSeek/deepseek-V4-Flash",
            agent_module.Agent.call_args.kwargs["instructions"],
        )
        self.assertEqual(agent_module.Runner.run_sync.call_count, 2)
        self.assertEqual(
            agent_module.Runner.run_sync.call_args_list[0].kwargs["input"],
            "What exact model ID, name, and version are you running? "
            "Do not guess; say if you cannot confirm.",
        )
        self.assertNotIn(
            "DeepSeek/deepseek-V4-Flash",
            agent_module.Runner.run_sync.call_args_list[0].kwargs["input"],
        )
        self.assertIn(
            "DeepSeek/deepseek-V4-Flash",
            agent_module.Runner.run_sync.call_args_list[1].kwargs["input"],
        )
        self.assertIn("Self-report: Model X v2", stdout.getvalue())
        self.assertIn(
            "Configured ID check: Yes, that matches.", stdout.getvalue()
        )
        self.assertEqual(
            agent_module.MultiProvider.call_args.kwargs["unknown_prefix_mode"],
            "model_id",
        )
        self.assertEqual(
            agent_module.Runner.run_sync.call_args.kwargs[
                "run_config"
            ].model_provider,
            agent_module.MultiProvider.return_value,
        )

    @patch("batch_test_models.subprocess.run")
    def test_model_name_probe_records_timeout(self, subprocess_run):
        subprocess_run.side_effect = subprocess.TimeoutExpired("probe", 60)

        self.assertEqual(
            identify_model_name("model-x"), ("TIMEOUT", "Exceeded 60 seconds")
        )

    @patch("batch_test_models.probe_web_search")
    @patch("batch_test_models.run_model")
    @patch("batch_test_models.fetch_models", return_value=["model-x"])
    @patch("batch_test_models.load_dotenv")
    def test_web_search_probe_runs_both_modes_when_enabled(
        self, load_dotenv, fetch_models, run_model, probe_web_search
    ):
        output = io.StringIO()
        probe_web_search.side_effect = [
            ("PASS", "Native web_search_call completed with cited sources"),
            ("NO_CALL", "Model did not invoke brave_web_search"),
        ]
        with patch.dict(
            "os.environ",
            {
                "OPENAI_BASE_URL": "https://provider.example/v1",
                "OPENAI_API_KEY": "secret",
                "BRAVE_API_KEY": "brave-secret",
                "BATCH_TEST_WEB_SEARCH": "1",
            },
            clear=True,
        ), redirect_stdout(output):
            run_model.return_value = ("PASS", "example output")
            result = main()

        self.assertEqual(
            probe_web_search.call_args_list,
            [
                unittest.mock.call("model-x", "native", "secret"),
                unittest.mock.call("model-x", "brave", "secret"),
            ],
        )
        self.assertIn("Native Responses web search (PASS)", output.getvalue())
        self.assertIn("Brave MCP web search (NO_CALL)", output.getvalue())
        self.assertEqual(result, 0)

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
            "self-reported name (REPORTED, unverified): Model X",
            output.getvalue(),
        )
        self.assertEqual(result, 0)

    @patch("batch_test_models.identify_model_name")
    @patch(
        "batch_test_models.run_model",
        return_value=("PASS", "example output"),
    )
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
        self.assertIn(
            "self-reported name (TIMEOUT, unverified)", output.getvalue()
        )

    @patch("batch_test_models.probe_web_search")
    @patch("batch_test_models.identify_model_name")
    @patch(
        "batch_test_models.run_model",
        return_value=("PASS", "example output"),
    )
    @patch("batch_test_models.fetch_models", return_value=["model-x"])
    @patch("batch_test_models.load_dotenv")
    def test_optional_probes_are_disabled_by_default(
        self,
        load_dotenv,
        fetch_models,
        run_model,
        identify_model_name,
        probe_web_search,
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
        probe_web_search.assert_not_called()

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
