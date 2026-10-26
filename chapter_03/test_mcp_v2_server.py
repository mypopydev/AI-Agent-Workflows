import sys
import unittest
from pathlib import Path

from agents.mcp import MCPServerStdio, MCPServerStdioParams


ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = Path(__file__).with_name("01_claude_mcp_server.py").resolve()
MCP_CLI = Path(sys.executable).with_name("mcp")


class MCPV2ServerSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_server_exposes_and_runs_research_tool(self):
        async with MCPServerStdio(
            name="Research Tools",
            params=MCPServerStdioParams(
                command=str(MCP_CLI),
                args=["run", str(SERVER_SCRIPT)],
                cwd=str(ROOT),
            ),
        ) as server:
            tools = await server.list_tools()
            self.assertIn("get_research_sources", {tool.name for tool in tools})

            result = await server.call_tool("get_research_sources", {})
            self.assertEqual(
                result.structured_content["result"],
                ["Wikipedia", "Google", "YouTube"],
            )


if __name__ == "__main__":
    unittest.main()
