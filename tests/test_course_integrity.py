"""Repository-level checks that do not require API keys or network access."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_PATTERN = re.compile(r"chapter_(\d{2})_.+/(\d{2})_.+\.py$")


def chapter_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.glob("chapter_*/*.py")
        if CHAPTER_PATTERN.search(path.relative_to(ROOT).as_posix())
    )


def course_text(include_html: bool = False) -> str:
    files = chapter_files()
    if include_html:
        files += [path.with_suffix(".html") for path in files]
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


class CourseIntegrityTests(unittest.TestCase):
    def test_has_chapters_zero_through_thirty_six(self) -> None:
        files = chapter_files()
        numbers = [int(path.parent.name.split("_")[1]) for path in files]
        self.assertEqual(numbers, list(range(37)))

    def test_all_python_sources_parse(self) -> None:
        for path in [ROOT / "build_html.py", *chapter_files()]:
            with self.subTest(path=path.relative_to(ROOT)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_each_chapter_has_freshness_marker(self) -> None:
        marker = re.compile(r"内容核对：2026-08-(?:01|09)")
        for path in chapter_files():
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertRegex(path.read_text(encoding="utf-8"), marker)

    def test_generated_html_matches_current_source(self) -> None:
        for path in chapter_files():
            html_path = path.with_suffix(".html")
            with self.subTest(path=html_path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                html = html_path.read_text(encoding="utf-8")
                self.assertIn(f"generated-from-sha256:{digest}", html)
                self.assertIn("内容核对：2026-08-", html)
                self.assertNotIn("\x00PH", html)

    def test_removed_protocol_framework_and_model_content_does_not_return(self) -> None:
        joined = course_text(include_html=True)
        forbidden = {
            "langgraph.prebuilt import create_react_agent": "deprecated LangGraph helper",
            "POST /a2a tasks/send": "pre-1.0 A2A operation",
            '"protocolVersion": "2025-06-18"': "session-oriented MCP snapshot",
            "taskStatusUpdate": "incorrect A2A v1 event wrapper",
            "taskArtifactUpdate": "incorrect A2A v1 event wrapper",
            "GPT-4o Vision 或 Claude 3.5 Sonnet": "stale multimodal recommendation",
            "它不是一个真实存在的开源项目": "OpenClaw existence error",
            "Harness 是一个假设的编码 Agent": "OpenHarness existence error",
            "便宜 100 倍": "unsupported Computer Use multiplier",
            "90% 的生产失败率": "unsupported Agentic RAG failure rate",
            "str(eval(": "unsafe calculator implementation",
        }
        for needle, reason in forbidden.items():
            with self.subTest(reason=reason):
                self.assertNotIn(needle, joined)

    def test_current_protocol_and_agent_stack_contracts_are_present(self) -> None:
        joined = course_text()
        required = {
            '"statusUpdate"': "A2A v1 status event",
            '"artifactUpdate"': "A2A v1 artifact event",
            "Mcp-Method": "MCP routing headers",
            "server/discover": "MCP discovery",
            "ttlMs": "MCP list caching",
            "programmatic_tool_calling": "OpenAI Programmatic Tool Calling",
            "allowed_callers": "programmatic tool allowlist",
            "previous_response_id": "Responses continuation",
            "background=true": "long-running Responses",
            "approval.required": "front-end approval event",
            "Realtime API": "voice agent transport",
        }
        for needle, reason in required.items():
            with self.subTest(reason=reason):
                self.assertIn(needle, joined)

    def test_readmes_index_and_package_are_consistent(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_en = (ROOT / "README_EN.md").read_text(encoding="utf-8")
        index = (ROOT / "index.html").read_text(encoding="utf-8")
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

        self.assertIn("37 章", readme)
        self.assertIn("37 Chapters", readme_en)
        self.assertIn("37 章节", index)
        self.assertIn("37章", package["description"])
        self.assertEqual(package["scripts"]["build"], "python build_html.py --all")

        combined = "\n".join([readme, readme_en, index, package["description"]])
        self.assertNotIn("gpt-4o-mini", combined)
        self.assertNotIn("更新-2026.05", combined)
        self.assertNotIn("Updated-2026.05", combined)
        self.assertNotIn("�", combined)

    def test_python_310_has_a_compatible_numpy_branch(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for text in (pyproject, requirements):
            self.assertIn("numpy>=2.2,<2.3", text)
            self.assertIn("python_version <", text)

    def test_fastapi_fullstack_demo_serves_ui_health_and_sse(self) -> None:
        from fastapi.testclient import TestClient

        path = ROOT / "chapter_13_fastapi" / "13_fastapi_agent_service.py"
        spec = importlib.util.spec_from_file_location("agent_study_ch13", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with TestClient(module.app) as client:
            root = client.get("/")
            self.assertEqual(root.status_code, 200)
            self.assertIn("Agent Streaming Console", root.text)
            self.assertEqual(client.get("/health").json()["status"], "healthy")
            stream = client.post(
                "/agent/chat/stream",
                json={"message": "计算 2+2", "stream": True},
            )
            self.assertEqual(stream.status_code, 200)
            self.assertIn("event: done", stream.text)


if __name__ == "__main__":
    unittest.main()
