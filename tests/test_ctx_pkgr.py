"""Tests for mr1.mini.ctx_pkgr"""

from unittest.mock import patch

import pytest
from mr1.core.logger import Logger
from mr1.mini.ctx_pkgr import package, _retrieve_and_trim, _format_memory_block


@pytest.fixture
def tmp_logger(tmp_path):
    return Logger(tasks_dir=str(tmp_path / "tasks"))


class TestPackage:
    def test_basic_package(self, tmp_logger):
        with patch("mr1.mini.ctx_pkgr.mem_rtvr.retrieve", return_value=[]):
            ctx = package(
                task_id="t1",
                task_description="read the README",
                logger=tmp_logger,
            )

        assert ctx["task_id"] == "t1"
        assert "read the README" in ctx["instructions"]

    def test_package_with_tools_and_paths(self, tmp_logger):
        with patch("mr1.mini.ctx_pkgr.mem_rtvr.retrieve", return_value=[]):
            ctx = package(
                task_id="t1",
                task_description="check files",
                allowed_tools=["Read", "Glob"],
                file_paths=["/path/to/file.py"],
                logger=tmp_logger,
            )

        assert ctx["allowed_tools"] == ["Read", "Glob"]
        assert ctx["file_paths"] == ["/path/to/file.py"]

    def test_package_with_extra_context(self, tmp_logger):
        with patch("mr1.mini.ctx_pkgr.mem_rtvr.retrieve", return_value=[]):
            ctx = package(
                task_id="t1",
                task_description="do thing",
                extra_context="Some prior context here.",
                logger=tmp_logger,
            )

        assert "ADDITIONAL CONTEXT" in ctx["instructions"]
        assert "Some prior context here." in ctx["instructions"]

    def test_package_includes_memory(self, tmp_logger):
        mock_results = [
            {"text": "relevant memory chunk", "source": "dump:test.md", "score": 0.8, "type": "dump"},
        ]
        with patch("mr1.mini.ctx_pkgr.mem_rtvr.retrieve", return_value=mock_results):
            ctx = package(
                task_id="t1",
                task_description="task with memory",
                logger=tmp_logger,
            )

        assert "RELEVANT MEMORY" in ctx["instructions"]
        assert "relevant memory chunk" in ctx["instructions"]


class TestRetrieveAndTrim:
    def test_filters_low_relevance(self, tmp_logger):
        mock_results = [
            {"text": "good", "source": "s1", "score": 0.5, "type": "rag"},
            {"text": "bad", "source": "s2", "score": 0.01, "type": "rag"},
        ]
        with patch("mr1.mini.ctx_pkgr.mem_rtvr.retrieve", return_value=mock_results):
            trimmed = _retrieve_and_trim("query", tmp_logger)

        assert len(trimmed) == 1
        assert trimmed[0]["text"] == "good"

    def test_respects_max_chunks(self, tmp_logger):
        mock_results = [
            {"text": f"chunk {i}", "source": f"s{i}", "score": 0.9, "type": "rag"}
            for i in range(10)
        ]
        with patch("mr1.mini.ctx_pkgr.mem_rtvr.retrieve", return_value=mock_results):
            trimmed = _retrieve_and_trim("query", tmp_logger)

        assert len(trimmed) <= 3  # _MAX_MEMORY_CHUNKS


class TestFormatMemoryBlock:
    def test_formats_correctly(self):
        chunks = [
            {"text": "first chunk", "source": "dump:a.md", "score": 0.8},
            {"text": "second chunk", "source": "rag", "score": 0.6},
        ]
        block = _format_memory_block(chunks)
        assert "RELEVANT MEMORY:" in block
        assert "[dump:a.md]" in block
        assert "first chunk" in block
        assert "[rag]" in block
