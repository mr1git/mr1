"""Tests for mr1.mini.mem_rtvr"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from mr1.core.logger import Logger
from mr1.mini.mem_rtvr import (
    ingest_chunks,
    retrieve,
    _search_dumps,
    collection_stats,
)


@pytest.fixture
def tmp_logger(tmp_path):
    return Logger(tasks_dir=str(tmp_path / "tasks"))


@pytest.fixture
def mock_collection():
    """Provide a mock chromadb collection."""
    coll = MagicMock()
    coll.count.return_value = 0
    coll.name = "mr1_memory"
    return coll


class TestIngestChunks:
    def test_empty_list(self, tmp_logger):
        assert ingest_chunks([], logger=tmp_logger) == 0

    def test_ingest_calls_upsert(self, mock_collection, tmp_logger):
        chunks = [
            {"id": "c1", "text": "hello", "source": "test"},
            {"id": "c2", "text": "world", "source": "test"},
        ]
        with patch("mr1.mini.mem_rtvr._get_collection", return_value=mock_collection):
            count = ingest_chunks(chunks, logger=tmp_logger)

        assert count == 2
        mock_collection.upsert.assert_called_once()


class TestSearchDumps:
    def test_no_dumps_dir(self, tmp_path):
        with patch("mr1.mini.mem_rtvr._DUMPS_DIR", tmp_path / "nonexistent"):
            results = _search_dumps("query", 5)
        assert results == []

    def test_finds_matching_dump(self, tmp_path):
        dumps_dir = tmp_path / "dumps"
        dumps_dir.mkdir()
        (dumps_dir / "test.md").write_text("The quick brown fox jumped over the lazy dog.")

        with patch("mr1.mini.mem_rtvr._DUMPS_DIR", dumps_dir):
            results = _search_dumps("quick fox", 5)

        assert len(results) > 0
        assert results[0]["type"] == "dump"

    def test_no_match(self, tmp_path):
        dumps_dir = tmp_path / "dumps"
        dumps_dir.mkdir()
        (dumps_dir / "test.md").write_text("completely unrelated content")

        with patch("mr1.mini.mem_rtvr._DUMPS_DIR", dumps_dir):
            results = _search_dumps("zebra giraffe", 5)

        assert results == []


class TestRetrieve:
    def test_retrieve_empty_collection(self, mock_collection, tmp_logger, tmp_path):
        with patch("mr1.mini.mem_rtvr._get_collection", return_value=mock_collection), \
             patch("mr1.mini.mem_rtvr._DUMPS_DIR", tmp_path / "nonexistent"):
            results = retrieve("test query", logger=tmp_logger)
        assert results == []


class TestCollectionStats:
    def test_returns_dict(self, mock_collection):
        with patch("mr1.mini.mem_rtvr._get_collection", return_value=mock_collection):
            stats = collection_stats()
        assert "count" in stats
        assert "name" in stats
