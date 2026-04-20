"""Tests for mr1.core.spawner"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest
from mr1.core.dispatcher import Dispatcher, PermissionDenied
from mr1.core.logger import Logger
from mr1.core.spawner import Spawner, ProcessRecord


@pytest.fixture
def tmp_logger(tmp_path):
    return Logger(tasks_dir=str(tmp_path / "tasks"))


@pytest.fixture
def spawner(tmp_logger):
    return Spawner(dispatcher=Dispatcher(), logger=tmp_logger)


class TestSpawn:
    @patch("mr1.core.spawner.subprocess.Popen")
    def test_spawn_returns_record(self, mock_popen, spawner):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        record = spawner.spawn(
            agent_type="kazi",
            task_id="task-001",
            prompt="test prompt",
            model="haiku",
            tools=["Read", "Glob"],
        )

        assert isinstance(record, ProcessRecord)
        assert record.pid == 12345
        assert record.agent_type == "kazi"
        assert record.task_id == "task-001"

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_spawn_builds_correct_cmd(self, mock_popen, spawner):
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_popen.return_value = mock_proc

        spawner.spawn(
            agent_type="kazi",
            task_id="task-001",
            prompt="do stuff",
            model="haiku",
            tools=["Read"],
            extra_flags=["--output-format", "json"],
        )

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--model" in cmd
        assert "--allowedTools" in cmd
        assert "--output-format" in cmd

    def test_spawn_denied_by_dispatcher(self, spawner):
        with pytest.raises(PermissionDenied):
            spawner.spawn(
                agent_type="kazi",
                task_id="task-001",
                prompt="test",
                tools=["Agent"],  # Kazi can't use Agent
            )


class TestKill:
    @patch("mr1.core.spawner.subprocess.Popen")
    def test_kill_by_pid(self, mock_popen, spawner):
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.poll.return_value = -15
        mock_popen.return_value = mock_proc

        record = spawner.spawn("kazi", "task-001", "test", tools=["Read"])
        assert spawner.kill_by_pid(100) is True
        mock_proc.terminate.assert_called_once()

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_kill_by_task(self, mock_popen, spawner):
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.poll.return_value = -15
        mock_popen.return_value = mock_proc

        spawner.spawn("kazi", "task-001", "test", tools=["Read"])
        killed = spawner.kill_by_task("task-001")
        assert killed == 1

    def test_kill_nonexistent_pid(self, spawner):
        assert spawner.kill_by_pid(99999) is False

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_kill_all(self, mock_popen, spawner):
        for i in range(3):
            mock_proc = MagicMock()
            mock_proc.pid = 100 + i
            mock_proc.poll.return_value = -15
            mock_popen.return_value = mock_proc
            spawner.spawn("kazi", f"task-{i}", "test", tools=["Read"])

        killed = spawner.kill_all()
        assert killed == 3


class TestStatus:
    @patch("mr1.core.spawner.subprocess.Popen")
    def test_list_active(self, mock_popen, spawner):
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.poll.return_value = None  # Still running
        mock_popen.return_value = mock_proc

        spawner.spawn("kazi", "task-001", "test", tools=["Read"])
        active = spawner.list_active()
        assert len(active) == 1
        assert active[0]["pid"] == 100
        assert active[0]["running"] is True

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_is_alive(self, mock_popen, spawner):
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        spawner.spawn("kazi", "task-001", "test", tools=["Read"])
        assert spawner.is_alive(100) is True
        assert spawner.is_alive(999) is False


class TestGetResultCleanup:
    """Tests for the subprocess cleanup fix in get_result()."""

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_get_result_cleans_up_tracking(self, mock_popen, spawner):
        """After get_result(), the process should be removed from tracking."""
        mock_proc = MagicMock()
        mock_proc.pid = 100
        mock_proc.poll.return_value = 0  # Process finished
        mock_proc.communicate.return_value = (b"output", b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        spawner.spawn("kazi", "task-001", "test", tools=["Read"])

        # Process is tracked.
        assert 100 in spawner._by_pid
        assert "task-001" in spawner._by_task

        # Get the result — this should clean up.
        result = spawner.get_result(100)
        assert result is not None
        assert result["returncode"] == 0
        assert result["stdout"] == "output"

        # Process should no longer be tracked.
        assert 100 not in spawner._by_pid
        assert "task-001" not in spawner._by_task

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_get_result_returns_none_for_running(self, mock_popen, spawner):
        """get_result() returns None and does NOT clean up running processes."""
        mock_proc = MagicMock()
        mock_proc.pid = 200
        mock_proc.poll.return_value = None  # Still running
        mock_popen.return_value = mock_proc

        spawner.spawn("kazi", "task-002", "test", tools=["Read"])
        result = spawner.get_result(200)
        assert result is None

        # Still tracked.
        assert 200 in spawner._by_pid

    @patch("mr1.core.spawner.subprocess.Popen")
    def test_no_accumulation_after_many_spawns(self, mock_popen, spawner):
        """Verify processes don't accumulate when results are collected."""
        for i in range(10):
            mock_proc = MagicMock()
            mock_proc.pid = 1000 + i
            mock_proc.poll.return_value = 0
            mock_proc.communicate.return_value = (b"ok", b"")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            spawner.spawn("kazi", f"task-{i:03d}", "test", tools=["Read"])
            spawner.get_result(1000 + i)

        # All cleaned up.
        assert len(spawner._by_pid) == 0
        assert len(spawner._by_task) == 0
