from __future__ import annotations

import json

from tests.runtime_qa import runner


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def test_runner_uses_process_pool_for_parallel_jobs(tmp_path, monkeypatch):
    recorded: dict[str, object] = {}

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers, mp_context):
            recorded["max_workers"] = max_workers
            recorded["mp_context"] = mp_context
            recorded["submitted"] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            recorded["submitted"].append((fn, args, kwargs))
            return _FakeFuture(fn(*args, **kwargs))

    def fake_run_one_named(scenario_name: str, *, verbose: bool = False) -> dict[str, object]:
        return {
            "scenario": scenario_name,
            "category": "identity",
            "description": "fake",
            "turn_count": 1,
            "elapsed_s": 0.01,
            "findings": [],
            "session_error": None,
        }

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(runner, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(runner, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(runner, "_run_one_named", fake_run_one_named)

    rc = runner.main(["identity_reserved_word_title_all", "--jobs=4"])

    assert rc == 0
    assert recorded["max_workers"] == 4
    submitted = recorded["submitted"]
    assert len(submitted) == 1
    fn, args, kwargs = submitted[0]
    assert fn is fake_run_one_named
    assert args == ("identity_reserved_word_title_all",)
    assert kwargs == {"verbose": False}

    summary = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_scenarios"] == 1
    assert summary["crashed_scenarios"] == []
