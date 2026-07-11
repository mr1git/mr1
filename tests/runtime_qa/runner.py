"""Scenario runner for runtime QA.

Usage:
    python -m tests.runtime_qa.runner                  # run all scenarios
    python -m tests.runtime_qa.runner nl_hello_direct  # run one
    python -m tests.runtime_qa.runner --category=safety
    python -m tests.runtime_qa.runner --jobs=4         # concurrency
    python -m tests.runtime_qa.runner --list           # list scenarios

Each scenario runs in its own isolated MR1 runtime (temp dir).
Results: tests/runtime_qa/results/<name>.jsonl  +  <name>.meta.json
A top-level tests/runtime_qa/results/summary.json is written at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mr1.runtime_test_cli import (
    RuntimePaths,
    RuntimeTestSession,
    _patched_runtime_paths,
)

from tests.runtime_qa.scenarios import SCENARIOS, Scenario

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _make_paths(runtime_root: Path) -> RuntimePaths:
    return RuntimePaths(
        isolated=True,
        runtime_root=runtime_root,
        workflow_root=runtime_root / "workflows",
        state_path=runtime_root / "active" / "mr1_state.json",
        context_path=runtime_root / "active" / "mr1_context.md",
        dumps_root=runtime_root / "dumps",
        rag_root=runtime_root / "rag",
    )


def _run_one(scenario: Scenario, *, verbose: bool = False) -> dict[str, Any]:
    start = time.time()
    payloads: list[dict[str, Any]] = []
    session_err: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix=f"mr1-qa-{scenario.name}-") as tmp:
        paths = _make_paths(Path(tmp))
        try:
            with _patched_runtime_paths(paths):
                session = RuntimeTestSession(paths)
                try:
                    for i, turn in enumerate(scenario.turns, start=1):
                        if verbose:
                            print(f"  [{scenario.name}] turn {i}: {turn!r}", flush=True)
                        payload = session.handle_input(turn, request_index=i)
                        payloads.append(payload)
                finally:
                    with contextlib.suppress(Exception):
                        session.shutdown()
        except Exception as exc:
            session_err = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "trace": traceback.format_exc(),
            }

    findings: list[tuple[str, str, str]] = []
    for check in scenario.checks:
        try:
            findings.extend(check(payloads))
        except Exception as exc:
            findings.append(
                (
                    "high",
                    f"Check {check.__name__} crashed",
                    f"{exc.__class__.__name__}: {exc}",
                )
            )

    elapsed = time.time() - start
    out_jsonl = RESULTS_DIR / f"{scenario.name}.jsonl"
    out_meta = RESULTS_DIR / f"{scenario.name}.meta.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in payloads:
            f.write(json.dumps(p, sort_keys=True) + "\n")
    meta = {
        "scenario": scenario.name,
        "category": scenario.category,
        "description": scenario.description,
        "turn_count": len(scenario.turns),
        "elapsed_s": round(elapsed, 2),
        "findings": [
            {"severity": s, "title": t, "detail": d} for (s, t, d) in findings
        ],
        "session_error": session_err,
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def _run_one_named(scenario_name: str, *, verbose: bool = False) -> dict[str, Any]:
    selected = _select_scenarios([scenario_name], None)
    if not selected:
        raise ValueError(f"unknown scenario: {scenario_name}")
    return _run_one(selected[0], verbose=verbose)


def _summarize(metas: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    crashed: list[str] = []
    for m in metas:
        if m.get("session_error"):
            crashed.append(m["scenario"])
        for f in m.get("findings", []):
            sev = f.get("severity", "?")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        cat = m.get("category", "?")
        cat_bucket = by_category.setdefault(cat, {"scenarios": 0, "findings": 0})
        cat_bucket["scenarios"] += 1
        cat_bucket["findings"] += len(m.get("findings", []))
    return {
        "total_scenarios": len(metas),
        "crashed_scenarios": crashed,
        "findings_by_severity": by_severity,
        "by_category": by_category,
    }


def _select_scenarios(
    names: list[str] | None,
    category: str | None,
) -> list[Scenario]:
    selected = SCENARIOS
    if category:
        selected = [s for s in selected if s.category == category]
    if names:
        wanted = set(names)
        selected = [s for s in selected if s.name in wanted]
        missing = wanted - {s.name for s in selected}
        if missing:
            print(f"warning: unknown scenarios: {sorted(missing)}", file=sys.stderr)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tests.runtime_qa.runner")
    parser.add_argument("names", nargs="*", help="Specific scenario name(s).")
    parser.add_argument("--category", default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel scenarios.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    selected = _select_scenarios(args.names or None, args.category)

    if args.list:
        for s in selected:
            print(f"{s.category:18s} {s.name:36s} turns={len(s.turns):2d}  {s.description}")
        return 0

    if not selected:
        print("No scenarios selected.", file=sys.stderr)
        return 2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(selected)} scenarios with {args.jobs} workers...", flush=True)
    started = time.time()
    metas: list[dict[str, Any]] = []

    if args.jobs <= 1:
        for s in selected:
            print(f"  → {s.name} ({s.category}) ...", flush=True)
            m = _run_one(s, verbose=args.verbose)
            metas.append(m)
            print(
                f"    done in {m['elapsed_s']}s, "
                f"findings={len(m['findings'])}, "
                f"crash={'Y' if m['session_error'] else 'n'}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=multiprocessing.get_context("spawn"),
        ) as ex:
            futures = {
                ex.submit(_run_one_named, s.name, verbose=False): s
                for s in selected
            }
            for fut in as_completed(futures):
                scen = futures[fut]
                try:
                    m = fut.result()
                except Exception as exc:
                    m = {
                        "scenario": scen.name,
                        "category": scen.category,
                        "description": scen.description,
                        "turn_count": len(scen.turns),
                        "elapsed_s": -1,
                        "findings": [],
                        "session_error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "trace": traceback.format_exc(),
                        },
                    }
                metas.append(m)
                print(
                    f"  done: {m['scenario']:36s} {m['elapsed_s']}s "
                    f"findings={len(m['findings'])}  "
                    f"crash={'Y' if m['session_error'] else 'n'}",
                    flush=True,
                )

    summary = _summarize(metas)
    summary["total_elapsed_s"] = round(time.time() - started, 2)
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
