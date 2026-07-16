"""
CLI for the hierarchical autonomy soak.

    # quick validation (real everything except the brain), a few minutes
    python -m tests.soak.hierarchical --planner fake --dir soak-runs/hier-smoke

    # a brief real-planner smoke: one arc against real claude
    python -m tests.soak.hierarchical --planner real --max-turns 4 \
        --dir soak-runs/hier-real-smoke

    # the overnight run: real claude, realistic pacing, fixed wall-clock
    python -m tests.soak.hierarchical --planner real --duration 10h \
        --turn-interval 45s --sample-interval 60s --dir soak-runs/hier-overnight

    # resume an interrupted run, or re-analyse one without executing
    python -m tests.soak.hierarchical --resume --dir soak-runs/hier-overnight
    python -m tests.soak.hierarchical --report --dir soak-runs/hier-overnight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tests.soak.hierarchical.driver import RunLayout
from tests.soak.hierarchical.report import write_report
from tests.soak.hierarchical.soak import SoakConfig, analyze, run_soak


def _duration(text: str) -> float:
    text = text.strip().lower()
    if not text:
        return 0.0
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tests.soak.hierarchical", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="Run directory (created if absent).")
    p.add_argument("--planner", choices=["fake", "real"], default="fake")
    p.add_argument("--duration", type=_duration, default=0.0,
                   help="Wall-clock budget (e.g. 10h). 0 => run the corpus once and stop.")
    p.add_argument("--turn-interval", type=_duration, default=0.0,
                   help="Pause between turns (e.g. 45s) for realistic pacing.")
    p.add_argument("--sample-interval", type=_duration, default=0.0,
                   help="Min wall gap between idle-phase samples.")
    p.add_argument("--drain-ticks", type=int, default=4,
                   help="Scheduler ticks after each turn to execute in-flight work.")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Cap the number of scripted turns (for a real-planner smoke).")
    p.add_argument("--no-restart", action="store_true",
                   help="Skip the mid-conversation restart.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run.")
    p.add_argument("--report", action="store_true",
                   help="Report-only: re-analyse a run without executing it.")
    p.add_argument("--seed", type=int, default=7)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    layout = RunLayout(Path(args.dir).expanduser().resolve())

    if args.report:
        result = analyze(layout)
        path = write_report(layout, result)
        _print_summary(result)
        print(f"\nreport: {path}")
        return 0 if result.get("passed") else 1

    config = SoakConfig(
        planner=args.planner,
        duration_s=args.duration,
        turn_interval_s=args.turn_interval,
        sample_interval_s=args.sample_interval,
        drain_ticks=args.drain_ticks,
        do_restart=not args.no_restart,
        seed=args.seed,
    )
    result = run_soak(
        layout, config,
        resume=args.resume,
        progress=lambda m: print(m, flush=True),
        max_turns=args.max_turns,
    )
    path = write_report(layout, result)
    _print_summary(result)
    print(f"\nreport: {path}")
    return 0 if result.get("passed") else 1


def _print_summary(result: dict) -> None:
    verdict = "PASSED" if result.get("passed") else ("PARTIAL" if result.get("partial") else "FAILED")
    counts = result.get("counts", {})
    findings = result.get("findings", {})
    total = sum(len(v) for v in findings.values())
    highs = sum(1 for group in findings.values() for f in group if f and f[0] == "high")
    print("\n" + "=" * 60)
    print(f"HIERARCHICAL SOAK — {verdict}   ({result.get('planner')} planner)")
    print(f"  turns={counts.get('turns')}  agents={counts.get('agents')}  "
          f"messages={counts.get('messages')}  workflows={counts.get('workflows')}")
    print(f"  findings: {total} total, {highs} high-severity")
    rollup = result.get("rollup") or {}
    if rollup:
        print(f"  RSS {rollup.get('rss_start_mb')}→{rollup.get('rss_end_mb')} MB  "
              f"FDs {rollup.get('fd_start')}→{rollup.get('fd_end')}  "
              f"idle-brain-calls {rollup.get('idle_brain_calls')}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
