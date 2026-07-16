"""
CLI for the Phase D partner-behavior QA harness.

    # zero-cost dry run — proves the wiring, not MR1's behavior
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-smoke --planner fake

    # the real run: real MR1 brain, real judge, costs real tokens
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-1 --planner real \\
        --judge-model sonnet

    # only Category B, in parallel
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-b --planner real \\
        --category B --jobs 4

    # a couple of named clusters
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-smoke --planner real \\
        --clusters drone_backend,drone_backend_followup

    # resume an interrupted run, or re-render the report without executing
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-1 --planner real --resume
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-1 --report

    # a judge-side rate limit hit mid-run: rejudge already-recorded
    # transcripts without re-spending any real MR1 conversation turns
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-1 --rejudge

    # throttle judge calls to reduce rate-limit risk on a big run
    python -m tests.behavior_qa --dir soak-runs/behavior-qa-1 --planner real \\
        --jobs 1 --judge-delay 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.behavior_qa.corpus import CLUSTERS
from tests.behavior_qa.driver import RunLayout
from tests.behavior_qa.report import render_report
from tests.behavior_qa.runner import BehaviorQAConfig, rejudge_corpus, run_corpus


def _select_clusters(category: str | None, names: list[str] | None):
    selected = list(CLUSTERS)
    if category:
        selected = [c for c in selected if c.category == category]
    if names:
        wanted = set(names)
        selected = [c for c in selected if c.name in wanted]
        missing = wanted - {c.name for c in selected}
        if missing:
            print(f"warning: unknown cluster(s): {sorted(missing)}", file=sys.stderr)
    return selected


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.behavior_qa", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", required=True, help="Run directory (created if absent).")
    p.add_argument("--planner", choices=["fake", "real"], default="fake")
    p.add_argument("--judge-model", default="sonnet")
    p.add_argument("--drain-ticks", type=int, default=4)
    p.add_argument("--jobs", type=int, default=1, help="Parallel clusters within a wave.")
    p.add_argument("--judge-delay", type=float, default=0.0,
                   help="Seconds to sleep before each judge call (throttle rate-limit risk).")
    p.add_argument("--category", choices=["A", "B"], default=None)
    p.add_argument("--clusters", default=None, help="Comma-separated cluster names.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run.")
    p.add_argument("--rejudge", action="store_true",
                   help="Re-run only the judge against already-recorded transcripts "
                        "(no re-driving MR1). Useful after a judge-side rate limit.")
    p.add_argument("--report", action="store_true",
                    help="Report-only: re-render the report from a persisted result.json.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    layout = RunLayout(Path(args.dir).expanduser().resolve())

    if args.report:
        if not layout.result.exists():
            print(f"no result.json at {layout.result}", file=sys.stderr)
            return 2
        result = json.loads(layout.result.read_text(encoding="utf-8"))
        text = render_report(result)
        layout.report.write_text(text, encoding="utf-8")
        _print_summary(result)
        print(f"\nreport: {layout.report}")
        return 0

    names = [n.strip() for n in args.clusters.split(",")] if args.clusters else None
    selected = _select_clusters(args.category, names)
    if not selected:
        print("No clusters selected.", file=sys.stderr)
        return 2

    if args.rejudge:
        result = rejudge_corpus(
            layout,
            judge_model=args.judge_model,
            clusters=selected,
            judge_delay_s=args.judge_delay,
            progress=lambda m: print(m, flush=True),
        )
        text = render_report(result)
        layout.report.write_text(text, encoding="utf-8")
        _print_summary(result)
        print(f"\nreport: {layout.report}")
        return 0

    config = BehaviorQAConfig(
        planner=args.planner,
        judge_model=args.judge_model,
        drain_ticks=args.drain_ticks,
        jobs=args.jobs,
        judge_delay_s=args.judge_delay,
    )
    result = run_corpus(
        layout, config,
        clusters=selected, resume=args.resume,
        progress=lambda m: print(m, flush=True),
    )
    text = render_report(result)
    layout.report.write_text(text, encoding="utf-8")
    _print_summary(result)
    print(f"\nreport: {layout.report}")
    return 0


def _print_summary(result: dict) -> None:
    rollup = (result.get("rollup") or {}).get("overall") or {}
    counts = result.get("counts") or {}
    print("\n" + "=" * 60)
    print(f"PARTNER BEHAVIOR QA — planner={result.get('planner')} "
          f"judge={result.get('judge_model')}")
    print(f"  clusters={counts.get('clusters')}  judged={counts.get('judged_clusters')}  "
          f"turns={counts.get('turns')}")
    error_clusters = counts.get("error_clusters") or 0
    if error_clusters:
        infra = counts.get("infra_error_clusters") or 0
        print(
            f"  PARTIAL RUN: {error_clusters} cluster(s) excluded from scores "
            f"({infra} infrastructure/rate-limit, {error_clusters - infra} other judge error) "
            f"— {', '.join(result.get('error_cluster_names') or [])}"
        )
        print("  retry with --resume (re-drive + re-judge) or --rejudge (judge-only)")
    if rollup:
        print(f"  mean partner score: {rollup.get('mean_partner_score')}/10")
        print(f"  would continue using: {rollup.get('would_continue_using_pct')}  "
              f"would have done himself: {rollup.get('would_have_done_himself_pct')}")
        print(f"  initiative calibration: {rollup.get('initiative_calibration')}  "
              f"behavior distribution: {rollup.get('behavior_distribution')}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
