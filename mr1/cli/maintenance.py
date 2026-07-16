"""CLI handlers for retention and archival (B1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mr1.autonomy.retention import RetentionManager, RetentionPolicy


def _runtime_root(store) -> Path:
    return Path(store.root).parent


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _policy_from_args(args) -> RetentionPolicy:
    """Every threshold is overridable per-invocation; defaults are the policy."""
    policy = RetentionPolicy()
    for attr in (
        "events_max_live_bytes",
        "events_keep_recent",
        "workflow_archive_after_days",
        "workflow_keep_recent",
        "audit_archive_after_days",
        "snapshot_archive_after_days",
        "purge_archives_after_days",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(policy, attr, value)
    return policy.validate()


def _manager(args, store) -> RetentionManager:
    return RetentionManager(_runtime_root(store), policy=_policy_from_args(args))


def _cmd_maintenance_run(args, store, caller_agent_id, scoped_agent_store) -> int:
    manager = _manager(args, store)
    dry_run = bool(getattr(args, "dry_run", False))
    report = manager.run(dry_run=dry_run)

    if getattr(args, "json", False):
        _print_json(report.to_dict())
        return 2 if report.errors else 0

    header = "retention dry run (nothing was changed)" if dry_run else "retention run"
    print(f"{header}: {report.summary()}")
    if report.events_rotated:
        print(
            f"  events:    {report.events_archived_count} archived"
            + (f" -> {report.events_segment}" if report.events_segment else "")
        )
    if report.workflows_archived:
        print(f"  workflows: {len(report.workflows_archived)} archived")
        for workflow_id in report.workflows_archived[:10]:
            print(f"               {workflow_id}")
        if len(report.workflows_archived) > 10:
            print(f"               ... and {len(report.workflows_archived) - 10} more")
    if report.workflows_kept:
        kept = ", ".join(f"{reason}={count}" for reason, count in sorted(report.workflows_kept.items()))
        print(f"  kept:      {kept}")
    if report.audits_archived:
        print(f"  audits:    {report.audits_archived} archived")
    if report.snapshots_archived:
        print(f"  snapshots: {report.snapshots_archived} archived")
    if report.archives_purged:
        print(f"  purged:    {len(report.archives_purged)} archived items DELETED")
    print(f"  bytes:     {report.bytes_reclaimed:,} {'reclaimable' if dry_run else 'reclaimed'}")
    if not dry_run:
        print(f"  archive:   {manager.archive_root}")
    for error in report.errors:
        print(f"  error:     {error}")
    return 2 if report.errors else 0


def _cmd_maintenance_status(args, store, caller_agent_id, scoped_agent_store) -> int:
    manager = _manager(args, store)
    payload = manager.status()

    if getattr(args, "json", False):
        _print_json(payload)
        return 0

    events = payload["events"]
    print(f"events (total):      {events['total_events']:,}")
    print(f"  live file:         {events['live_bytes']:,} bytes")
    print(
        f"  archive:           {events['archive_bytes']:,} bytes"
        f" in {events['archive_segments']} segment(s),"
        f" {events['archived_events']:,} events"
    )
    print(
        "  cache:             "
        f"{events['cached_events']:,}/{events['cache_limit']:,}"
        + ("  (complete)" if events["cache_is_complete"] else "  (window only — history reads hit disk)")
    )
    print(f"rotation due:        {'yes' if payload['events_rotation_due'] else 'no'}")
    print(f"archive on disk:     {payload['archive_bytes']:,} bytes")
    print(f"retention reports:   {payload['reports']} (last: {payload['last_report'] or 'none'})")
    return 0
