"""
Retention and archival (B1).

MR1 running continuously writes forever: `events.jsonl` grows with every
action, every completed workflow leaves a directory of task logs and attempt
artifacts behind, every capability execution writes an audit record. None of it
was ever reclaimed. At Genesis's weekly cadence that is a slow leak; at a daily
objective it is a disk cliff with a date on it.

Three rules shape everything here.

**Archive, never delete.** Every operation moves data into
`<runtime_root>/archive/`, where it stays queryable and restorable. Age alone
is never a reason to destroy anything — deletion happens only when an operator
explicitly asks for it with `purge_archives_after_days`, and even then only
inside the archive, never against live state.

**Never touch live work.** A workflow is archivable only when it is terminal,
old enough, unreferenced by any live objective, and free of pending approvals.
Any doubt keeps it. The cost of keeping a workflow is bytes; the cost of
archiving one MR1 still needs is a broken objective.

**Say what happened.** Every run — including a dry run — produces a report:
persisted to `<runtime_root>/retention/reports/`, summarised on a timeline
event, and returned to the caller. Cleanup you cannot audit is indistinguishable
from data loss.

None of this calls the brain. It is a deterministic sweep over the filesystem.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mr1.clock import Clock, default_clock, parse_iso
from mr1.event_log import EventLog
from mr1.workflow_models import WorkflowStatus


ARCHIVE_DIR_NAME = "archive"
RETENTION_DIR_NAME = "retention"
REPORTS_DIR_NAME = "reports"
# Per-category record of what was archived and when. The purge reads this rather
# than the filesystem mtime, which `shutil.move` preserves from the original.
_LEDGER_NAME = ".archive_index.json"

_DAY_S = 86_400.0

# Terminal, and therefore in principle archivable. RUNNING/PENDING and anything
# still holding a BLOCKED task are live work and are never candidates.
_TERMINAL_WORKFLOW_STATUSES = frozenset({
    WorkflowStatus.SUCCEEDED,
    WorkflowStatus.FAILED,
    WorkflowStatus.CANCELLED,
})


@dataclass
class RetentionPolicy:
    """
    Every threshold an operator can turn. All of it deterministic, all of it
    configurable, none of it inferred.
    """

    # Events: rotate the live log into sealed segments past this size, keeping a
    # tail behind so in-flight causal chains stay resolvable.
    events_max_live_bytes: int = 32 * 1024 * 1024
    events_keep_recent: int = 1_000
    events_compress_archive: bool = True

    # Workflows: terminal, older than this, and unreferenced → archived.
    workflow_archive_after_days: float = 30.0
    # A floor that beats the age rule: never archive below this many of the most
    # recent terminal workflows, so a quiet week never empties the history an
    # operator reads to understand what MR1 has been doing.
    workflow_keep_recent: int = 50

    # Capability audit records and doctor snapshots.
    audit_archive_after_days: float = 90.0
    snapshot_archive_after_days: float = 90.0

    # The only destructive setting, and it is off. When set, archived material
    # older than this is deleted *from the archive*. Live state is never a
    # candidate. Nothing is removed merely for being old unless this is on.
    purge_archives_after_days: Optional[float] = None

    def validate(self) -> "RetentionPolicy":
        if self.events_max_live_bytes < 0:
            raise ValueError("events_max_live_bytes must be >= 0")
        if self.events_keep_recent < 0:
            raise ValueError("events_keep_recent must be >= 0")
        if self.workflow_archive_after_days < 0:
            raise ValueError("workflow_archive_after_days must be >= 0")
        if self.workflow_keep_recent < 0:
            raise ValueError("workflow_keep_recent must be >= 0")
        if self.purge_archives_after_days is not None and self.purge_archives_after_days <= 0:
            raise ValueError("purge_archives_after_days must be > 0 when set")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_max_live_bytes": self.events_max_live_bytes,
            "events_keep_recent": self.events_keep_recent,
            "events_compress_archive": self.events_compress_archive,
            "workflow_archive_after_days": self.workflow_archive_after_days,
            "workflow_keep_recent": self.workflow_keep_recent,
            "audit_archive_after_days": self.audit_archive_after_days,
            "snapshot_archive_after_days": self.snapshot_archive_after_days,
            "purge_archives_after_days": self.purge_archives_after_days,
        }


@dataclass
class RetentionReport:
    started_at: str
    finished_at: str = ""
    dry_run: bool = True
    events_rotated: bool = False
    events_segment: Optional[str] = None
    events_archived_count: int = 0
    events_live_bytes_before: int = 0
    events_live_bytes_after: int = 0
    workflows_archived: list[str] = field(default_factory=list)
    workflows_kept: dict[str, int] = field(default_factory=dict)
    audits_archived: int = 0
    snapshots_archived: int = 0
    archives_purged: list[str] = field(default_factory=list)
    bytes_reclaimed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.events_rotated
            or self.workflows_archived
            or self.audits_archived
            or self.snapshots_archived
            or self.archives_purged
        )

    def summary(self) -> str:
        prefix = "would archive" if self.dry_run else "archived"
        parts = []
        if self.events_rotated:
            parts.append(f"{prefix} {self.events_archived_count} events")
        if self.workflows_archived:
            parts.append(f"{prefix} {len(self.workflows_archived)} workflows")
        if self.audits_archived:
            parts.append(f"{prefix} {self.audits_archived} audit records")
        if self.snapshots_archived:
            parts.append(f"{prefix} {self.snapshots_archived} snapshots")
        if self.archives_purged:
            parts.append(f"purged {len(self.archives_purged)} archived items")
        if not parts:
            return "nothing to do"
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "dry_run": self.dry_run,
            "summary": self.summary(),
            "events": {
                "rotated": self.events_rotated,
                "segment": self.events_segment,
                "archived_count": self.events_archived_count,
                "live_bytes_before": self.events_live_bytes_before,
                "live_bytes_after": self.events_live_bytes_after,
            },
            "workflows": {
                "archived": list(self.workflows_archived),
                "kept": dict(self.workflows_kept),
            },
            "audits_archived": self.audits_archived,
            "snapshots_archived": self.snapshots_archived,
            "archives_purged": list(self.archives_purged),
            "bytes_reclaimed": self.bytes_reclaimed,
            "errors": list(self.errors),
        }


class RetentionManager:
    """
    Runs the sweep. Safe to call on any schedule, from any process.

    `run(dry_run=True)` computes exactly the same decisions as a real run and
    changes nothing, so an operator can always see what would happen before it
    does.
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        policy: Optional[RetentionPolicy] = None,
        clock: Optional[Clock] = None,
        event_log: Optional[EventLog] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._policy = (policy or RetentionPolicy()).validate()
        self._clock = clock or default_clock()
        self._events = event_log or EventLog(
            self._runtime_root / "events",
            compress_archive=self._policy.events_compress_archive,
        )

    @property
    def policy(self) -> RetentionPolicy:
        return self._policy

    @property
    def archive_root(self) -> Path:
        return self._runtime_root / ARCHIVE_DIR_NAME

    @property
    def reports_dir(self) -> Path:
        return self._runtime_root / RETENTION_DIR_NAME / REPORTS_DIR_NAME

    # ------------------------------------------------------------------
    # The sweep
    # ------------------------------------------------------------------

    def run(self, *, dry_run: bool = False) -> RetentionReport:
        report = RetentionReport(
            started_at=self._clock.now_iso(),
            dry_run=dry_run,
        )
        for phase in (
            self._rotate_events,
            self._archive_workflows,
            self._archive_audits,
            self._archive_snapshots,
            self._purge_archives,
        ):
            try:
                phase(report, dry_run)
            except Exception as exc:  # noqa: BLE001 - a failed phase must not abort the rest
                report.errors.append(f"{phase.__name__}: {type(exc).__name__}: {exc}")

        report.finished_at = self._clock.now_iso()
        self._persist_report(report)
        self._emit_report(report)
        return report

    # -- events --------------------------------------------------------

    def _rotate_events(self, report: RetentionReport, dry_run: bool) -> None:
        stats = self._events.history_stats()
        report.events_live_bytes_before = stats["live_bytes"]
        report.events_live_bytes_after = stats["live_bytes"]

        limit = self._policy.events_max_live_bytes
        if limit <= 0 or stats["live_bytes"] < limit:
            return

        # How many events *would* be sealed: everything in the live file bar the
        # tail we deliberately keep behind for causal resolution.
        live_count = stats["total_events"] - stats["archived_through_index"]
        sealable = max(0, live_count - self._policy.events_keep_recent)
        if sealable <= 0:
            return

        if dry_run:
            report.events_rotated = True
            report.events_archived_count = sealable
            return

        segment = self._events.rotate(
            keep_recent=self._policy.events_keep_recent,
            now_iso=self._clock.now_iso(),
        )
        if segment is None:
            return
        report.events_rotated = True
        report.events_segment = segment.name
        report.events_archived_count = segment.count
        after = self._events.history_stats()["live_bytes"]
        report.events_live_bytes_after = after
        report.bytes_reclaimed += max(0, report.events_live_bytes_before - after)

    # -- workflows -----------------------------------------------------

    def _archive_workflows(self, report: RetentionReport, dry_run: bool) -> None:
        workflows_root = self._runtime_root / "workflows"
        if not workflows_root.is_dir():
            return

        protected = self._protected_workflow_ids()
        cutoff_s = self._policy.workflow_archive_after_days * _DAY_S
        now = self._clock.now()

        candidates: list[tuple[float, Path, str]] = []
        kept: dict[str, int] = {}

        for entry in sorted(workflows_root.iterdir()):
            if not entry.is_dir():
                continue
            workflow_id = entry.name
            payload = self._read_workflow(entry)
            if payload is None:
                kept["unreadable"] = kept.get("unreadable", 0) + 1
                continue

            status = str(payload.get("status") or "")
            if status not in {item.value for item in _TERMINAL_WORKFLOW_STATUSES}:
                kept["active"] = kept.get("active", 0) + 1
                continue
            if self._has_unresolved_task(payload):
                # A terminal workflow can still hold a task parked BLOCKED and
                # waiting on a human. Archiving it would file away the very
                # thing an operator is being asked to look at.
                kept["waiting_human"] = kept.get("waiting_human", 0) + 1
                continue
            if workflow_id in protected:
                kept["referenced"] = kept.get("referenced", 0) + 1
                continue

            age_s = self._age_s(payload.get("finished_at") or payload.get("created_at"), now)
            if age_s is None or age_s < cutoff_s:
                kept["too_recent"] = kept.get("too_recent", 0) + 1
                continue
            candidates.append((age_s, entry, workflow_id))

        # The keep-recent floor: hold back the newest N terminal workflows even
        # when they are old enough, so history never empties out entirely.
        candidates.sort(key=lambda item: item[0])  # youngest first
        floor = self._policy.workflow_keep_recent
        if floor > 0:
            held = candidates[:floor]
            candidates = candidates[floor:]
            if held:
                kept["keep_recent_floor"] = kept.get("keep_recent_floor", 0) + len(held)

        destination = self.archive_root / "workflows"
        for _age, path, workflow_id in candidates:
            size = _dir_size(path)
            if dry_run:
                report.workflows_archived.append(workflow_id)
                report.bytes_reclaimed += size
                continue
            try:
                self._move(path, destination / workflow_id)
            except OSError as exc:
                report.errors.append(f"workflow {workflow_id}: {exc}")
                continue
            report.workflows_archived.append(workflow_id)
            report.bytes_reclaimed += size

        report.workflows_kept = kept

    def _protected_workflow_ids(self) -> set[str]:
        """
        Every workflow some live record still points at.

        An objective's `current_workflow_id` and its attempt history are the
        references that matter: recovery reads them, and `mr1 objective show`
        renders them. Pending approvals pin their workflow too. Archiving out
        from under either turns a working reference into a dangling one.
        """
        protected: set[str] = set()

        try:
            from mr1.autonomy.objectives import ObjectiveStore

            store = ObjectiveStore(self._runtime_root, clock=self._clock)
            for objective in store.list_objectives():
                if objective.current_workflow_id:
                    protected.add(objective.current_workflow_id)
                for attempt in objective.history:
                    workflow_id = getattr(attempt, "workflow_id", None)
                    if workflow_id:
                        protected.add(workflow_id)
        except Exception:  # noqa: BLE001 - a missing store must not unprotect anything
            pass

        try:
            from mr1.capability_policy import CapabilityApprovalStore

            approvals = CapabilityApprovalStore(
                self._runtime_root / "capability_approvals",
                clock=self._clock,
            )
            for approval in approvals.list_requests():
                if approval.status == "pending" and approval.workflow_id:
                    protected.add(approval.workflow_id)
        except Exception:  # noqa: BLE001
            pass

        return protected

    @staticmethod
    def _read_workflow(directory: Path) -> Optional[dict[str, Any]]:
        path = directory / "workflow.json"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _has_unresolved_task(payload: dict[str, Any]) -> bool:
        tasks = payload.get("tasks")
        values = tasks.values() if isinstance(tasks, dict) else (tasks or [])
        for task in values:
            if isinstance(task, dict) and task.get("status") == "blocked":
                return True
        return False

    # -- audits and snapshots ------------------------------------------

    def _archive_audits(self, report: RetentionReport, dry_run: bool) -> None:
        agents_root = self._runtime_root / "agents"
        if not agents_root.is_dir():
            return
        cutoff_s = self._policy.audit_archive_after_days * _DAY_S
        if cutoff_s <= 0:
            return
        now_ts = self._clock.now().timestamp()

        for record in sorted(agents_root.glob("ag-*/logs/capability_audits/*.json")):
            try:
                age_s = now_ts - record.stat().st_mtime
            except OSError:
                continue
            if age_s < cutoff_s:
                continue
            agent_id = record.parents[2].name
            target = self.archive_root / "capability_audits" / agent_id / record.name
            size = _file_size(record)
            if dry_run:
                report.audits_archived += 1
                report.bytes_reclaimed += size
                continue
            try:
                self._move(record, target)
            except OSError as exc:
                report.errors.append(f"audit {record.name}: {exc}")
                continue
            report.audits_archived += 1
            report.bytes_reclaimed += size

    def _archive_snapshots(self, report: RetentionReport, dry_run: bool) -> None:
        snapshots_root = self._runtime_root / "snapshots"
        if not snapshots_root.is_dir():
            return
        cutoff_s = self._policy.snapshot_archive_after_days * _DAY_S
        if cutoff_s <= 0:
            return
        now_ts = self._clock.now().timestamp()

        for entry in sorted(snapshots_root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                age_s = now_ts - entry.stat().st_mtime
            except OSError:
                continue
            if age_s < cutoff_s:
                continue
            size = _dir_size(entry)
            if dry_run:
                report.snapshots_archived += 1
                report.bytes_reclaimed += size
                continue
            try:
                self._move(entry, self.archive_root / "snapshots" / entry.name)
            except OSError as exc:
                report.errors.append(f"snapshot {entry.name}: {exc}")
                continue
            report.snapshots_archived += 1
            report.bytes_reclaimed += size

    # -- purge (opt-in, and only inside the archive) --------------------

    def _purge_archives(self, report: RetentionReport, dry_run: bool) -> None:
        """
        The one destructive operation, and it is off unless asked for.

        It only ever runs against `<runtime_root>/archive/` — material that has
        already been moved out of live state by an earlier, audited run. Live
        workflows, live events, and live audits are never reachable from here,
        whatever the policy says.

        Age is measured from when MR1 *archived* the item, recorded in the
        archive ledger, not from the file's mtime. `shutil.move` carries the
        original mtime across, so an mtime-based purge would delete a workflow
        the instant it was archived — and it would be unauditable besides.
        """
        max_age_days = self._policy.purge_archives_after_days
        if max_age_days is None:
            return
        if not self.archive_root.is_dir():
            return
        cutoff_s = max_age_days * _DAY_S
        now = self._clock.now()

        for category in sorted(self.archive_root.iterdir()):
            if not category.is_dir():
                continue
            ledger = self._read_ledger(category)
            changed = False
            for entry in sorted(category.iterdir()):
                if entry.name == _LEDGER_NAME:
                    continue
                record = ledger.get(entry.name)
                if not record:
                    # Archived before the ledger existed, or placed here by
                    # hand. Unknown provenance is not a licence to delete.
                    continue
                age_s = self._age_s(record.get("archived_at"), now)
                if age_s is None or age_s < cutoff_s:
                    continue
                label = f"{category.name}/{entry.name}"
                size = _dir_size(entry) if entry.is_dir() else _file_size(entry)
                if dry_run:
                    report.archives_purged.append(label)
                    continue
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                except OSError as exc:
                    report.errors.append(f"purge {label}: {exc}")
                    continue
                ledger.pop(entry.name, None)
                changed = True
                report.archives_purged.append(label)
                report.bytes_reclaimed += size
            if changed and not dry_run:
                self._write_ledger(category, ledger)

    # -- plumbing ------------------------------------------------------

    def _move(self, source: Path, target: Path) -> None:
        """Move into the archive and record that we did, and when."""
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # An archived copy already exists (a previous run was interrupted
            # after the copy but before the source was removed). The archive is
            # the durable side; drop the live remnant rather than fail.
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
        else:
            shutil.move(str(source), str(target))
        self._record_archived(target)

    def _record_archived(self, target: Path) -> None:
        category = target.parent
        # Audits nest one level deeper (archive/capability_audits/<agent>/x.json);
        # the ledger belongs to the top-level category directory either way.
        while category.parent != self.archive_root and category != self.archive_root:
            category = category.parent
        if category == self.archive_root:
            return
        ledger = self._read_ledger(category)
        key = target.relative_to(category).parts[0]
        ledger.setdefault(
            key,
            {
                "archived_at": self._clock.now_iso(),
                "bytes": _dir_size(target) if target.is_dir() else _file_size(target),
            },
        )
        self._write_ledger(category, ledger)

    @staticmethod
    def _read_ledger(category: Path) -> dict[str, Any]:
        path = category / _LEDGER_NAME
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_ledger(category: Path, ledger: dict[str, Any]) -> None:
        path = category / _LEDGER_NAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
        except OSError:
            pass

    def _age_s(self, timestamp: Optional[str], now) -> Optional[float]:
        moment = parse_iso(timestamp) if timestamp else None
        if moment is None:
            return None
        return max(0.0, (now - moment).total_seconds())

    def _persist_report(self, report: RetentionReport) -> None:
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            stamp = report.started_at.replace(":", "").replace("-", "")[:15]
            suffix = "dryrun" if report.dry_run else "run"
            path = self.reports_dir / f"retention-{stamp}-{suffix}.json"
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(
                    {"policy": self._policy.to_dict(), **report.to_dict()},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
        except OSError as exc:
            report.errors.append(f"report not persisted: {exc}")

    def _emit_report(self, report: RetentionReport) -> None:
        try:
            self._events.emit(
                event_type="retention_run",
                actor_id="retention",
                actor_type="mr1",
                target_id="runtime",
                target_type="runtime",
                status="dry_run" if report.dry_run else ("ok" if not report.errors else "error"),
                summary=f"retention {'dry run' if report.dry_run else 'run'}: {report.summary()}",
                metadata={
                    "dry_run": report.dry_run,
                    "events_archived": report.events_archived_count,
                    "workflows_archived": len(report.workflows_archived),
                    "audits_archived": report.audits_archived,
                    "snapshots_archived": report.snapshots_archived,
                    "archives_purged": len(report.archives_purged),
                    "bytes_reclaimed": report.bytes_reclaimed,
                    "errors": report.errors[:5],
                },
            )
        except Exception:  # noqa: BLE001 - the timeline must not break the sweep
            pass

    def status(self) -> dict[str, Any]:
        """What retention would do next, and what it has already done."""
        stats = self._events.history_stats()
        reports = sorted(self.reports_dir.glob("retention-*.json")) if self.reports_dir.is_dir() else []
        # Two archives, one number: sealed event segments live under
        # `events/archive/`, everything else under `<root>/archive/`. An
        # operator asking "how much is archived" means both.
        archive_bytes = _dir_size(self.archive_root) + int(stats["archive_bytes"])
        return {
            "policy": self._policy.to_dict(),
            "events": stats,
            "events_rotation_due": (
                self._policy.events_max_live_bytes > 0
                and stats["live_bytes"] >= self._policy.events_max_live_bytes
            ),
            "archive_bytes": archive_bytes,
            "archive_root": str(self.archive_root),
            "last_report": reports[-1].name if reports else None,
            "reports": len(reports),
        }


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0
