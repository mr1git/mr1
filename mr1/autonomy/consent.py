"""
Objective-scoped consent grants — standing authority, safely bounded.

A one-off human approval is keyed to one exact invocation in one exact
workflow: its ID is a hash over (actor, capability, args, scope, workflow_id,
task_id). That is the *correct* semantic for "yes, do this one thing" — and it
is why it cannot support unattended operation. A recurring objective mints a
new workflow_id on every run, so it hashes to a new approval every time and
asks a human again, forever.

A consent grant is a different mechanism with different semantics: coarser,
predicate-matched, TTL'd, revocable, and attached to a *mission* rather than to
an invocation. It does not modify or weaken one-off approval matching; it is a
second, parallel override path with its own rules:

  * objective-scoped — a grant authorizes exactly one objective, never another
  * TTL required — there are no immortal grants
  * revocable — instantly, and `mr1 halt` revokes every one of them
  * risk-bounded — it cannot authorize above its own `max_risk`
  * scope-bound — it cannot widen its own scope roots
  * predicate-matched — the args must satisfy what the human actually allowed
  * root-granted — only the human/root may grant risk-1.0 authority
  * audited — every grant-authorized execution carries its grant_id

Fail-closed everywhere: an unknown predicate operator, a missing arg, an
unparseable expiry, or a malformed grant file all mean "no match".
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from mr1.capability_policy import CapabilityMetadata, CapabilityRequest, normalize_path
from mr1.clock import Clock, default_clock, parse_iso
from mr1.event_log import EventLog
from mr1.scoped_agents import MAX_AUTONOMOUS_CLEARANCE, PersistentAgentStore


CONSENT_DIR_NAME = "consent_grants"
GRANTEE_KIND_OBJECTIVE = "objective"

_ALLOWED_PREDICATE_OPS = frozenset({"equals", "one_of", "prefix", "regex"})


class ConsentGrantError(ValueError):
    """Raised when a grant is malformed or the grantor lacks the authority."""


def new_grant_id() -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"grant-{stamp}-{uuid.uuid4().hex[:6]}"


def _predicate_text(value: Any) -> Optional[str]:
    """The single string a predicate is evaluated against, or None if unmatchable."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        # argv-style list args are matched against the joined command line, so
        # `--allow '^pytest'` does what an operator expects for ["pytest", "-q"].
        return " ".join(value)
    return None


def _predicate_matches(value: Any, spec: Any) -> bool:
    if not isinstance(spec, dict) or not spec:
        return False
    text = _predicate_text(value)
    if text is None:
        return False
    for op, expected in spec.items():
        if op not in _ALLOWED_PREDICATE_OPS:
            return False
        if op == "equals":
            if text != str(expected):
                return False
        elif op == "one_of":
            if not isinstance(expected, list):
                return False
            if text not in [str(item) for item in expected]:
                return False
        elif op == "prefix":
            if not text.startswith(str(expected)):
                return False
        elif op == "regex":
            try:
                if re.search(str(expected), text) is None:
                    return False
            except re.error:
                return False
    return True


@dataclass(frozen=True)
class ConsentGrant:
    grant_id: str
    grantee_id: str
    capability_name: str
    scope_roots: list[str]
    max_risk: float
    granted_by: str
    granted_at: str
    expires_at: str
    grantee_kind: str = GRANTEE_KIND_OBJECTIVE
    arg_predicate: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    revoked_at: Optional[str] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None
    use_count: int = 0
    last_used_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.grantee_kind != GRANTEE_KIND_OBJECTIVE:
            raise ConsentGrantError(f"unsupported grantee_kind '{self.grantee_kind}'")
        if not self.grantee_id:
            raise ConsentGrantError("a consent grant must name the objective it authorizes")
        if not self.capability_name:
            raise ConsentGrantError("a consent grant must name a capability")
        if not 0.0 <= float(self.max_risk) <= 1.0:
            raise ConsentGrantError("max_risk must be between 0.0 and 1.0")
        if not self.expires_at:
            raise ConsentGrantError("a consent grant must expire; there are no immortal grants")
        if parse_iso(self.expires_at) is None:
            raise ConsentGrantError(f"unparseable expires_at: {self.expires_at!r}")
        if not self.scope_roots:
            raise ConsentGrantError("a consent grant must bind at least one scope root")

    # -- lifecycle ----------------------------------------------------

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    def is_expired(self, now: datetime) -> bool:
        deadline = parse_iso(self.expires_at)
        if deadline is None:
            return True  # unparseable expiry is treated as expired
        return now >= deadline

    def is_active(self, now: datetime) -> bool:
        return not self.revoked and not self.is_expired(now)

    def status(self, now: datetime) -> str:
        if self.revoked:
            return "revoked"
        if self.is_expired(now):
            return "expired"
        return "active"

    # -- the predicate ------------------------------------------------

    def match_failure(
        self,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        *,
        now: datetime,
    ) -> Optional[str]:
        """
        None when this grant authorizes the request; otherwise why it does not.

        The reason string is written to the audit record, so a denied autonomous
        execution explains itself without a human having to reconstruct it.
        """
        if self.revoked:
            return "grant_revoked"
        if self.is_expired(now):
            return "grant_expired"
        objective_id = getattr(request, "objective_id", None)
        if not objective_id:
            return "request_has_no_objective"
        if objective_id != self.grantee_id:
            # A grant authorizes one mission. It can never be borrowed by another.
            return "grant_belongs_to_another_objective"
        if request.capability_name != self.capability_name:
            return "capability_mismatch"
        if float(metadata.risk_score) > float(self.max_risk):
            return "risk_exceeds_grant_max_risk"
        scope_failure = self._scope_failure(request, metadata)
        if scope_failure is not None:
            return scope_failure
        return self._predicate_failure(request)

    def _scope_failure(
        self,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
    ) -> Optional[str]:
        """Every path argument must fall inside the roots the human granted."""
        roots = [normalize_path(root) for root in self.scope_roots]
        for field_name in metadata.path_arg_fields:
            raw = request.args.get(field_name)
            if raw in (None, "") or not isinstance(raw, (str, Path)):
                return f"missing_path_arg:{field_name}"
            candidate = normalize_path(raw)
            if not any(
                candidate == root or _is_within(candidate, root)
                for root in roots
            ):
                return f"path_outside_grant_scope:{field_name}"
        return None

    def _predicate_failure(self, request: CapabilityRequest) -> Optional[str]:
        for field_name, spec in (self.arg_predicate or {}).items():
            if field_name not in request.args:
                return f"predicate_arg_missing:{field_name}"
            if not _predicate_matches(request.args[field_name], spec):
                return f"predicate_rejected:{field_name}"
        return None

    def matches(
        self,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        *,
        now: datetime,
    ) -> bool:
        return self.match_failure(request, metadata, now=now) is None

    # -- serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "grantee_kind": self.grantee_kind,
            "grantee_id": self.grantee_id,
            "capability_name": self.capability_name,
            "scope_roots": list(self.scope_roots),
            "arg_predicate": dict(self.arg_predicate),
            "max_risk": float(self.max_risk),
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "reason": self.reason,
            "revoked_at": self.revoked_at,
            "revoked_by": self.revoked_by,
            "revoke_reason": self.revoke_reason,
            "use_count": int(self.use_count),
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsentGrant":
        return cls(
            grant_id=data["grant_id"],
            grantee_kind=data.get("grantee_kind", GRANTEE_KIND_OBJECTIVE),
            grantee_id=data["grantee_id"],
            capability_name=data["capability_name"],
            scope_roots=list(data.get("scope_roots", [])),
            arg_predicate=dict(data.get("arg_predicate") or {}),
            max_risk=float(data["max_risk"]),
            granted_by=data["granted_by"],
            granted_at=data["granted_at"],
            expires_at=data["expires_at"],
            reason=str(data.get("reason") or ""),
            revoked_at=data.get("revoked_at"),
            revoked_by=data.get("revoked_by"),
            revoke_reason=data.get("revoke_reason"),
            use_count=int(data.get("use_count", 0)),
            last_used_at=data.get("last_used_at"),
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class ConsentGrantStore:
    """
    One JSON file per grant, atomically written, `flock`-serialised.

    Use counts are incremented under the same cross-process lock as writes, so
    "what did MR1 do without asking" stays accurate even with a supervisor and
    a REPL sharing the runtime root.
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        clock: Optional[Clock] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._root = self._runtime_root / CONSENT_DIR_NAME
        self._root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or default_clock()
        self._scoped_agents = scoped_agent_store
        self._event_log = EventLog(self._runtime_root / "events")
        self._lock_path = self._root / ".consent.lock"

    @property
    def root(self) -> Path:
        return self._root

    def grant_path(self, grant_id: str) -> Path:
        return self._root / f"{grant_id}.json"

    # -- creation ------------------------------------------------------

    def create(
        self,
        *,
        objective_id: str,
        capability_name: str,
        scope_roots: list[str | Path],
        max_risk: float,
        granted_by: str,
        ttl_s: float,
        arg_predicate: Optional[dict[str, Any]] = None,
        reason: str = "",
    ) -> ConsentGrant:
        if ttl_s is None or float(ttl_s) <= 0:
            raise ConsentGrantError("a consent grant requires a positive TTL")
        self._require_grantor_authority(granted_by, float(max_risk))
        now = self._clock.now()
        grant = ConsentGrant(
            grant_id=new_grant_id(),
            grantee_id=objective_id,
            capability_name=capability_name,
            scope_roots=[str(normalize_path(root)) for root in scope_roots],
            arg_predicate=dict(arg_predicate or {}),
            max_risk=float(max_risk),
            granted_by=granted_by,
            granted_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=float(ttl_s))).isoformat(),
            reason=reason,
        )
        with self._locked():
            self._write(grant)
        self._emit(
            "consent_grant_created",
            grant,
            status="active",
            summary=f"consent granted: {grant.capability_name} for {grant.grantee_id}",
            metadata={
                "capability_name": grant.capability_name,
                "grantee_id": grant.grantee_id,
                "max_risk": grant.max_risk,
                "scope_roots": list(grant.scope_roots),
                "arg_predicate": dict(grant.arg_predicate),
                "expires_at": grant.expires_at,
                "granted_by": grant.granted_by,
                "reason": grant.reason,
            },
        )
        return grant

    def _require_grantor_authority(self, granted_by: str, max_risk: float) -> None:
        """
        Only the human (root) may hand out risk-1.0 standing authority.

        This mirrors the existing rule on `grant_scope`: the autonomous ceiling
        is MAX_AUTONOMOUS_CLEARANCE (0.99), so nothing MR1 can become is allowed
        to grant itself the power to shell out.
        """
        if self._scoped_agents is None:
            if max_risk > MAX_AUTONOMOUS_CLEARANCE:
                raise ConsentGrantError(
                    "risk-1.0 consent requires the root agent; no agent store was provided"
                )
            return
        if self._scoped_agents.is_root_agent(granted_by):
            return
        if max_risk > MAX_AUTONOMOUS_CLEARANCE:
            raise ConsentGrantError(
                "only the root agent may grant consent above the autonomous clearance ceiling"
            )
        grantor = self._scoped_agents.load_agent(granted_by)
        if grantor is None:
            raise ConsentGrantError(f"unknown grantor: {granted_by}")
        if float(grantor.security_clearance) < max_risk:
            raise ConsentGrantError(
                "grantor security clearance is below the requested max_risk"
            )

    # -- reads ---------------------------------------------------------

    def load(self, grant_id: str) -> Optional[ConsentGrant]:
        path = self.grant_path(grant_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return ConsentGrant.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A malformed grant grants nothing.
            return None

    def require(self, grant_id: str) -> ConsentGrant:
        grant = self.load(grant_id)
        if grant is None:
            raise ConsentGrantError(f"consent grant not found: {grant_id}")
        return grant

    def list_grants(self) -> list[ConsentGrant]:
        grants: list[ConsentGrant] = []
        for path in sorted(self._root.glob("grant-*.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    grants.append(ConsentGrant.from_dict(json.load(handle)))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        grants.sort(key=lambda item: (item.granted_at, item.grant_id), reverse=True)
        return grants

    def list_active(self, *, objective_id: Optional[str] = None) -> list[ConsentGrant]:
        now = self._clock.now()
        return [
            grant
            for grant in self.list_grants()
            if grant.is_active(now)
            and (objective_id is None or grant.grantee_id == objective_id)
        ]

    def list_for_objective(self, objective_id: str) -> list[ConsentGrant]:
        return [grant for grant in self.list_grants() if grant.grantee_id == objective_id]

    def expiring_within(self, seconds: float) -> list[ConsentGrant]:
        horizon = self._clock.now() + timedelta(seconds=float(seconds))
        return [
            grant
            for grant in self.list_active()
            if (parse_iso(grant.expires_at) or horizon) <= horizon
        ]

    # -- matching ------------------------------------------------------

    def match(
        self,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        *,
        objective_id: Optional[str] = None,
    ) -> Optional[ConsentGrant]:
        """The first active grant that authorizes this request, if any."""
        target = objective_id or getattr(request, "objective_id", None)
        if not target:
            return None
        now = self._clock.now()
        for grant in self.list_active(objective_id=target):
            if grant.matches(request, metadata, now=now):
                return grant
        return None

    # -- lifecycle -----------------------------------------------------

    def record_use(
        self,
        grant_id: str,
        *,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        audit_id: Optional[str] = None,
    ) -> Optional[ConsentGrant]:
        """
        Count one unattended execution against a grant.

        This is *the* accountability metric: what MR1 did without asking.
        """
        with self._locked():
            grant = self.load(grant_id)
            if grant is None:
                return None
            updated = ConsentGrant.from_dict({
                **grant.to_dict(),
                "use_count": int(grant.use_count) + 1,
                "last_used_at": self._clock.now_iso(),
            })
            self._write(updated)
        self._emit(
            "consent_grant_used",
            updated,
            status="used",
            summary=f"consent grant used: {updated.capability_name}",
            workflow_id=workflow_id,
            task_id=task_id,
            audit_id=audit_id,
            metadata={
                "capability_name": updated.capability_name,
                "grantee_id": updated.grantee_id,
                "use_count": updated.use_count,
                "audit_id": audit_id,
            },
        )
        return updated

    def revoke(
        self,
        grant_id: str,
        *,
        revoked_by: str = "operator",
        reason: str = "revoked",
    ) -> Optional[ConsentGrant]:
        with self._locked():
            grant = self.load(grant_id)
            if grant is None:
                return None
            if grant.revoked:
                return grant
            updated = ConsentGrant.from_dict({
                **grant.to_dict(),
                "revoked_at": self._clock.now_iso(),
                "revoked_by": revoked_by,
                "revoke_reason": reason,
            })
            self._write(updated)
        self._emit(
            "consent_grant_revoked",
            updated,
            status="revoked",
            summary=f"consent revoked: {updated.capability_name} for {updated.grantee_id}",
            metadata={
                "capability_name": updated.capability_name,
                "grantee_id": updated.grantee_id,
                "revoked_by": revoked_by,
                "reason": reason,
                "use_count": updated.use_count,
            },
        )
        return updated

    def revoke_all(
        self,
        *,
        revoked_by: str = "operator",
        reason: str = "revoked",
        objective_id: Optional[str] = None,
    ) -> list[str]:
        revoked: list[str] = []
        for grant in self.list_active(objective_id=objective_id):
            if self.revoke(grant.grant_id, revoked_by=revoked_by, reason=reason) is not None:
                revoked.append(grant.grant_id)
        return revoked

    def expire_stale(self) -> list[str]:
        """
        Emit `consent_grant_expired` once for every grant that just aged out.

        Expiry needs no write — `is_active()` is computed against the clock, so
        an expired grant stops authorizing the moment its deadline passes, even
        if nothing ever sweeps. The sweep exists to make it *visible*.
        """
        now = self._clock.now()
        newly_expired: list[str] = []
        for grant in self.list_grants():
            if grant.revoked or not grant.is_expired(now):
                continue
            marker = self._expiry_marker_path(grant.grant_id)
            if marker.exists():
                continue
            marker.write_text(self._clock.now_iso(), encoding="utf-8")
            self._emit(
                "consent_grant_expired",
                grant,
                status="expired",
                summary=f"consent expired: {grant.capability_name} for {grant.grantee_id}",
                metadata={
                    "capability_name": grant.capability_name,
                    "grantee_id": grant.grantee_id,
                    "expires_at": grant.expires_at,
                    "use_count": grant.use_count,
                },
            )
            newly_expired.append(grant.grant_id)
        return newly_expired

    def unattended_executions(self) -> dict[str, int]:
        return {grant.grant_id: int(grant.use_count) for grant in self.list_grants()}

    # -- internals -----------------------------------------------------

    def _expiry_marker_path(self, grant_id: str) -> Path:
        return self._root / f"{grant_id}.expired"

    def _write(self, grant: ConsentGrant) -> None:
        path = self.grant_path(grant.grant_id)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(grant.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)

    def _locked(self):
        return _FileLock(self._lock_path)

    def _emit(
        self,
        event_type: str,
        grant: ConsentGrant,
        *,
        status: str,
        summary: str,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        audit_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            self._event_log.emit(
                event_type=event_type,
                actor_id=grant.granted_by,
                actor_type="mr1",
                target_id=grant.grant_id,
                target_type="consent_grant",
                status=status,
                summary=summary,
                workflow_id=workflow_id,
                task_id=task_id,
                audit_id=audit_id,
                record_path=str(self.grant_path(grant.grant_id)),
                metadata={"grant_id": grant.grant_id, **dict(metadata or {})},
            )
        except Exception:
            pass


class _FileLock:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._handle = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+b")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: Any) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
