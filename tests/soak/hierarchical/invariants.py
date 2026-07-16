"""
Hierarchical and runtime-health invariants.

Two families of checks, asserted throughout the soak and at the end:

* **Hierarchy** — read from the real agent / message / workflow stores:
  no cycle, one valid parent per child, depth and breadth within ceilings,
  no duplicate titles, every non-root agent has a traceable mission, every
  agent-owned workflow is attributable, and every message has a valid
  sender, recipient, and delivery state (no orphans).

* **Runtime health** — read from the driver's resource samples: RSS, file
  descriptors, tick latency, event-byte growth, and — the property the whole
  autonomy design rests on — that idle turns do not burn brain calls.

Every check returns `(severity, title, detail)` findings; an empty list is a
pass. Each is written so it is *capable of failing* — a soak tool that cannot
fail manufactures confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Finding = Tuple[str, str, str]  # (severity, title, detail)

# Phrases that mark a mission as an unresolved clarifying question rather than
# an actual mission — the design brain declined to design (asked the human a
# question instead), but agent creation proceeded anyway. Discovered from a
# real overnight run: a hardcoded title fallback let this through silently.
_UNRESOLVED_MISSION_MARKERS = (
    "i need clarification",
    "i'm not sure what",
    "i am not sure what",
    "could you clarify",
    "can you clarify",
    "please clarify",
    "please specify",
    "what domain do you want",
    "which domain",
    "what would you like",
)


def _looks_like_unresolved_question(mission: str) -> bool:
    text = (mission or "").strip().casefold()
    if not text:
        return False
    return any(marker in text[:200] for marker in _UNRESOLVED_MISSION_MARKERS)


@dataclass(frozen=True)
class HierarchyLimits:
    """Hard safety ceilings. Reaching one must trigger clarify/reuse/defer/
    escalate — never a silent breach."""

    max_depth: int = 3
    max_children_per_agent: int = 3
    max_total_agents: int = 12
    max_total_workflows: int = 40
    max_workflows_per_agent: int = 5
    max_agent_creations_per_hour: int = 4


# ----------------------------------------------------------------------
# Hierarchy invariants
# ----------------------------------------------------------------------

def check_hierarchy(agents: List[Any], root_id: str, limits: HierarchyLimits) -> List[Finding]:
    """`agents` is a list of PersistentAgent (from `store.list_agents()`)."""
    findings: List[Finding] = []
    by_id: Dict[str, Any] = {a.agent_id: a for a in agents}
    non_root = [a for a in agents if a.agent_id != root_id]

    # -- total count ceiling
    if len(non_root) > limits.max_total_agents:
        findings.append((
            "high", "Too many persistent agents",
            f"{len(non_root)} persistent agents exceeds ceiling {limits.max_total_agents}",
        ))

    # -- duplicate titles (case-insensitive), across live agents
    seen: Dict[str, str] = {}
    for a in agents:
        key = (a.title or "").strip().casefold()
        if not key:
            continue
        status = str(getattr(a, "status", "active"))
        if status in {"terminated", "abandoned"}:
            continue
        if key in seen and seen[key] != a.agent_id:
            findings.append((
                "high", "Duplicate agent title",
                f"title {a.title!r} held by both {seen[key]} and {a.agent_id}",
            ))
        else:
            seen[key] = a.agent_id

    # -- one valid parent per child, no cycle, depth
    children_count: Dict[str, int] = {}
    for a in non_root:
        parent_id = a.parent_agent_id
        if parent_id is None:
            findings.append((
                "high", "Child without a parent",
                f"agent {a.agent_id} ({a.title!r}) has no parent_agent_id",
            ))
            continue
        if parent_id not in by_id:
            findings.append((
                "high", "Child points at an unknown parent",
                f"agent {a.agent_id} parent {parent_id} is not a known agent",
            ))
            continue
        children_count[parent_id] = children_count.get(parent_id, 0) + 1

        # walk to root; detect cycle and measure depth
        depth = 0
        cursor: Optional[str] = a.agent_id
        visited: set[str] = set()
        while cursor is not None and cursor != root_id:
            if cursor in visited:
                findings.append((
                    "high", "Cycle in parent-child graph",
                    f"agent {a.agent_id} is in a parent cycle at {cursor}",
                ))
                break
            visited.add(cursor)
            node = by_id.get(cursor)
            if node is None:
                findings.append((
                    "high", "Broken ancestry chain",
                    f"agent {a.agent_id} ancestry reaches unknown {cursor}",
                ))
                break
            cursor = node.parent_agent_id
            depth += 1
            if depth > limits.max_depth + 2:  # guard against runaway walk
                findings.append((
                    "high", "Ancestry deeper than the ceiling allows",
                    f"agent {a.agent_id} exceeds max_depth {limits.max_depth}",
                ))
                break
        else:
            if depth > limits.max_depth:
                findings.append((
                    "high", "Agent exceeds max depth",
                    f"agent {a.agent_id} ({a.title!r}) at depth {depth} > {limits.max_depth}",
                ))

    # -- breadth ceiling
    #
    # The per-agent breadth limit exists to stop a *sub-agent's* delegation from
    # fanning out. The root orchestrator (MR1) is expected to own many
    # specialists directly, so it is bounded by the *total-agents* ceiling
    # instead — runaway fanout at root is still caught, just by the right limit.
    for parent_id, count in children_count.items():
        limit = limits.max_total_agents if parent_id == root_id else limits.max_children_per_agent
        if count > limit:
            parent = by_id.get(parent_id)
            title = parent.title if parent else parent_id
            findings.append((
                "high", "Agent exceeds max children",
                f"agent {title!r} has {count} children > {limit}",
            ))

    # -- mission traceability + workflow attribution
    for a in non_root:
        if str(getattr(a, "status", "active")) in {"terminated", "abandoned"}:
            continue
        mission = getattr(a, "mission", None)
        parent_request = getattr(a, "parent_request", None)
        if not (mission or parent_request):
            findings.append((
                "medium", "Agent has no traceable mission",
                f"agent {a.agent_id} ({a.title!r}) has neither a mission nor a parent request",
            ))
        elif mission and _looks_like_unresolved_question(mission):
            findings.append((
                "high", "Agent created with an unresolved question as its mission",
                f"agent {a.agent_id} ({a.title!r}) mission reads as a clarifying question, "
                f"not a mission: {mission[:150]!r}",
            ))
        owned = list(getattr(a, "owned_workflow_ids", []) or [])
        if len(owned) > limits.max_workflows_per_agent:
            findings.append((
                "high", "Agent owns too many workflows",
                f"agent {a.title!r} owns {len(owned)} > {limits.max_workflows_per_agent}",
            ))

    return findings


def check_messages(messages: List[Any], agents: List[Any]) -> List[Finding]:
    """`messages` from `message_store.list_messages()`."""
    findings: List[Finding] = []
    agent_ids = {a.agent_id for a in agents}
    for m in messages:
        if m.from_agent_id not in agent_ids:
            findings.append((
                "high", "Message from an unknown sender",
                f"message {m.message_id} from {m.from_agent_id} (not a known agent)",
            ))
        if m.to_agent_id not in agent_ids:
            findings.append((
                "high", "Message to an unknown recipient",
                f"message {m.message_id} to {m.to_agent_id} (not a known agent)",
            ))
        status = str(getattr(m, "status", "") or "")
        if not status:
            findings.append((
                "medium", "Message without a delivery state",
                f"message {m.message_id} has no status",
            ))
    return findings


def check_workflows_at_shutdown(workflows: List[Dict[str, Any]], limits: HierarchyLimits) -> List[Finding]:
    """
    `workflows` is a list of workflow dicts (from runtime_access.list_workflows).
    At shutdown every workflow must be terminal, explicitly blocked, or belong
    to active work — never silently mid-flight-and-forgotten.
    """
    findings: List[Finding] = []
    if len(workflows) > limits.max_total_workflows:
        findings.append((
            "high", "Too many workflows",
            f"{len(workflows)} workflows exceeds ceiling {limits.max_total_workflows}",
        ))
    terminal = {"succeeded", "failed", "cancelled", "timed_out", "skipped"}
    ok_nonterminal = {"blocked", "running", "waiting", "ready", "created"}
    for wf in workflows:
        status = str(wf.get("status") or wf.get("state") or "").lower()
        if not status:
            findings.append((
                "medium", "Workflow without a status",
                f"workflow {wf.get('workflow_id')} has no status field",
            ))
            continue
        if status not in terminal and status not in ok_nonterminal:
            findings.append((
                "medium", "Workflow in an unexpected end state",
                f"workflow {wf.get('workflow_id')} status={status!r}",
            ))
    return findings


# ----------------------------------------------------------------------
# Runtime-health invariants (over resource samples)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class HealthThresholds:
    max_rss_growth_ratio: float = 2.0      # end RSS / start RSS
    max_fd_growth: int = 64                # end FD - start FD
    max_latency_growth_ratio: float = 4.0  # late median / early median
    max_event_bytes_per_turn: int = 200_000


def check_health(samples: List[Dict[str, Any]], *, thresholds: Optional[HealthThresholds] = None) -> List[Finding]:
    thresholds = thresholds or HealthThresholds()
    findings: List[Finding] = []
    if len(samples) < 2:
        findings.append((
            "medium", "Too few resource samples",
            f"only {len(samples)} samples; cannot assess runtime health",
        ))
        return findings

    first, last = samples[0], samples[-1]

    def _f(sample: Dict[str, Any], key: str) -> Optional[float]:
        v = sample.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    rss0, rss1 = _f(first, "rss_bytes"), _f(last, "rss_bytes")
    if rss0 and rss1 and rss0 > 0 and rss1 / rss0 > thresholds.max_rss_growth_ratio:
        findings.append((
            "high", "RSS grew unhealthily",
            f"RSS {rss0/1e6:.1f}MB -> {rss1/1e6:.1f}MB ({rss1/rss0:.2f}x)",
        ))

    fd0, fd1 = _f(first, "open_fds"), _f(last, "open_fds")
    if fd0 is not None and fd1 is not None and (fd1 - fd0) > thresholds.max_fd_growth:
        findings.append((
            "high", "File descriptors leaked",
            f"open FDs {fd0:.0f} -> {fd1:.0f} (+{fd1-fd0:.0f})",
        ))

    # latency creep: compare median of first third vs last third
    lat = [s.get("turn_ms") for s in samples if isinstance(s.get("turn_ms"), (int, float))]
    if len(lat) >= 6:
        third = max(1, len(lat) // 3)
        early = sorted(lat[:third])[third // 2]
        late = sorted(lat[-third:])[third // 2]
        if early > 0 and late / early > thresholds.max_latency_growth_ratio:
            findings.append((
                "medium", "Turn latency crept upward",
                f"median turn {early:.1f}ms (early) -> {late:.1f}ms (late)",
            ))

    # event-byte growth should scale with work, not explode per turn
    ev0, ev1 = _f(first, "event_bytes"), _f(last, "event_bytes")
    turns = _f(last, "turn_index") or len(samples)
    if ev0 is not None and ev1 is not None and turns and turns > 0:
        per_turn = (ev1 - ev0) / turns
        if per_turn > thresholds.max_event_bytes_per_turn:
            findings.append((
                "medium", "Event log grew faster than expected",
                f"~{per_turn:.0f} event bytes/turn over {turns:.0f} turns",
            ))

    return findings


def check_idle_brain_discipline(
    idle_brain_calls: int,
    idle_ticks: int,
) -> List[Finding]:
    """
    Idle scheduler drain ticks between turns must not call the brain. The
    conversation drives brain calls; the drain must be free.
    """
    if idle_brain_calls > 0:
        return [(
            "high", "Brain called on idle drain ticks",
            f"{idle_brain_calls} brain call(s) across {idle_ticks} idle drain ticks",
        )]
    return []
