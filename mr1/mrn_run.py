"""
Bounded multi-step MRn runs under explicit policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from mr1.messages import MessageStore
from mr1.mrn_loop import ALLOWED_MRN_ACTIONS, MRnStepResult, MRnStepRunner
from mr1.scoped_agents import (
    AgentScopeError,
    PersistentAgentStore,
    is_agent_terminal,
    new_run_id,
)
from mr1.workflow_store import WorkflowStore


RUN_STOP_REASONS = frozenset({
    "max_steps",
    "waiting",
    "blocked",
    "idle",
    "parent_message",
    "workflow_running",
    "workflow_limit",
    "runtime_limit",
    "disallowed_action",
    "confirmation_required",
})
_RUNNING_WORKFLOW_STATUSES = frozenset({"pending", "running"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MRnRunPolicy:
    max_steps: int = 3
    max_workflows_created: int = 2
    stop_on_parent_message: bool = True
    stop_on_blocked: bool = True
    stop_on_waiting: bool = True
    stop_on_idle: bool = False
    stop_on_workflow_running: bool = True
    allowed_actions: Optional[list[str]] = None
    require_confirmation_for_workflows: bool = True
    max_runtime_s: Optional[int] = None

    def validate(self) -> "MRnRunPolicy":
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_workflows_created < 0:
            raise ValueError("max_workflows_created must be >= 0")
        if self.max_runtime_s is not None and self.max_runtime_s < 1:
            raise ValueError("max_runtime_s must be >= 1")
        if self.allowed_actions is not None:
            unknown = sorted(set(self.allowed_actions) - set(ALLOWED_MRN_ACTIONS))
            if unknown:
                raise ValueError(
                    "unknown actions rejected deterministically: "
                    + ", ".join(unknown)
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_workflows_created": self.max_workflows_created,
            "stop_on_parent_message": self.stop_on_parent_message,
            "stop_on_blocked": self.stop_on_blocked,
            "stop_on_waiting": self.stop_on_waiting,
            "stop_on_idle": self.stop_on_idle,
            "stop_on_workflow_running": self.stop_on_workflow_running,
            "allowed_actions": list(self.allowed_actions) if self.allowed_actions is not None else None,
            "require_confirmation_for_workflows": self.require_confirmation_for_workflows,
            "max_runtime_s": self.max_runtime_s,
        }


@dataclass(frozen=True)
class MRnRunResult:
    run_id: str
    agent_id: str
    caller_agent_id: str
    started_at: str
    finished_at: str
    policy: dict[str, Any]
    steps: list[dict[str, Any]]
    workflows_created: int
    messages_created: int
    stopped_reason: str
    status: str
    final_run_status: str

    @property
    def steps_completed(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "caller_agent_id": self.caller_agent_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "policy": dict(self.policy),
            "steps": list(self.steps),
            "workflows_created": self.workflows_created,
            "messages_created": self.messages_created,
            "stopped_reason": self.stopped_reason,
            "status": self.status,
            "final_run_status": self.final_run_status,
        }


class MRnRunRunner:
    def __init__(
        self,
        *,
        workflow_store: Optional[WorkflowStore] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        message_store: Optional[MessageStore] = None,
        step_runner: Optional[MRnStepRunner] = None,
        workflow_compiler: Optional[Any] = None,
    ):
        self._workflow_store = workflow_store or WorkflowStore()
        self._scoped_agents = scoped_agent_store or PersistentAgentStore(
            root=self._workflow_store.root.parent / "agents"
        )
        self._message_store = message_store or MessageStore(
            root=self._workflow_store.root.parent / "messages",
            scoped_agent_store=self._scoped_agents,
        )
        self._workflow_compiler = workflow_compiler
        self._step_runner = step_runner

    def run(
        self,
        agent_id: str,
        policy: MRnRunPolicy,
        caller_agent_id: Optional[str] = None,
    ) -> MRnRunResult:
        resolved_caller = caller_agent_id or self._scoped_agents.root_agent_id
        validated_policy = policy.validate()
        if not self._scoped_agents.can_manage_agent(resolved_caller, agent_id):
            raise AgentScopeError("access denied: agent not in scope")
        agent = self._scoped_agents.require_agent(agent_id)
        if is_agent_terminal(agent):
            raise ValueError(f"agent terminated: {agent_id}")
        if not (agent.mission and agent.mission.strip()):
            raise ValueError(f"no mission assigned: {agent_id}")

        run_id = new_run_id()
        started_at = _now_iso()
        started_monotonic = time.monotonic()
        step_runner = self._step_runner or MRnStepRunner(
            workflow_store=self._workflow_store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
            workflow_compiler=self._workflow_compiler,
            require_confirmation_for_workflows=validated_policy.require_confirmation_for_workflows,
        )

        step_summaries: list[dict[str, Any]] = []
        workflows_created = 0
        messages_created = 0
        stopped_reason: Optional[str] = None
        status = "completed"

        try:
            for _ in range(validated_policy.max_steps):
                elapsed_s = time.monotonic() - started_monotonic
                if (
                    validated_policy.max_runtime_s is not None
                    and elapsed_s >= validated_policy.max_runtime_s
                ):
                    stopped_reason = "runtime_limit"
                    status = "stopped"
                    break

                step_result = step_runner.step(
                    agent_id,
                    caller_agent_id=resolved_caller,
                    run_id=run_id,
                )
                step_summaries.append(step_result.to_dict())
                if step_result.created_workflow_id is not None:
                    workflows_created += 1
                if step_result.message_id is not None:
                    messages_created += 1

                elapsed_s = time.monotonic() - started_monotonic
                stopped_reason = self._stop_reason_for_step(
                    agent_id=agent_id,
                    policy=validated_policy,
                    step_result=step_result,
                    workflows_created=workflows_created,
                    elapsed_s=elapsed_s,
                )
                if stopped_reason is not None:
                    status = "stopped" if stopped_reason != "max_steps" else "completed"
                    break

            if stopped_reason is None:
                stopped_reason = "max_steps"
                status = "completed"

            finished_at = _now_iso()
            final_run_status = self._scoped_agents.require_agent(agent_id).run_status
            result = MRnRunResult(
                run_id=run_id,
                agent_id=agent_id,
                caller_agent_id=resolved_caller,
                started_at=started_at,
                finished_at=finished_at,
                policy=validated_policy.to_dict(),
                steps=step_summaries,
                workflows_created=workflows_created,
                messages_created=messages_created,
                stopped_reason=stopped_reason,
                status=status,
                final_run_status=final_run_status,
            )
            self._persist_run_result(result)
            return result
        except Exception:
            finished_at = _now_iso()
            final_run_status = self._scoped_agents.require_agent(agent_id).run_status
            failed_result = MRnRunResult(
                run_id=run_id,
                agent_id=agent_id,
                caller_agent_id=resolved_caller,
                started_at=started_at,
                finished_at=finished_at,
                policy=validated_policy.to_dict(),
                steps=step_summaries,
                workflows_created=workflows_created,
                messages_created=messages_created,
                stopped_reason=stopped_reason or "runtime_limit",
                status="failed",
                final_run_status=final_run_status,
            )
            self._persist_run_result(failed_result)
            raise

    def _stop_reason_for_step(
        self,
        *,
        agent_id: str,
        policy: MRnRunPolicy,
        step_result: MRnStepResult,
        workflows_created: int,
        elapsed_s: float,
    ) -> Optional[str]:
        agent = self._scoped_agents.require_agent(agent_id)
        if policy.allowed_actions is not None and step_result.action not in set(policy.allowed_actions):
            return "disallowed_action"
        if step_result.confirmation_required:
            return "confirmation_required"
        if policy.stop_on_blocked and agent.run_status == "blocked":
            return "blocked"
        if policy.stop_on_parent_message and step_result.created_parent_message_id is not None:
            return "parent_message"
        if (
            policy.stop_on_workflow_running
            and step_result.created_workflow_status in _RUNNING_WORKFLOW_STATUSES
        ):
            return "workflow_running"
        if step_result.created_workflow_id is not None and workflows_created >= policy.max_workflows_created:
            return "workflow_limit"
        if policy.stop_on_waiting and agent.run_status == "waiting":
            return "waiting"
        if policy.stop_on_idle and agent.run_status == "idle":
            return "idle"
        if policy.max_runtime_s is not None and elapsed_s >= policy.max_runtime_s:
            return "runtime_limit"
        return None

    def _persist_run_result(self, result: MRnRunResult) -> None:
        record = result.to_dict()
        self._scoped_agents.write_run_log(result.agent_id, result.run_id, record)
        self._scoped_agents.append_run_summary(
            result.agent_id,
            {
                "run_id": result.run_id,
                "agent_id": result.agent_id,
                "caller_agent_id": result.caller_agent_id,
                "finished_at": result.finished_at,
                "stopped_reason": result.stopped_reason,
                "status": result.status,
                "steps_completed": result.steps_completed,
                "workflows_created": result.workflows_created,
                "messages_created": result.messages_created,
                "final_run_status": result.final_run_status,
            },
        )
        agent = self._scoped_agents.require_agent(result.agent_id)
        agent.last_run = {
            "run_id": result.run_id,
            "stopped_reason": result.stopped_reason,
            "step_count": result.steps_completed,
            "finished_at": result.finished_at,
        }
        self._scoped_agents.save_agent(agent)
