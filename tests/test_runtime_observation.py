from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mr1.dataflow import ResolvedTaskInput, TaskOutput
from mr1.messages import MessageStore
from mr1.runtime_access import RuntimeAccess
from mr1.runtime_observation import (
    execute_observation_request,
    parse_observation_request,
)
from mr1.scoped_agents import (
    PersistentAgentStore,
    build_assignment_packet,
    render_assignment_mission,
)
from mr1.workflow_models import Provenance, Task, TaskStatus, Workflow, WorkflowStatus
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def message_store(tmp_path, agent_store):
    return MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)


@pytest.fixture
def runtime_access(workflow_store, agent_store, message_store):
    return RuntimeAccess(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )


def _seed_runtime_detail(
    workflow_store: WorkflowStore,
    agent_store: PersistentAgentStore,
    message_store: MessageStore,
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "Sentinel")
    parent_request = "Parent request " + ("beta " * 120)
    assignment_packet = build_assignment_packet(
        root,
        child.title,
        parent_request,
        {
            "assigned_clearance": child.security_clearance,
            "agents": [root.agent_id, child.agent_id],
            "messages": ["msg-1"],
            "workflows": ["wf-1"],
        },
    )
    child.mission = render_assignment_mission(assignment_packet) or ("Mission " + ("alpha " * 120))
    child.parent_request = parent_request
    child.assignment_packet = assignment_packet
    agent_store.save_agent(child)
    long_body = "Full body " + ("delta " * 1200)
    message = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="question",
        subject="Need detail",
        body=long_body,
    )
    task = Task(
        task_id="tk-1",
        workflow_id="wf-1",
        label="inspect",
        title="Inspect",
        task_kind="agent",
        agent_type="kazi",
        prompt="Inspect the runtime detail.",
        status=TaskStatus.FAILED,
        last_error="Detailed failure " * 40,
        result_summary="Summary " * 40,
    )
    workflow = Workflow(
        workflow_id="wf-1",
        title="Observed workflow",
        status=WorkflowStatus.FAILED,
        created_by=Provenance(type="agent", id=root.agent_id),
        owner_agent_id=child.agent_id,
        owner_agent_title=child.title,
        tasks={task.task_id: task},
        label_to_task_id={task.label: task.task_id},
    )
    workflow_store.save_workflow(workflow)
    workflow_store.write_result(
        workflow.workflow_id,
        task.task_id,
        {
            "task_id": task.task_id,
            "workflow_id": workflow.workflow_id,
            "status": "failed",
            "summary": "Detailed summary",
            "text": "Complete workflow detail",
            "error": "policy block",
            "error_type": "policy_block",
            "failure_type": "policy_block",
        },
    )
    workflow_store.write_task_output(
        workflow.workflow_id,
        task.task_id,
        TaskOutput(
            task_id=task.task_id,
            workflow_id=workflow.workflow_id,
            status="failed",
            summary="Detailed output summary",
            text="Complete output payload",
        ),
    )
    workflow_store.write_task_inputs(
        workflow.workflow_id,
        task.task_id,
        [
            ResolvedTaskInput(
                name="message_body",
                source=f"{message.message_id}.body",
                resolved_task_id=task.task_id,
                resolved_type="text",
                value=long_body,
            )
        ],
    )
    return root, child, message, workflow


def _maybe_execute(raw: str, runtime_access: RuntimeAccess, *, caller_agent_id: str):
    parsed = parse_observation_request(raw)
    if parsed.status == "valid":
        return execute_observation_request(
            parsed.request,
            runtime_access,
            caller_agent_id=caller_agent_id,
        )
    return None


def test_parser_accepts_valid_observe_block():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"read_message","args":{"message_id":"msg-123"}}]}[/OBSERVE]'
    )

    assert parsed.status == "valid"
    assert parsed.request is not None
    assert parsed.request.calls[0].name == "read_message"
    assert parsed.request.calls[0].args == {"message_id": "msg-123"}


def test_parser_rejects_malformed_json():
    parsed = parse_observation_request('[OBSERVE]{"calls":[}[/OBSERVE]')

    assert parsed.status == "invalid"
    assert parsed.error == "invalid observation JSON"


def test_parser_rejects_unknown_call_name():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"send_message","args":{"message_id":"msg-123"}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"
    assert parsed.error == "unknown observation call: send_message"


def test_parser_rejects_unexpected_args():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"read_message","args":{"message_id":"msg-123","full":false}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"
    assert parsed.error == "call 0 has unexpected args: full"


def test_executor_reads_full_message_body(runtime_access, workflow_store, agent_store, message_store):
    root, _child, message, _workflow = _seed_runtime_detail(
        workflow_store,
        agent_store,
        message_store,
    )
    parsed = parse_observation_request(
        f'[OBSERVE]{{"calls":[{{"name":"read_message","args":{{"message_id":"{message.message_id}"}}}}]}}[/OBSERVE]'
    )

    result = execute_observation_request(
        parsed.request,
        runtime_access,
        caller_agent_id=root.agent_id,
    )

    assert result["request_valid"] is True
    assert result["executed"] is True
    assert result["calls"][0]["ok"] is True
    assert result["calls"][0]["result"]["message_id"] == message.message_id
    assert result["calls"][0]["result"]["body"] == message.body
    assert result["calls"][0]["result"]["body_truncated"] is False


def test_executor_reads_full_agent_detail(runtime_access, workflow_store, agent_store, message_store):
    root, child, _message, _workflow = _seed_runtime_detail(
        workflow_store,
        agent_store,
        message_store,
    )
    parsed = parse_observation_request(
        f'[OBSERVE]{{"calls":[{{"name":"read_agent","args":{{"agent_id":"{child.agent_id}"}}}}]}}[/OBSERVE]'
    )

    result = execute_observation_request(
        parsed.request,
        runtime_access,
        caller_agent_id=root.agent_id,
    )

    assert result["calls"][0]["ok"] is True
    assert result["calls"][0]["result"]["agent_id"] == child.agent_id
    assert result["calls"][0]["result"]["mission"] == child.mission
    assert result["calls"][0]["result"]["parent_request"] == child.parent_request
    assert result["calls"][0]["result"]["assignment_packet"]["full_parent_request"] == child.parent_request


def test_executor_reads_full_workflow_detail(runtime_access, workflow_store, agent_store, message_store):
    root, _child, _message, workflow = _seed_runtime_detail(
        workflow_store,
        agent_store,
        message_store,
    )
    parsed = parse_observation_request(
        f'[OBSERVE]{{"calls":[{{"name":"read_workflow","args":{{"workflow_id":"{workflow.workflow_id}"}}}}]}}[/OBSERVE]'
    )

    result = execute_observation_request(
        parsed.request,
        runtime_access,
        caller_agent_id=root.agent_id,
    )

    assert result["calls"][0]["ok"] is True
    workflow_detail = result["calls"][0]["result"]
    assert workflow_detail["workflow_id"] == workflow.workflow_id
    assert workflow_detail["task_details"]["tk-1"]["result_payload"]["text"] == "Complete workflow detail"
    assert workflow_detail["task_details"]["tk-1"]["output_payload"]["text"] == "Complete output payload"


def test_invalid_observation_does_not_execute_anything():
    runtime_access = MagicMock(spec=RuntimeAccess)

    result = _maybe_execute(
        '[OBSERVE]{"calls":[{"name":"read_message","args":{"message_id":"msg-1","full":true}}]}[/OBSERVE]',
        runtime_access,
        caller_agent_id="MR1",
    )

    assert result is None
    runtime_access.read_message.assert_not_called()


# ---------------------------------------------------------------------------
# search_memory observation tests
# ---------------------------------------------------------------------------

def test_search_memory_in_allowlist():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"authentication bug"}}]}[/OBSERVE]'
    )

    assert parsed.status == "valid"
    assert parsed.request.calls[0].name == "search_memory"
    assert parsed.request.calls[0].args == {"query": "authentication bug"}


def test_search_memory_with_limit_in_allowlist():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"crash","limit":3}}]}[/OBSERVE]'
    )

    assert parsed.status == "valid"


def test_search_memory_requires_query():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"
    assert "query" in parsed.error


def test_search_memory_query_must_be_string():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":42}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"


def test_search_memory_limit_above_10_rejected():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"x","limit":11}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"
    assert "limit" in parsed.error


def test_search_memory_limit_zero_rejected():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"x","limit":0}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"
    assert "limit" in parsed.error


def test_search_memory_limit_non_int_rejected():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"x","limit":"five"}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"


def test_search_memory_unknown_arg_rejected():
    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"x","scope":"root"}}]}[/OBSERVE]'
    )

    assert parsed.status == "invalid"
    assert "scope" in parsed.error


def test_search_memory_executes_and_returns_structured_result(runtime_access):
    runtime_access.search_memory = MagicMock(return_value={
        "query": "authentication bug",
        "results": [
            {
                "source": "insight",
                "id": "ins-001",
                "summary": "Auth tokens expire unexpectedly under load",
                "score": 0.91,
                "ref": "ins-001",
                "full_available": True,
            }
        ],
    })

    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"authentication bug","limit":5}}]}[/OBSERVE]'
    )
    result = execute_observation_request(parsed.request, runtime_access, caller_agent_id="MR1")

    assert result["request_valid"] is True
    assert result["calls"][0]["ok"] is True
    data = result["calls"][0]["result"]
    assert data["query"] == "authentication bug"
    assert len(data["results"]) == 1
    assert data["results"][0]["source"] == "insight"
    runtime_access.search_memory.assert_called_once_with(
        query="authentication bug",
        limit=5,
        caller_agent_id="MR1",
    )


def test_search_memory_empty_memory_returns_empty_results(runtime_access):
    runtime_access.search_memory = MagicMock(return_value={
        "query": "prior deployment",
        "results": [],
    })

    parsed = parse_observation_request(
        '[OBSERVE]{"calls":[{"name":"search_memory","args":{"query":"prior deployment"}}]}[/OBSERVE]'
    )
    result = execute_observation_request(parsed.request, runtime_access, caller_agent_id="MR1")

    assert result["calls"][0]["ok"] is True
    assert result["calls"][0]["result"]["results"] == []
