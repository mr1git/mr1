"""Tests for deterministic workflow dataflow helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mr1.dataflow import (
    Artifact,
    ResolvedTaskInput,
    TaskInputSpec,
    TaskOutput,
    build_materialized_prompt,
    materialize_task_inputs,
    parse_input_reference,
)
from mr1.scheduler import build_workflow_from_spec
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


def _workflow():
    return build_workflow_from_spec(
        {
            "title": "dataflow",
            "tasks": [
                {"label": "produce", "title": "Produce", "prompt": "produce"},
                {
                    "label": "consume",
                    "title": "Consume",
                    "prompt": "consume",
                    "depends_on": ["produce"],
                    "inputs": [{"name": "input_text", "from": "produce.result.text"}],
                },
            ],
        },
        PROV,
    )


class TestModelRoundtrip:
    def test_task_output_roundtrip(self):
        output = TaskOutput(
            task_id="tk-1",
            workflow_id="wf-1",
            status="succeeded",
            summary="done",
            text="hello",
            data={"x": 1},
            metrics={"accuracy": 0.9},
            artifacts=[
                Artifact(
                    artifact_id="art-1",
                    workflow_id="wf-1",
                    task_id="tk-1",
                    name="report",
                    kind="json",
                    path="/tmp/report.json",
                )
            ],
            metadata={"runner": "mock"},
        )
        assert TaskOutput.from_dict(output.to_dict()).to_dict() == output.to_dict()

    def test_artifact_roundtrip(self):
        artifact = Artifact(
            artifact_id="art-1",
            workflow_id="wf-1",
            task_id="tk-1",
            name="report",
            kind="json",
            path="/tmp/report.json",
            metadata={"size": 10},
        )
        assert Artifact.from_dict(artifact.to_dict()) == artifact

    def test_task_input_spec_roundtrip(self):
        spec = TaskInputSpec(name="metrics", from_ref="eval.result.metrics")
        assert TaskInputSpec.from_dict(spec.to_dict()) == spec

    def test_resolved_input_roundtrip(self):
        resolved = ResolvedTaskInput(
            name="producer_text",
            source="produce.result.text",
            resolved_task_id="tk-1",
            resolved_type="text",
            value="hello",
            metadata={"truncated": False},
        )
        assert ResolvedTaskInput.from_dict(resolved.to_dict()) == resolved


class TestReferenceResolution:
    def test_parse_reference(self):
        parsed = parse_input_reference("train.result.metrics.accuracy")
        assert parsed.label == "train"
        assert parsed.root == "result"
        assert parsed.path == ("metrics", "accuracy")

    def test_parse_branch_context_reference(self):
        parsed = parse_input_reference("branch.status")
        assert parsed.label == "branch"
        assert parsed.root == "status"
        assert parsed.path == ()

    def test_resolves_result_fields_and_nested_mappings(self, store):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        consumer = wf.task_by_label("consume")
        store.save_workflow(wf)
        store.write_task_output(
            wf.workflow_id,
            producer.task_id,
            TaskOutput(
                task_id=producer.task_id,
                workflow_id=wf.workflow_id,
                status="succeeded",
                summary="short",
                text="hello world",
                data={"nested": {"value": 7}},
                metrics={"accuracy": 0.91},
            ),
        )

        consumer.inputs = [
            TaskInputSpec(name="summary", from_ref="produce.result.summary"),
            TaskInputSpec(name="text", from_ref="produce.result.text"),
            TaskInputSpec(name="data_key", from_ref="produce.result.data.nested.value"),
            TaskInputSpec(name="metric", from_ref="produce.result.metrics.accuracy"),
        ]
        resolved = materialize_task_inputs(wf, consumer, store)
        assert [item.resolved_type for item in resolved] == ["text", "text", "json", "json"]
        assert resolved[0].value == "short"
        assert resolved[1].value == "hello world"
        assert resolved[2].value == 7
        assert resolved[3].value == 0.91

    def test_missing_nested_result_key_becomes_missing(self, store):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        consumer = wf.task_by_label("consume")
        store.save_workflow(wf)
        store.write_task_output(
            wf.workflow_id,
            producer.task_id,
            TaskOutput(
                task_id=producer.task_id,
                workflow_id=wf.workflow_id,
                status="succeeded",
                summary="short",
                text="hello",
                data={},
                metrics={},
            ),
        )
        consumer.inputs = [
            TaskInputSpec(name="missing_data", from_ref="produce.result.data.ghost"),
            TaskInputSpec(name="missing_metric", from_ref="produce.result.metrics.accuracy"),
        ]
        resolved = materialize_task_inputs(wf, consumer, store)
        assert all(item.resolved_type == "missing" for item in resolved)

    def test_resolves_stdout_and_stderr_with_truncation_metadata(self, store):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        stdout_path, stderr_path = store.task_log_paths(wf.workflow_id, producer.task_id)
        stdout_path.write_text("x" * 5000, encoding="utf-8")
        stderr_path.write_text("err", encoding="utf-8")
        producer.log_stdout_path = str(stdout_path)
        producer.log_stderr_path = str(stderr_path)
        consumer = wf.task_by_label("consume")
        consumer.inputs = [
            TaskInputSpec(name="stdout", from_ref="produce.stdout"),
            TaskInputSpec(name="stderr", from_ref="produce.stderr"),
        ]
        resolved = materialize_task_inputs(wf, consumer, store)
        assert resolved[0].resolved_type == "text"
        assert resolved[0].metadata["truncated"] is True
        assert resolved[0].metadata["truncated_size"] == 4096
        assert len(resolved[0].value) == 4096
        assert resolved[1].value == "err"
        assert resolved[1].metadata["truncated"] is False

    def test_resolves_artifact_exact_match_only(self, store, tmp_path):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        consumer = wf.task_by_label("consume")
        artifact_path = store.task_artifacts_dir(wf.workflow_id, producer.task_id) / "report.json"
        artifact_path.write_text("{}", encoding="utf-8")
        producer.artifacts = [
            Artifact(
                artifact_id="art-1",
                workflow_id=wf.workflow_id,
                task_id=producer.task_id,
                name="report",
                kind="json",
                path=str(artifact_path),
            )
        ]
        consumer.inputs = [
            TaskInputSpec(name="artifact", from_ref="produce.artifact.report"),
            TaskInputSpec(name="missing", from_ref="produce.artifact.nope"),
        ]
        resolved = materialize_task_inputs(wf, consumer, store)
        assert resolved[0].resolved_type == "artifact"
        assert resolved[0].artifact_path == str(artifact_path)
        assert resolved[1].resolved_type == "missing"

    def test_resolves_status_for_succeeded_and_skipped_tasks(self, store):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        consumer = wf.task_by_label("consume")
        consumer.inputs = [
            TaskInputSpec(name="producer_status", from_ref="produce.status"),
        ]

        producer.status = TaskStatus.SUCCEEDED
        resolved = materialize_task_inputs(wf, consumer, store)
        assert resolved[0].resolved_type == "text"
        assert resolved[0].value == "succeeded"

        producer.status = TaskStatus.SKIPPED
        resolved = materialize_task_inputs(wf, consumer, store)
        assert resolved[0].resolved_type == "text"
        assert resolved[0].value == "skipped"

    def test_resolves_branch_context_task_fields(self, store):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        consumer = wf.task_by_label("consume")
        producer.status = TaskStatus.SKIPPED
        producer.condition_result = {"passed": False, "reason": "condition false"}
        producer.skip_reason = "condition evaluated false"
        consumer.inputs = [
            TaskInputSpec(name="producer_status", from_ref="produce.status"),
            TaskInputSpec(name="producer_condition", from_ref="produce.condition_result"),
            TaskInputSpec(name="producer_skip_reason", from_ref="produce.skip_reason"),
        ]

        resolved = materialize_task_inputs(wf, consumer, store)

        assert [item.resolved_type for item in resolved] == ["text", "json", "text"]
        assert resolved[0].value == "skipped"
        assert resolved[1].value == {"passed": False, "reason": "condition false"}
        assert resolved[2].value == "condition evaluated false"

    def test_branch_context_optional_fields_resolve_null_when_absent(self, store):
        wf = _workflow()
        producer = wf.task_by_label("produce")
        consumer = wf.task_by_label("consume")
        producer.status = TaskStatus.SUCCEEDED
        producer.condition_result = None
        producer.skip_reason = None
        consumer.inputs = [
            TaskInputSpec(name="producer_condition", from_ref="produce.condition_result"),
            TaskInputSpec(name="producer_skip_reason", from_ref="produce.skip_reason"),
        ]

        resolved = materialize_task_inputs(wf, consumer, store)

        assert [item.resolved_type for item in resolved] == ["json", "json"]
        assert resolved[0].value is None
        assert resolved[1].value is None

    def test_missing_task_label_still_fails(self, store):
        wf = _workflow()
        consumer = wf.task_by_label("consume")
        consumer.inputs = [
            TaskInputSpec(name="unknown_status", from_ref="unknown.status"),
        ]

        resolved = materialize_task_inputs(wf, consumer, store)

        assert resolved[0].resolved_type == "missing"
        assert resolved[0].metadata["reason"] == "unknown_label"

    def test_normal_result_refs_still_fail_when_output_missing(self, store):
        wf = _workflow()
        consumer = wf.task_by_label("consume")
        consumer.inputs = [
            TaskInputSpec(name="producer_text", from_ref="produce.result.text"),
        ]

        resolved = materialize_task_inputs(wf, consumer, store)

        assert resolved[0].resolved_type == "missing"
        assert resolved[0].metadata["reason"] == "missing_output"

    def test_invalid_label_and_invalid_path(self):
        with pytest.raises(Exception):
            parse_input_reference("result.text")
        with pytest.raises(Exception):
            parse_input_reference("produce.nope")


class TestPromptMaterialization:
    def test_build_materialized_prompt_is_deterministic(self):
        prompt = build_materialized_prompt(
            "Summarize the upstream text.",
            [
                ResolvedTaskInput(
                    name="producer_text",
                    source="produce.result.text",
                    resolved_task_id="tk-1",
                    resolved_type="text",
                    value="hello world",
                ),
                ResolvedTaskInput(
                    name="report",
                    source="produce.artifact.report",
                    resolved_task_id="tk-1",
                    resolved_type="artifact",
                    artifact_path="/tmp/report.json",
                ),
            ],
        )
        assert "Summarize the upstream text." in prompt
        assert "[Workflow Inputs]" in prompt
        assert "Input: producer_text" in prompt
        assert "Source: produce.result.text" in prompt
        assert "Type: text" in prompt
        assert "Value:\nhello world" in prompt
        assert "Type: artifact" in prompt
        assert "Value:\n/tmp/report.json" in prompt
