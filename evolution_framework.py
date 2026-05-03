"""
Evolution framework: utilities for the tool evolution agent.

Provides templates, metrics tracking, and analysis for iterative tool development.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json


@dataclass
class ToolMetrics:
    """Metrics for a single tool implementation."""
    tool_name: str
    iteration: int
    proposed_at: str
    implemented_at: Optional[str] = None
    tested_at: Optional[str] = None
    test_count: int = 0
    test_pass_count: int = 0
    code_coverage_pct: float = 0.0
    lines_of_code: int = 0
    implementation_minutes: Optional[float] = None
    bugs_escaped: int = 0  # defects found after implementation
    status: str = "proposed"  # proposed|implemented|tested|released|failed
    failure_reason: Optional[str] = None
    notes: str = ""

    @property
    def test_pass_rate(self) -> float:
        """Pass rate as percentage."""
        if self.test_count == 0:
            return 0.0
        return (self.test_pass_count / self.test_count) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolMetrics:
        return cls(**data)


@dataclass
class IterationReport:
    """Report for one evolution cycle."""
    iteration: int
    started_at: str
    completed_at: Optional[str] = None
    tools_proposed: list[str] = field(default_factory=list)
    tools_implemented: list[str] = field(default_factory=list)
    tools_tested: list[str] = field(default_factory=list)
    tools_failed: list[str] = field(default_factory=list)
    metrics: dict[str, ToolMetrics] = field(default_factory=dict)
    patterns_discovered: list[str] = field(default_factory=list)
    improvements_for_next: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def success_rate(self) -> float:
        """% of proposed tools that passed tests."""
        if not self.tools_proposed:
            return 0.0
        return (len(self.tools_tested) / len(self.tools_proposed)) * 100.0

    @property
    def avg_test_pass_rate(self) -> float:
        """Average test pass rate across all tools."""
        if not self.metrics:
            return 0.0
        pass_rates = [m.test_pass_rate for m in self.metrics.values()]
        return sum(pass_rates) / len(pass_rates)

    @property
    def avg_code_coverage(self) -> float:
        """Average code coverage across all tools."""
        if not self.metrics:
            return 0.0
        coverages = [m.code_coverage_pct for m in self.metrics.values()]
        return sum(coverages) / len(coverages)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metrics"] = {k: v.to_dict() for k, v in self.metrics.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterationReport:
        metrics = {k: ToolMetrics.from_dict(v) for k, v in data.pop("metrics", {}).items()}
        report = cls(**data)
        report.metrics = metrics
        return report


class EvolutionHistory:
    """Tracks evolution metrics across iterations."""

    def __init__(self, history_path: Path = Path(".mr1/evolution_history.jsonl")):
        self.history_path = history_path
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports: list[IterationReport] = []
        self._load()

    def _load(self) -> None:
        """Load existing history."""
        if not self.history_path.exists():
            return
        with open(self.history_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.reports.append(IterationReport.from_dict(data))

    def add_report(self, report: IterationReport) -> None:
        """Add a new iteration report."""
        self.reports.append(report)
        with open(self.history_path, "a") as f:
            f.write(json.dumps(report.to_dict()) + "\n")

    def latest_iteration(self) -> Optional[IterationReport]:
        """Get the most recent iteration."""
        return self.reports[-1] if self.reports else None

    def get_iteration(self, num: int) -> Optional[IterationReport]:
        """Get a specific iteration by number."""
        for report in self.reports:
            if report.iteration == num:
                return report
        return None

    def trend_analysis(self) -> dict[str, Any]:
        """Analyze trends across iterations."""
        if len(self.reports) < 2:
            return {"message": "Need at least 2 iterations for trend analysis"}

        success_rates = [r.success_rate for r in self.reports]
        avg_pass_rates = [r.avg_test_pass_rate for r in self.reports]
        avg_coverages = [r.avg_code_coverage for r in self.reports]

        return {
            "total_iterations": len(self.reports),
            "success_rate_trend": success_rates,
            "success_rate_improving": success_rates[-1] > success_rates[0],
            "test_pass_rate_trend": avg_pass_rates,
            "test_pass_rate_improving": avg_pass_rates[-1] > avg_pass_rates[0],
            "code_coverage_trend": avg_coverages,
            "code_coverage_improving": avg_coverages[-1] > avg_coverages[0],
            "total_tools_created": sum(len(r.tools_tested) for r in self.reports),
            "total_tools_failed": sum(len(r.tools_failed) for r in self.reports),
        }

    def pattern_analysis(self) -> dict[str, Any]:
        """Analyze patterns discovered across iterations."""
        all_patterns: dict[str, int] = {}
        for report in self.reports:
            for pattern in report.patterns_discovered:
                all_patterns[pattern] = all_patterns.get(pattern, 0) + 1

        return {
            "total_patterns_discovered": len(all_patterns),
            "recurring_patterns": {k: v for k, v in all_patterns.items() if v > 1},
            "patterns_by_iteration": [
                {
                    "iteration": r.iteration,
                    "patterns": r.patterns_discovered,
                    "count": len(r.patterns_discovered),
                }
                for r in self.reports
            ],
        }


# ============================================================================
# Templates for the evolution agent
# ============================================================================

TOOL_PROPOSAL_TEMPLATE = """
# Tool Proposal: {tool_name}

## Purpose
{purpose}

## Type
{tool_type}  # orchestration | general_purpose | developer | experimental

## Rationale
{rationale}

## Example Use Case
{example_use_case}

## Estimated Complexity
{complexity}  # low | medium | high

## Config Schema
```json
{config_schema}
```

## Test Plan
- [ ] Validate config errors caught properly
- [ ] Success path works as expected
- [ ] Edge cases handled (empty input, missing fields, etc.)
- [ ] Error messages are clear
- [ ] Performance acceptable ({target_time}ms for typical input)

## Risks
{risks}

## Success Criteria
{success_criteria}
"""

ITERATION_TEMPLATE = """
# Evolution Iteration {iteration}

## Status
- **Proposed**: {proposed_count} tools
- **Implemented**: {implemented_count} tools
- **Tested**: {tested_count} tools
- **Failed**: {failed_count} tools

## Success Rate
{success_rate}% of proposed tools passed testing

## Metrics
| Metric | Value |
|--------|-------|
| Avg Test Pass Rate | {avg_test_pass}% |
| Avg Code Coverage | {avg_coverage}% |
| Total Lines of Code | {total_loc} |
| Total Tests Written | {total_tests} |

## Tools Created This Iteration
{tools_summary}

## Patterns Discovered
{patterns}

## Learning & Improvements for Next Iteration
{improvements}

## Detailed Tool Metrics
{detailed_metrics}
"""
