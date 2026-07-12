"""
Autonomous operation (Phase A).

A thin governed layer above the existing runtime. The scheduler still owns
execution; nothing in this package executes a task itself. What lives here:

  * `control`   — control.json, the modes, the service singleton lock
  * `health`    — heartbeat, gauges, health.json
  * `consent`   — objective-scoped standing consent grants
  * `recovery`  — the failure classifier and the bounded recovery ladder
  * `objectives`— persisted long-lived goals
  * `budget`    — shared planning/action budgets (supervisor and inbox triage)
  * `escalation`— "I need a human", delivered to the inbox and the timeline
  * `service`   — the headless supervisor host and its ordered tick
"""

from __future__ import annotations
