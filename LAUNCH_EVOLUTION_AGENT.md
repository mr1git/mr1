# Launching the Evolution Agent

## Quick Start

### Create a New Evolution Agent

**In your session:**
```
Route a message to MR2 or create a new persistent child agent with this mission:

You are the Evolution Agent for MR1. Your responsibility is to own the continuous tool 
evolution cycle:

1. Analyze system gaps and previous tool performance
2. Propose 3-5 new tools or tool improvements
3. Implement them with comprehensive tests
4. Evaluate metrics (test pass rate, code coverage, bugs escaped, implementation time)
5. Report findings and extract learning for the next iteration

Read EVOLUTION_AGENT_CHARTER.md for your full mission.
Follow EVOLUTION_PLAYBOOK.md for step-by-step instructions on each iteration.

Use the EvolutionHistory class in evolution_framework.py to track metrics.
```

### Run an Iteration Manually (Easiest to Start)

**Iteration 2 example:**
```
User: "Run iteration 2 of the evolution cycle. Analyze the iteration 1 report, 
propose 3 new tools, implement and test them, then report findings."

Agent: [Follows EVOLUTION_PLAYBOOK phases 1-6]
       [Writes iteration 2 report to .mr1/evolution_history.jsonl]
       [Reports back with metrics and learning]
```

### Set Up Autonomous Recurring Iterations

Once you've run 2-3 iterations manually and validated the process:
```
/schedule the evolution agent to run iteration N every Monday at 9am

Or: /loop "Run the next evolution iteration" every 7 days
```

---

## Understanding the Process

### The 6-Phase Cycle

```
┌─────────────┐
│   ANALYZE   │  Review previous iteration, identify system gaps
└──────┬──────┘
       ↓
┌─────────────┐
│  PROPOSE    │  Design 3-5 tools with detailed specs
└──────┬──────┘
       ↓
┌─────────────┐
│IMPLEMENT    │  Write tool classes and test stubs
└──────┬──────┘
       ↓
┌─────────────┐
│    TEST     │  Run tests, measure coverage, track bugs
└──────┬──────┘
       ↓
┌─────────────┐
│ EVALUATE    │  Calculate metrics, find patterns, write report
└──────┬──────┘
       ↓
┌─────────────┐
│    PLAN     │  Identify improvements for next iteration
└─────────────┘
```

Each phase has clear inputs, steps, and outputs. See EVOLUTION_PLAYBOOK.md for details.

### Key Metrics Tracked

**Per-Tool:**
- Test count and pass rate (%)
- Code coverage (%)
- Lines of code
- Implementation time (minutes)
- Bugs escaped (defects found after testing)

**Per-Iteration:**
- Success rate (% of proposed tools that passed testing)
- Average test pass rate
- Average code coverage
- Total lines of code written
- Total tests written

**Historical Trends:**
- Is quality improving over time?
- Is velocity improving (faster implementation)?
- Are we discovering and avoiding failure modes?

---

## Artifacts Created by Evolution Agent

### Per Iteration

**`.mr1/evolution_history.jsonl`**
- Append-only log of all iteration reports
- One JSON object per line per iteration
- Contains all metrics, patterns, improvements

**Iteration Report** (printed to stdout)
- Status summary
- Metrics table
- Tool summaries
- Patterns discovered
- Recommendations for next iteration

### Permanent Guidance

**Already Created:**
- `EVOLUTION_AGENT_CHARTER.md` — Full mission and responsibilities
- `EVOLUTION_PLAYBOOK.md` — Step-by-step instructions for each phase
- `evolution_framework.py` — Utilities, templates, metrics tracking
- `.mr1/evolution_history.jsonl` — Historical metrics (starts with iteration 1 baseline)

---

## Example: Running Iteration 2

**User message:**
```
Run iteration 2 of the evolution cycle. Use the playbook and historical metrics.
Propose tools that address the "improvements_for_next" from iteration 1.
```

**Agent would:**

1. **ANALYZE** (read from evolution_history.jsonl)
   - Iteration 1 success rate: 100%
   - Avg test pass rate: 88.1%
   - Patterns: read-only tools pass tests faster, strict config validation works
   - Improvements noted: focus on orchestration tools, measure velocity

2. **PROPOSE**
   - Tool 1: Task Dependency Resolver (medium complexity)
   - Tool 2: Conditional Executor (medium complexity)
   - Tool 3: Batch Aggregator (high complexity)
   - Rationale: addresses "orchestration tools are next priority"

3. **IMPLEMENT**
   - Write 3 tool classes (following patterns from iteration 1)
   - Track implementation time for each
   - Create ~30-40 test cases total

4. **TEST**
   - Run all tests
   - Measure coverage
   - Record metrics

5. **EVALUATE**
   - Calculate pass rates, coverage
   - Did orchestration tools take longer? (predict complexity)
   - Did patterns from iteration 1 help?
   - Extract 2-3 new patterns

6. **PLAN**
   - Based on results, what to try next iteration?
   - Any hypotheses to validate?

7. **REPORT**
   - Iteration 2 report with all metrics
   - Trends compared to iteration 1
   - Recommendations

---

## Monitoring Evolution

### After Each Iteration

Check the metrics:
```bash
tail -1 .mr1/evolution_history.jsonl | python -m json.tool
```

### Track Trends

Ask the evolution agent:
```
"Analyze the evolution history. Are we improving? What trends do you see?
What should we focus on next?"
```

The agent can use `EvolutionHistory.trend_analysis()` and `pattern_analysis()` 
to report on:
- Success rate trajectory (improving?)
- Test pass rate trend (getting better?)
- Code coverage evolution
- Recurring failure patterns
- Most valuable insights discovered

### Success Indicators

**After 5 iterations:**
- Success rate ≥70% (most tools pass tests)
- Avg test pass rate ≥85%
- Avg code coverage ≥75%
- Clear patterns emerging in what works/fails

**After 10 iterations:**
- Success rate ≥85%
- Avg test pass rate ≥90%
- Avg code coverage ≥80%
- Agent proposing targeted improvements

**After 20 iterations:**
- Success rate ≥95%
- System approaching quality plateau
- New failures are rare and novel
- Agent learning about itself

---

## Constraints & Safety

All tools proposed by the evolution agent must be:

✅ **Safe**
- Read-only by default
- No destructive operations
- Bounded (timeouts, limits)

✅ **Testable**
- Can be tested in isolation
- Clear success criteria
- No flaky tests

✅ **Auditable**
- Clear purpose and behavior
- Minimal side effects
- Transparent error handling

---

## Next Steps

1. **Review** `EVOLUTION_AGENT_CHARTER.md` — understand the mission
2. **Read** `EVOLUTION_PLAYBOOK.md` — understand each phase
3. **Check** `.mr1/evolution_history.jsonl` — see iteration 1 baseline
4. **Create** an evolution agent (persistent child) with the charter above
5. **Request** "Run iteration 2" to validate the process
6. **Monitor** metrics and trends after each iteration
7. **Adjust** playbook or process based on what you learn

---

## Questions for the Evolution Agent

After each iteration, ask:

- "Why did tool X pass immediately but tool Y needed fixes?"
- "What test patterns catch the most bugs?"
- "Are we getting faster at implementing tools?"
- "What design pattern should we standardize on?"
- "Should we try a different approach to config validation?"
- "Are there tool interactions we should explore?"
- "What would make us 10x better at tool development?"

These questions drive learning and continuous improvement.
