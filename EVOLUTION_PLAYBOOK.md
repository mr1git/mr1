# Evolution Agent Playbook

## Overview
This playbook guides the evolution agent through each iteration of the tool development cycle. Follow this for consistent, measurable tool evolution.

---

## Phase 1: Analyze (15-20 min)

### Input
- Previous iteration report (if any)
- Current system state
- Evolution history

### Steps

1. **Review Previous Performance** (if iteration > 1)
   ```
   - What was the success rate? (% of proposed tools that passed tests)
   - What patterns failed? What patterns succeeded?
   - What improvements were planned? Were they acted on?
   - What was the avg test pass rate? Avg code coverage?
   ```

2. **Identify System Gaps**
   - Scan `mr1/tools.py` and `mr1/new_tools.py` for missing capabilities
   - Look at failed tool proposals from previous iterations
   - Check for common patterns in tool failures
   - Ask: "What would make the system more capable?"

3. **Analyze Failure Signatures** (if iteration > 1)
   - Tools that failed to implement: Why? (too complex? unclear design?)
   - Tools with low test pass rates: Why? (edge cases? validation issues?)
   - Tools with low code coverage: Why? (untestable design? vague requirements?)
   - **Goal**: Extract 2-3 specific improvements for Phase 2

4. **Output**: Iteration Analysis Report
   - System gaps identified
   - Previous patterns (what worked/failed)
   - Improvement hypotheses
   - Proposal direction

---

## Phase 2: Propose (20-30 min)

### Input
- Iteration Analysis Report
- Tool proposal template
- Design patterns from previous iterations

### Steps

1. **Generate 3-5 Tool Ideas**
   - Draw from system gaps identified in Phase 1
   - Each tool should address a clear gap or improve a previous tool
   - Mix difficulty levels (1-2 low, 1-2 medium, 0-1 high)

2. **For Each Tool: Write Proposal**
   ```markdown
   # Purpose
   - 1-2 sentences explaining what problem it solves
   
   # Design Decisions
   - Why this approach vs. alternatives?
   - What did we learn from previous tools?
   
   # Config Schema
   - Minimal, unambiguous
   - Test with 3 example configs
   
   # Test Plan
   - How will we verify it works?
   - What edge cases matter?
   - What's the success bar?
   
   # Estimated Complexity
   - How many lines of code?
   - How many tests needed?
   - Any novel logic?
   ```

3. **Rank by Impact**
   - High impact: solves major gap, unblocks other tools
   - Medium impact: improves existing capability
   - Low impact: nice-to-have, educational

4. **Select Top 3 for This Iteration**
   - Aim for mix of difficulty (easier to validate faster + harder for learning)
   - Balance implementation time (don't overcommit)

5. **Output**: Formal Proposals
   - 3 ranked proposals with detailed designs
   - Rationale for selection
   - Success criteria for each tool

---

## Phase 3: Implement (Varies: 30-90 min total)

### Input
- Formal proposals
- Patterns from successful previous tools
- Implementation guidelines

### Steps

1. **For Each Tool: Write Implementation**
   ```
   - Follow MR1 tool pattern from developer_tools.py or new_tools.py
   - Stubs for ToolRunner protocol
   - validate_config() method with strict checks
   - run() method with comprehensive error handling
   - At least 5-10 test cases prepared
   ```

2. **Track Implementation Details**
   ```python
   start_time = now()
   # implement tool
   impl_time = now() - start_time
   ```

3. **Code Quality Checks**
   - Does it follow existing patterns?
   - Are error messages clear?
   - Is config validation strict?
   - Any edge cases missed?

4. **Output**: Implementation Code + Test Stubs
   - Tool class in `mr1/` directory
   - Test file with 10-20 test cases
   - Implementation time recorded

---

## Phase 4: Test (Varies: 20-60 min)

### Input
- Implementation code
- Test stubs
- Evolution history (for comparison)

### Steps

1. **Run Tests**
   ```bash
   pytest tests/test_<tool_name>.py -v
   ```

2. **Measure Coverage**
   ```bash
   pytest tests/test_<tool_name>.py --cov=mr1.<tool_module> --cov-report=term-out
   ```

3. **Record Metrics**
   ```python
   for each tool:
     - test_count (how many tests?)
     - test_pass_count (how many passed?)
     - code_coverage_pct (% of code covered)
     - lines_of_code (implementation size)
     - bugs_escaped (defects found post-test)
   ```

4. **Analyze Patterns**
   - Which tests caught the most bugs?
   - Are there common validation gaps?
   - Did config schema catch the errors we expected?
   - What test types are missing?

5. **Output**: Test Results + Coverage Report
   - Pass/fail for each test
   - Coverage report
   - Bug escape analysis

---

## Phase 5: Evaluate & Report (20-30 min)

### Input
- All metrics from implementation and testing
- Evolution history
- Patterns from this iteration

### Steps

1. **Calculate Metrics**
   ```python
   For each tool:
     - Test pass rate = (test_pass_count / test_count) * 100
     - Code coverage = measured from pytest
     - Implementation minutes = tracked during Phase 3
     - Bugs escaped = defects found after tests pass
   
   For iteration:
     - Success rate = (tools_tested / tools_proposed) * 100
     - Avg test pass rate = mean of all pass rates
     - Avg code coverage = mean of all coverage %
   ```

2. **Identify Patterns Discovered**
   ```
   - Did tools with config validation catch more bugs than those without?
   - Do certain test types correlate with low bug escape rate?
   - Are tools easier/harder to implement than previous similar tools?
   - Did design patterns from previous iterations help or hurt?
   - What failure modes appeared again?
   ```

3. **Extract Learning**
   ```
   - 2-3 things that went well (repeat next iteration)
   - 2-3 things that went poorly (improve next iteration)
   - 1-2 new patterns discovered (validate in future)
   - 1-2 hypotheses to test next iteration
   ```

4. **Write Iteration Report** (use ITERATION_TEMPLATE)
   - Status summary (proposed/implemented/tested/failed)
   - Metrics table
   - Detailed tool summaries
   - Patterns discovered
   - Improvements for next iteration

5. **Update Evolution History**
   ```python
   history = EvolutionHistory()
   history.add_report(iteration_report)
   
   # Analyze trends
   trends = history.trend_analysis()
   patterns = history.pattern_analysis()
   ```

6. **Output**: Iteration Report + Historical Analysis
   - This iteration's report (saved to file)
   - Trend analysis (improving? regressing?)
   - Pattern analysis (recurring issues?)

---

## Phase 6: Plan Next Iteration (5-10 min)

### Input
- Current iteration report
- Trend and pattern analysis
- Constraints (time, complexity budget)

### Steps

1. **Identify Improvements**
   - Based on failure analysis, what should we change?
   - Are there design patterns we should adopt?
   - Should we propose similar tools (for consistency) or diverse tools?

2. **Set Hypotheses for Next Iteration**
   ```
   - "If we add stricter config validation, bug escape rate will drop by 50%"
   - "If we write tests first, implementation time will increase but quality will improve"
   - "If we reuse patterns from tool X, tool Y will be faster to implement"
   ```

3. **Document Recommendations**
   - Specific improvements to try next iteration
   - Hypotheses to validate
   - Metrics to focus on

4. **Output**: Next Iteration Preparation
   - Saved to iteration report as "improvements_for_next"
   - Ready for evolution agent to consume in Phase 1 of next iteration

---

## Checklist: Running One Complete Iteration

- [ ] **Phase 1**: Review previous iteration, identify gaps, analyze failures
- [ ] **Phase 2**: Propose 3-5 tools with detailed designs
- [ ] **Phase 3**: Implement selected tools with time tracking
- [ ] **Phase 4**: Test all tools, measure coverage
- [ ] **Phase 5**: Calculate metrics, identify patterns, write report
- [ ] **Phase 6**: Plan next iteration improvements
- [ ] **Output**: Iteration report + updated history file

---

## Success Criteria per Iteration

| Metric | Acceptable | Good | Excellent |
|--------|-----------|------|-----------|
| Tools Tested | ≥1 | ≥2 | ≥3 |
| Avg Test Pass Rate | ≥70% | ≥85% | ≥95% |
| Avg Code Coverage | ≥60% | ≥75% | ≥85% |
| Success Rate | ≥50% | ≥70% | ≥90% |
| Bugs Escaped (avg) | ≤2 | ≤1 | 0 |

---

## Long-Term Learning Targets

**After 5 iterations:**
- Repeatable patterns established
- Test strategies that work identified
- Configuration validation best practices learned
- Implementation velocity measured and optimized

**After 10 iterations:**
- Able to predict tool difficulty accurately
- Failure modes catalogued and prevented
- Design patterns standardized
- Self-improvement becoming autonomous

**After 20+ iterations:**
- System approaching steady state
- New tools rarely fail
- Able to propose improvements to own tools
- Learning rate stable or increasing
