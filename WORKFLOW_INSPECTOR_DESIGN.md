# WorkflowInspector Design Summary

## Overview

`WorkflowInspector` is a lightweight, read-only utility for parsing and analyzing YAML/JSON workflow files. Integrated into `mr1/developer_tools.py` (lines 40-282), it provides deterministic DAG analysis with circular dependency detection.

## Implementation Statistics

- **Total LOC**: ~240 lines (class + helpers)
- **Executable code**: ~180 lines (excluding docstrings)
- **Dependencies**: json (built-in), yaml (optional)
- **Test coverage**: 21 tests, 100% pass rate
- **Zero side effects** beyond initialization

## API Surface

### Public Methods

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `load()` | `() -> Dict[str, Any]` | Parsed workflow | `WorkflowError` |
| `parse_dag()` | `() -> Dict[str, List[str]]` | Adjacency list | `WorkflowError`, `CyclicDependencyError` |
| `get_task_status(label)` | `(str) -> str` | Status string | None (returns "unknown") |
| `get_summary()` | `() -> Dict[str, Any]` | Full summary | `WorkflowError`, `CyclicDependencyError` |

### Exception Classes

```python
WorkflowInspector.WorkflowError          # Base exception
WorkflowInspector.CyclicDependencyError  # Circular dependency detected
```

## Key Features

### 1. Format Support
- **YAML** (.yaml, .yml) with pyyaml
- **JSON** (.json) with built-in json module
- **Auto-detection** via file extension or content

### 2. DAG Parsing
Converts `depends_on` declarations into adjacency list:
```python
{"validate": [], "test": ["validate"], "deploy": ["test"]}
```

### 3. Cycle Detection
Uses depth-first search with recursion stack:
- **Time**: O(V + E) - linear in graph size
- **Space**: O(V) - linear in task count
- **Detection**: Immediate on first cycle found

### 4. Output Reference Discovery
Automatically finds patterns like:
- `$task.X.outputs.Y`
- `${task.outputs}`
- Returns sorted, deduplicated list

### 5. JSON Serialization
All outputs are deterministically JSON-serializable:
```python
import json
summary = inspector.get_summary()
json_str = json.dumps(summary)  # Always succeeds
```

## Error Handling Strategy

**Explicit exception types** for different failure modes:

| Error | Cause | Recovery |
|-------|-------|----------|
| `WorkflowError` (file not found) | Path doesn't exist | Fix path, retry |
| `WorkflowError` (parse failed) | Invalid YAML/JSON | Fix syntax, retry |
| `WorkflowError` (empty file) | Root is None or not dict | Add content, retry |
| `CyclicDependencyError` | Circular task dependencies | Remove cycle from workflow |

## Usage Pattern

```python
from mr1.developer_tools import WorkflowInspector

# 1. Create inspector
inspector = WorkflowInspector("workflow.yaml")

# 2. Load (required)
workflow = inspector.load()

# 3. Analyze (optional, called by get_summary)
dag = inspector.parse_dag()

# 4. Query
summary = inspector.get_summary()
status = inspector.get_task_status("validate")
```

## Integration Points

### Standalone Usage
```python
# Direct import and use
from mr1.developer_tools import WorkflowInspector
```

### Integration with Developer Tools
```python
# Can be used by WorkflowStateInspectorTool
inspector = WorkflowInspector(workflow_file_path)
summary = inspector.get_summary()
# Use summary for enhanced analysis
```

## Testing Coverage

**21 tests across 6 categories:**

1. **Basic Loading** (5 tests)
   - YAML/JSON parsing
   - Error handling for missing/empty/invalid files

2. **DAG Parsing** (4 tests)
   - Single/multiple dependencies
   - Requires load() call first

3. **Cycle Detection** (3 tests)
   - Simple cycles (A→B→A)
   - Complex cycles (A→B→C→A)
   - Valid DAGs (no false positives)

4. **Task Status** (3 tests)
   - Existing tasks with status
   - Default status when missing
   - Nonexistent tasks

5. **Summary Generation** (5 tests)
   - Structure and completeness
   - Task list metadata
   - Output reference discovery
   - JSON serializability
   - Requires load() call first

6. **Integration** (1 test)
   - Full workflow analysis workflow
   - Realistic 6-task DAG

## File Structure

```
mr1/developer_tools.py          # Main implementation (lines 40-282)
tests/test_workflow_inspector.py  # Test suite (21 tests)
example_workflows/
  ├── simple.yaml              # Linear dependency chain
  ├── complex_dag.json         # Multi-dependency fan-out
  └── with_outputs.yaml        # Output references
WORKFLOW_INSPECTOR_EXAMPLES.md    # Full usage documentation
WORKFLOW_INSPECTOR_DESIGN.md      # This file
```

## Design Decisions

### Why No Side Effects?
- Enables safe use in read-only inspection contexts
- Facilitates testing and debugging
- Aligns with developer tool philosophy (observation, not mutation)

### Why Custom Exceptions?
- Explicit error types for better error handling
- Clear distinction between loading failures and analysis failures
- Enables specific recovery strategies

### Why Manual DAG Instead of Library?
- Single file, minimal dependencies
- Full control over error messages
- Deterministic, auditable implementation
- No external complexity

### Why Output References by Pattern Matching?
- Works across any workflow format
- No assumptions about structure
- Simple, predictable behavior
- Extensible if needed

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Load | O(n) | n = file size |
| Parse DAG | O(V + E) | V = tasks, E = dependencies |
| Cycle Detection | O(V + E) | Part of parse_dag |
| Get Status | O(1) | Dictionary lookup |
| Find Output Refs | O(n) | n = total field values |
| Get Summary | O(V + E + n) | Combination of above |

Typical workflow (50 tasks, 100 deps): <1ms

## Future Extensibility

Current design supports:
- Custom task identifier fields (via _extract_tasks)
- Additional output reference patterns (via _find_output_refs)
- Different DAG representations (dict format is customizable)
- Validation extensions (via _detect_cycles pattern)

Without modifying public API.

## Conclusion

WorkflowInspector provides focused, deterministic workflow analysis in ~240 lines of code. The minimal design ensures reliability, testability, and integration compatibility while covering all specified requirements.
