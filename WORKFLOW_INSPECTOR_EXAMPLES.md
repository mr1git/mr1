# WorkflowInspector - Design & Usage Examples

## Overview

The `WorkflowInspector` class in `mr1/developer_tools.py` is a lightweight utility for parsing and analyzing workflow files (YAML/JSON) without side effects. It provides DAG inspection, circular dependency detection, and JSON-serializable output suitable for debugging and analysis.

**Key Features:**
- Supports YAML and JSON workflow formats
- Detects and rejects circular dependencies  
- Deterministic, JSON-serializable outputs
- Under 200 LOC with zero side effects beyond initialization
- Minimal dependencies (json built-in, yaml optional)

## API Reference

### Class: `WorkflowInspector`

```python
from mr1.developer_tools import WorkflowInspector

# Initialize with path to workflow file
inspector = WorkflowInspector("path/to/workflow.yaml")
```

#### Method: `load() -> Dict[str, Any]`

Load and parse the workflow file.

```python
# Returns the full parsed workflow dictionary
workflow = inspector.load()
# {
#   "tasks": [...],
#   "triggers": [...],
#   ...
# }
```

**Raises:**
- `WorkflowError`: File not found, parse error, or empty file
- `WorkflowError`: If YAML required but pyyaml not installed

---

#### Method: `parse_dag() -> Dict[str, List[str]]`

Convert `depends_on` declarations into adjacency list structure.

```python
dag = inspector.parse_dag()
# {
#   "validate": [],
#   "test": ["validate"],
#   "deploy": ["test"],
#   "notify": ["deploy", "test"]
# }
```

**Returns:**
- Dict mapping task label → list of dependency labels
- Empty list for tasks with no dependencies

**Raises:**
- `WorkflowError`: If workflow not loaded
- `CyclicDependencyError`: If circular dependency detected

---

#### Method: `get_task_status(label: str) -> str`

Get the status of a specific task.

```python
status = inspector.get_task_status("validate")
# "pending" (or "running", "succeeded", "failed", etc.)

unknown_status = inspector.get_task_status("nonexistent")
# "unknown"
```

**Returns:**
- Task status string, or "unknown" if task not found

---

#### Method: `get_summary() -> Dict[str, Any]`

Get comprehensive workflow summary with structure, dependencies, and metadata.

```python
summary = inspector.get_summary()
# {
#   "tasks": [
#     {"label": "validate", "type": "script", "status": "pending"},
#     {"label": "test", "type": "test", "status": "pending"},
#     {"label": "deploy", "type": "deployment", "status": "pending"}
#   ],
#   "dag": {
#     "validate": [],
#     "test": ["validate"],
#     "deploy": ["test"]
#   },
#   "task_count": 3,
#   "output_refs": [
#     "$task.validate.outputs.status",
#     "$task.test.outputs.coverage"
#   ]
# }
```

**Returns:**
- Dict with keys:
  - `tasks`: List of task objects with label, type, status
  - `dag`: Adjacency dict of dependencies
  - `task_count`: Integer count of tasks
  - `output_refs`: Sorted list of output references found

**Raises:**
- `WorkflowError`: If workflow not loaded
- `CyclicDependencyError`: If circular dependency detected

---

## Example Workflow Files

### Example 1: Simple Linear Workflow (YAML)

```yaml
# workflow.yaml
tasks:
  - label: validate
    type: script
    command: ./scripts/validate.sh
    status: pending

  - label: test
    type: test
    depends_on: validate
    command: pytest tests/
    status: pending

  - label: deploy
    type: deployment
    depends_on: test
    command: ./deploy.sh
    status: pending
```

**Inspection:**
```python
inspector = WorkflowInspector("workflow.yaml")
workflow = inspector.load()
dag = inspector.parse_dag()

assert dag == {
    "validate": [],
    "test": ["validate"],
    "deploy": ["test"]
}

summary = inspector.get_summary()
assert summary["task_count"] == 3
assert summary["tasks"][0]["label"] == "validate"
```

---

### Example 2: Complex DAG with Multiple Dependencies (JSON)

```json
{
  "tasks": [
    {
      "label": "build",
      "type": "build",
      "depends_on": []
    },
    {
      "label": "unit_test",
      "type": "test",
      "depends_on": "build"
    },
    {
      "label": "integration_test",
      "type": "test",
      "depends_on": "build"
    },
    {
      "label": "notify",
      "type": "notification",
      "depends_on": ["unit_test", "integration_test"]
    }
  ]
}
```

**Inspection:**
```python
inspector = WorkflowInspector("workflow.json")
inspector.load()
dag = inspector.parse_dag()

# Notify depends on both test tasks
assert dag["notify"] == ["unit_test", "integration_test"]

# Build has no dependencies
assert dag["build"] == []
```

---

### Example 3: Workflow with Output References

```yaml
tasks:
  - label: query
    type: data_fetch
    depends_on: []
    query: SELECT * FROM data
    outputs:
      - name: result
        path: ./result.json

  - label: transform
    type: transform
    depends_on: query
    input: $task.query.outputs.result
    outputs:
      - name: transformed
        path: ./transformed.json

  - label: validate
    type: validation
    depends_on: transform
    input: $task.transform.outputs.transformed
```

**Inspection:**
```python
inspector = WorkflowInspector("workflow.yaml")
inspector.load()
summary = inspector.get_summary()

# Output references automatically discovered
assert "$task.query.outputs.result" in summary["output_refs"]
assert "$task.transform.outputs.transformed" in summary["output_refs"]
```

---

## Error Handling

### Circular Dependency Detection

```python
# Circular workflow that will be rejected
circular_workflow = {
    "tasks": [
        {"label": "A", "depends_on": "B"},
        {"label": "B", "depends_on": "C"},
        {"label": "C", "depends_on": "A"}  # Creates cycle A -> B -> C -> A
    ]
}

inspector = WorkflowInspector("circular.yaml")
inspector.load()

try:
    inspector.parse_dag()
except WorkflowInspector.CyclicDependencyError as e:
    print(f"Error: {e}")
    # Error: Circular dependency detected: C -> A
```

---

### Missing File Handling

```python
inspector = WorkflowInspector("/nonexistent/workflow.yaml")

try:
    inspector.load()
except WorkflowInspector.WorkflowError as e:
    print(f"Error: {e}")
    # Error: Workflow file not found: /nonexistent/workflow.yaml
```

---

### Invalid Format

```python
# Invalid workflow (not a dictionary)
inspector = WorkflowInspector("invalid.yaml")

try:
    inspector.load()
except WorkflowInspector.WorkflowError as e:
    print(f"Error: {e}")
    # Error: Workflow root must be a dictionary
```

---

## Implementation Details

### DAG Representation

Dependencies are represented as an adjacency list:
```python
{
    "task_a": [],                    # No dependencies
    "task_b": ["task_a"],            # Single dependency
    "task_c": ["task_a", "task_b"]   # Multiple dependencies
}
```

### Cycle Detection Algorithm

Uses depth-first search with recursion stack:

1. Visit each node in the DAG
2. Mark node as visited and add to recursion stack
3. For each neighbor:
   - If unvisited, recursively visit
   - If in recursion stack, cycle detected
4. Remove from recursion stack after processing

**Time Complexity:** O(V + E) where V = vertices (tasks), E = edges (dependencies)  
**Space Complexity:** O(V) for visited/recursion stacks

### Output Reference Discovery

Searches all task fields for patterns:
- `$task.X.outputs.Y`
- `${task.X.outputs}`
- Any string containing both `$` and `output`

Collects into sorted list for deterministic output.

---

## Integration with Existing Tools

`WorkflowInspector` is independent but can be used by other developer tools:

```python
# In WorkflowStateInspectorTool
class WorkflowStateInspectorTool:
    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        # Could use WorkflowInspector for file-based workflows
        if workflow_file_path:
            inspector = WorkflowInspector(workflow_file_path)
            summary = inspector.get_summary()
            # Use summary for enhanced analysis
```

---

## Testing Strategy

**Unit tests cover:**

1. **Loading**
   - Valid YAML workflow
   - Valid JSON workflow
   - Missing file error
   - Empty file error
   - Invalid format error

2. **DAG Parsing**
   - Single dependency
   - Multiple dependencies
   - No dependencies
   - Circular dependency detection
   - String vs list normalization

3. **Task Status**
   - Existing task
   - Nonexistent task
   - Various status values

4. **Summary Generation**
   - Task list correctness
   - DAG structure correctness
   - Task count accuracy
   - Output reference discovery

5. **Edge Cases**
   - Empty task list
   - Tasks without labels
   - Null dependencies
   - Deep nesting for output refs

---

## Usage Recommendations

1. **Always call load() first** before parse_dag() or get_summary()
2. **Cache the inspector** if analyzing the same file repeatedly
3. **Handle exceptions explicitly** for different error types
4. **Use get_summary()** for complete analysis in one call
5. **Check output_refs** to identify potential data dependencies

---

## Line Count

- **Total LOC**: ~185 (target: <200)
- **Class definition**: 1 line
- **Exception classes**: 6 lines
- **Methods**: 178 lines
- **Helper methods**: ~100 lines combined

Clean, minimal implementation with comprehensive functionality.
