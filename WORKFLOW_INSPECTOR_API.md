# WorkflowInspector API Reference

## Class: `WorkflowInspector`

**Module**: `mr1.developer_tools`

**Purpose**: Parse and analyze YAML/JSON workflow files with DAG inspection and circular dependency detection.

---

## Constructor

```python
WorkflowInspector(workflow_path: str)
```

**Parameters:**
- `workflow_path` (str): Path to YAML/JSON workflow file

**Example:**
```python
from mr1.developer_tools import WorkflowInspector
inspector = WorkflowInspector("path/to/workflow.yaml")
```

**Notes:**
- Supports .yaml, .yml, and .json file extensions
- Path validation occurs during `load()`, not `__init__`
- No side effects during initialization

---

## Methods

### `load() -> Dict[str, Any]`

Load and parse the workflow file.

**Returns:**
- Dictionary containing parsed workflow data

**Raises:**
- `WorkflowError`: If file not found, parse fails, or is empty
- `WorkflowError`: If YAML required but pyyaml not installed

**Example:**
```python
workflow = inspector.load()
print(workflow.get('tasks'))  # List of task dicts
```

**Notes:**
- Must be called before `parse_dag()` or `get_summary()`
- Tries JSON first, falls back to YAML
- Validates that root is a non-empty dictionary

---

### `parse_dag() -> Dict[str, List[str]]`

Parse task dependencies into an adjacency list (DAG).

**Returns:**
- Adjacency dict: `{task_label: [dependency_labels]}`

**Raises:**
- `WorkflowError`: If workflow not loaded (call `load()` first)
- `CyclicDependencyError`: If circular dependencies detected

**Example:**
```python
dag = inspector.parse_dag()
# {
#   "validate": [],
#   "test": ["validate"],
#   "deploy": ["test"],
#   "notify": ["deploy", "test"]
# }
```

**Algorithm:**
- Builds adjacency list from `depends_on` fields
- Normalizes string dependencies to lists
- Detects cycles using DFS with recursion stack
- Time complexity: O(V + E) where V=tasks, E=dependencies

---

### `get_task_status(label: str) -> str`

Get the status of a specific task.

**Parameters:**
- `label` (str): Task label/name from workflow

**Returns:**
- Status string: "pending", "running", "succeeded", "failed", etc.
- Returns "unknown" if task not found
- Returns "pending" if status field missing

**Example:**
```python
status = inspector.get_task_status("deploy")
if status == "failed":
    # Handle failure
```

---

### `get_summary() -> Dict[str, Any]`

Get comprehensive workflow summary with structure and metadata.

**Returns:**
- Dictionary with keys:
  - `tasks` (list): Task objects with {label, type, status}
  - `dag` (dict): Adjacency list of dependencies
  - `task_count` (int): Number of tasks
  - `output_refs` (list): Discovered output references

**Raises:**
- `WorkflowError`: If workflow not loaded
- `CyclicDependencyError`: If circular dependencies detected

**Example:**
```python
summary = inspector.get_summary()

# Structure:
# {
#   "tasks": [
#     {"label": "build", "type": "build", "status": "pending"},
#     {"label": "test", "type": "test", "status": "pending"}
#   ],
#   "dag": {
#     "build": [],
#     "test": ["build"]
#   },
#   "task_count": 2,
#   "output_refs": ["$task.build.outputs.artifacts"]
# }

print(f"Found {summary['task_count']} tasks")
for task in summary['tasks']:
    print(f"  - {task['label']}: {task['status']}")
```

**Notes:**
- Calls `parse_dag()` automatically if not called yet
- All outputs are JSON-serializable
- Task metadata depends on workflow format

---

## Exception Classes

### `WorkflowInspector.WorkflowError`

Base exception for workflow loading and parsing errors.

**Common scenarios:**
- File not found
- Parse error (invalid YAML/JSON)
- Empty file
- Root is not a dictionary
- YAML support not available

**Example:**
```python
try:
    inspector.load()
except WorkflowInspector.WorkflowError as e:
    print(f"Failed to load workflow: {e}")
```

---

### `WorkflowInspector.CyclicDependencyError`

Raised when circular dependencies are detected.

**Inherits from:** `WorkflowError`

**Example:**
```python
try:
    inspector.parse_dag()
except WorkflowInspector.CyclicDependencyError as e:
    print(f"Circular dependency: {e}")
    # Fix workflow, remove cycle
```

---

## Workflow Format Support

### YAML Format

```yaml
tasks:
  - label: validate
    type: script
    command: ./validate.sh
    depends_on: []
    status: pending

  - label: test
    type: test
    command: pytest
    depends_on: validate
    status: pending
```

**Field mapping:**
- Task identifier: `label`, `name`, or `id`
- Dependencies: `depends_on` (string or list of strings)
- Status: `status` field (defaults to "pending")
- Other fields: Preserved and included in summary

### JSON Format

```json
{
  "tasks": [
    {
      "label": "build",
      "type": "build",
      "depends_on": []
    },
    {
      "label": "test",
      "type": "test",
      "depends_on": "build"
    }
  ]
}
```

---

## Output Reference Discovery

Output references are automatically discovered in workflow fields:

**Pattern matching:**
- Contains `$` and `output`
- Examples: `$task.X.outputs.Y`, `${task.outputs}`, etc.
- Found recursively in all task fields

**Example workflow:**
```yaml
tasks:
  - label: fetch
    outputs:
      data: ./result.json

  - label: process
    depends_on: fetch
    input: $task.fetch.outputs.data
    output: $task.process.outputs.transformed

  - label: publish
    depends_on: process
    source: $task.process.outputs.transformed
```

**Summary output:**
```python
summary = inspector.get_summary()
# output_refs: [
#   "$task.fetch.outputs.data",
#   "$task.process.outputs.transformed"
# ]
```

---

## Complete Example

```python
from mr1.developer_tools import WorkflowInspector
import json

# Create inspector
inspector = WorkflowInspector("complex_workflow.yaml")

try:
    # Load workflow
    workflow = inspector.load()
    print(f"Loaded workflow with {len(workflow.get('tasks', []))} tasks")

    # Get summary
    summary = inspector.get_summary()
    
    # Display structure
    print(f"\nWorkflow Summary:")
    print(f"  Tasks: {summary['task_count']}")
    print(f"  Output refs: {len(summary['output_refs'])}")
    
    # Analyze dependencies
    print(f"\nDependency Graph:")
    for task, deps in sorted(summary['dag'].items()):
        if deps:
            print(f"  {task} → {', '.join(deps)}")
        else:
            print(f"  {task} (no dependencies)")
    
    # Query specific task
    deploy_status = inspector.get_task_status("deploy")
    print(f"\nDeploy status: {deploy_status}")
    
    # Export as JSON
    json_output = json.dumps(summary, indent=2)
    with open("workflow_analysis.json", "w") as f:
        f.write(json_output)
        
except WorkflowInspector.CyclicDependencyError:
    print("ERROR: Workflow contains circular dependencies")
except WorkflowInspector.WorkflowError as e:
    print(f"ERROR: {e}")
```

---

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| `load()` | O(n) | O(n) |
| `parse_dag()` | O(V + E) | O(V) |
| `get_task_status()` | O(1) | O(1) |
| `get_summary()` | O(V + E + n) | O(V + n) |

Where:
- n = file size
- V = number of tasks
- E = number of dependencies
- Typical workflow (50 tasks): < 1ms

---

## Design Notes

- **Deterministic**: Same input produces identical output
- **JSON-serializable**: All outputs can be serialized with `json.dumps()`
- **No side effects**: Read-only, no file modifications
- **Explicit errors**: Clear exception types for different failures
- **Minimal dependencies**: json (built-in) + yaml (optional)
- **Format agnostic**: Works with any task structure that has labels/names
