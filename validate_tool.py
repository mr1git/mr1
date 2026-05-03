#!/usr/bin/env python3
"""Validation script for data_query_transform tool integration."""

from mr1.tools import default_tool_registry

registry = default_tool_registry()
tools = registry.list_tools()
data_query_tool = next((t for t in tools if t.tool_type == "data_query_transform"), None)

if data_query_tool:
    print("✓ data_query_transform tool found in registry")
    print(f"  Description: {data_query_tool.description}")
    print(f"  Config schema keys: {list(data_query_tool.config_schema.keys())}")
    print(f"  Outputs: {list(data_query_tool.outputs.keys())}")
    print(f"  Examples: {len(data_query_tool.examples)}")

    # Validate a config
    try:
        registry.validate_spec("data_query_transform", {
            "mode": "sqlite",
            "db_path": "/tmp/test.db",
            "query": "SELECT * FROM users"
        })
        print("✓ SQLite config validation passed")
    except Exception as e:
        print(f"✗ SQLite config validation failed: {e}")

    # Validate data mode config
    try:
        registry.validate_spec("data_query_transform", {
            "mode": "data",
            "operation": "filter",
            "pattern": "test",
            "input_data": ["test", "data"]
        })
        print("✓ Data mode config validation passed")
    except Exception as e:
        print(f"✗ Data mode config validation failed: {e}")
else:
    print("✗ data_query_transform tool NOT found")
    print(f"Available tools: {[t.tool_type for t in tools]}")
