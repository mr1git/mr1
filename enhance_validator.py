import sys

# Read the file
with open('mr1/new_tools.py', 'r') as f:
    content = f.read()

# Find and replace the _validate_schema function
old_code = '''    for field, value in data.items():
        if field not in field_schemas:
            warnings.append(f"unexpected field: {field}")
            continue

        field_schema = field_schemas[field]
        expected_type = field_schema.get("type")

        if expected_type:
            actual_type = type(value).__name__
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"field '{field}': expected string, got {actual_type}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"field '{field}': expected integer, got {actual_type}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"field '{field}': expected number, got {actual_type}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"field '{field}': expected boolean, got {actual_type}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"field '{field}': expected array, got {actual_type}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"field '{field}': expected object, got {actual_type}")

        if "minimum" in field_schema and isinstance(value, (int, float)):
            if value < field_schema["minimum"]:
                errors.append(f"field '{field}': value {value} less than minimum {field_schema['minimum']}")

        if "maximum" in field_schema and isinstance(value, (int, float)):
            if value > field_schema["maximum"]:
                errors.append(f"field '{field}': value {value} greater than maximum {field_schema['maximum']}")'''

new_code = '''    for field, value in data.items():
        if field not in field_schemas:
            warnings.append(f"unexpected field: {field}")
            continue

        field_schema = field_schemas[field]
        expected_type = field_schema.get("type")

        if expected_type:
            actual_type = type(value).__name__
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"field '{field}': expected string, got {actual_type}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"field '{field}': expected integer, got {actual_type}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"field '{field}': expected number, got {actual_type}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"field '{field}': expected boolean, got {actual_type}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"field '{field}': expected array, got {actual_type}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"field '{field}': expected object, got {actual_type}")

        if expected_type == "object" and isinstance(value, dict):
            nested_required = field_schema.get("required", [])
            nested_properties = field_schema.get("properties", {})
            for req_field in nested_required:
                if req_field not in value:
                    errors.append(f"field '{field}.{req_field}': missing required field")
            for nested_field, nested_value in value.items():
                if nested_field in nested_properties:
                    nested_schema = nested_properties[nested_field]
                    nested_type = nested_schema.get("type")
                    if nested_type == "string" and not isinstance(nested_value, str):
                        errors.append(f"field '{field}.{nested_field}': expected string, got {type(nested_value).__name__}")
                    elif nested_type == "integer" and not isinstance(nested_value, int):
                        errors.append(f"field '{field}.{nested_field}': expected integer, got {type(nested_value).__name__}")

        if expected_type == "array" and isinstance(value, list):
            if "minItems" in field_schema and len(value) < field_schema["minItems"]:
                errors.append(f"field '{field}': array length {len(value)} less than minimum {field_schema['minItems']}")
            if "maxItems" in field_schema and len(value) > field_schema["maxItems"]:
                errors.append(f"field '{field}': array length {len(value)} greater than maximum {field_schema['maxItems']}")
            items_schema = field_schema.get("items")
            if items_schema and isinstance(items_schema, dict):
                item_type = items_schema.get("type")
                item_properties = items_schema.get("properties", {})
                item_required = items_schema.get("required", [])
                for i, item in enumerate(value):
                    if item_type:
                        if item_type == "string" and not isinstance(item, str):
                            errors.append(f"field '{field}[{i}]': expected string, got {type(item).__name__}")
                        elif item_type == "integer" and not isinstance(item, int):
                            errors.append(f"field '{field}[{i}]': expected integer, got {type(item).__name__}")
                        elif item_type == "object" and isinstance(item, dict):
                            for req_field in item_required:
                                if req_field not in item:
                                    errors.append(f"field '{field}[{i}].{req_field}': missing required field")
                            for key, val in item.items():
                                if key in item_properties:
                                    item_field_schema = item_properties[key]
                                    item_field_type = item_field_schema.get("type")
                                    if item_field_type == "string" and not isinstance(val, str):
                                        errors.append(f"field '{field}[{i}].{key}': expected string, got {type(val).__name__}")
                                    elif item_field_type == "integer" and not isinstance(val, int):
                                        errors.append(f"field '{field}[{i}].{key}': expected integer, got {type(val).__name__}")

        if "minimum" in field_schema and isinstance(value, (int, float)):
            if value < field_schema["minimum"]:
                errors.append(f"field '{field}': value {value} less than minimum {field_schema['minimum']}")

        if "maximum" in field_schema and isinstance(value, (int, float)):
            if value > field_schema["maximum"]:
                errors.append(f"field '{field}': value {value} greater than maximum {field_schema['maximum']}")

        if "minLength" in field_schema and isinstance(value, str):
            if len(value) < field_schema["minLength"]:
                errors.append(f"field '{field}': string length {len(value)} less than minimum {field_schema['minLength']}")

        if "maxLength" in field_schema and isinstance(value, str):
            if len(value) > field_schema["maxLength"]:
                errors.append(f"field '{field}': string length {len(value)} greater than maximum {field_schema['maxLength']}")

        if "pattern" in field_schema and isinstance(value, str):
            if not re.match(field_schema["pattern"], value):
                errors.append(f"field '{field}': does not match pattern {field_schema['pattern']}")

        if "enum" in field_schema:
            if value not in field_schema["enum"]:
                errors.append(f"field '{field}': value '{value}' not in enum {field_schema['enum']}")'''

content = content.replace(old_code, new_code)

# Write the file
with open('mr1/new_tools.py', 'w') as f:
    f.write(content)

print("File updated successfully")
