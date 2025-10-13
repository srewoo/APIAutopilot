# -*- coding: utf-8 -*-
"""
Enhanced Test Generator V2 - Template-based with Multi-stage Pipeline
Implements all recommended improvements for production-ready test generation
"""

import json
import re
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class TestProfile(Enum):
    """Test generation profiles for different use cases"""
    QUICK_SMOKE = "quick_smoke"
    SECURITY_AUDIT = "security_audit"
    FULL_REGRESSION = "full_regression"
    CI_CD_PIPELINE = "ci_cd"
    CUSTOM = "custom"


class TestFramework(Enum):
    """Supported test frameworks"""
    JEST = "jest"
    MOCHA = "mocha"
    CYPRESS = "cypress"
    PYTEST = "pytest"
    REQUESTS = "requests"
    TESTNG = "testng"
    JUNIT = "junit"
    RESTASSURED = "restassured"  # RestAssured (Java)
    BEHAVE = "behave"  # Behave BDD (Python)


@dataclass
class APIAnalysis:
    """Analysis results for an API"""
    api_type: str  # rest, graphql, soap
    auth_type: Optional[str]  # bearer, basic, api_key, oauth2
    endpoints: List[Dict[str, Any]]
    base_url: str
    headers: Dict[str, str]
    request_body: Optional[Dict[str, Any]]
    response_example: Optional[Dict[str, Any]]
    response_schema: Optional[Dict[str, Any]]
    method: str = "GET"
    content_type: str = "application/json"


@dataclass
class TestCase:
    """Individual test case representation"""
    name: str
    category: str  # positive, negative, security, edge, performance
    method: str
    endpoint: str
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]]
    expected_status: int
    assertions: List[str]
    description: str = ""
    priority: int = 1  # 1-5, 1 being highest
    tags: List[str] = field(default_factory=list)


@dataclass
class GeneratedTestSuite:
    """Complete test suite with metadata"""
    framework: TestFramework
    test_cases: List[TestCase]
    setup_code: str
    teardown_code: str
    helper_functions: str
    imports: str
    coverage_report: Dict[str, Any]
    quality_score: float
    generation_time: float
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# TEST PROFILE CONFIGURATIONS
# ============================================================================

TEST_PROFILES = {
    TestProfile.QUICK_SMOKE: {
        "categories": ["positive", "auth"],
        "test_count": {"min": 5, "max": 10},
        "depth": "shallow",
        "priority": [1, 2]
    },
    TestProfile.SECURITY_AUDIT: {
        "categories": ["security", "injection", "auth", "encryption"],
        "test_count": {"min": 30, "max": 50},
        "depth": "deep",
        "priority": [1, 2, 3]
    },
    TestProfile.FULL_REGRESSION: {
        "categories": ["positive", "negative", "security", "edge", "performance"],
        "test_count": {"min": 50, "max": 100},
        "depth": "comprehensive",
        "priority": [1, 2, 3, 4, 5]
    },
    TestProfile.CI_CD_PIPELINE: {
        "categories": ["positive", "critical_negative", "auth"],
        "test_count": {"min": 15, "max": 20},
        "depth": "balanced",
        "priority": [1, 2]
    }
}


# ============================================================================
# TEMPLATE ENGINE
# ============================================================================

class TemplateEngine:
    """Template-based code generation for different frameworks"""

    def __init__(self, framework: TestFramework):
        self.framework = framework
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load templates for the selected framework"""
        if self.framework == TestFramework.JEST:
            return {
                "file": """const axios = require('axios');
const Ajv = require('ajv');
const ajv = new Ajv();

// Test Configuration
const BASE_URL = '{base_url}';
const DEFAULT_TIMEOUT = 30000;

// Helper Functions
{helper_functions}

// Test Suite
describe('{suite_name}', () => {{
  // Setup
  beforeAll(async () => {{
    {setup_code}
  }});

  // Teardown
  afterAll(async () => {{
    {teardown_code}
  }});

  // Test Cases
  {test_cases}
}});
""",
                "test_case": """
  it('{test_name}', async () => {{
    {test_body}
  }}, DEFAULT_TIMEOUT);""",

                "positive_test": """
    const requestBody = {request_body};
    const headers = {headers};

    const response = await axios.{method}(
      BASE_URL + '{endpoint}',
      {axios_body}
      {{ headers, timeout: DEFAULT_TIMEOUT }}
    );

    expect(response.status).toBe({expected_status});
    {assertions}""",

                "negative_test": """
    const requestBody = {request_body};
    const headers = {headers};

    try {{
      await axios.{method}(
        BASE_URL + '{endpoint}',
        {axios_body}
        {{ headers, timeout: DEFAULT_TIMEOUT }}
      );
      fail('Expected request to fail');
    }} catch (error) {{
      expect(error.response.status).toBe({expected_status});
      {assertions}
    }}""",

                "assertion": "expect({target}).{matcher}({expected});",

                "schema_validation": """
    const schema = {schema};
    const valid = ajv.validate(schema, response.data);
    expect(valid).toBe(true);
    if (!valid) console.log(ajv.errors);"""
            }

        elif self.framework == TestFramework.PYTEST:
            return {
                "file": """import pytest
import requests
import json
from jsonschema import validate, ValidationError
from typing import Dict, Any
import time

# Test Configuration
BASE_URL = '{base_url}'
DEFAULT_TIMEOUT = 30

# Helper Functions
{helper_functions}

class Test{suite_name}:
    \"\"\"Generated test suite for {suite_name}\"\"\"

    @classmethod
    def setup_class(cls):
        \"\"\"Setup for test suite\"\"\"
        {setup_code}

    @classmethod
    def teardown_class(cls):
        \"\"\"Teardown for test suite\"\"\"
        {teardown_code}

    {test_cases}
""",
                "test_case": """
    def test_{test_name}(self):
        \"\"\"
        {test_description}
        \"\"\"
        {test_body}""",

                "positive_test": """
        request_body = {request_body}
        headers = {headers}

        response = requests.{method}(
            f"{{BASE_URL}}{endpoint}",
            json=request_body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT
        )

        assert response.status_code == {expected_status}
        {assertions}""",

                "negative_test": """
        request_body = {request_body}
        headers = {headers}

        response = requests.{method}(
            f"{{BASE_URL}}{endpoint}",
            json=request_body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT
        )

        assert response.status_code == {expected_status}
        {assertions}""",

                "assertion": "assert {target} {operator} {expected}",

                "schema_validation": """
        schema = {schema}
        validate(instance=response.json(), schema=schema)"""
            }

        # Add more framework templates as needed
        return {}

    def generate_test_file(self, test_suite: GeneratedTestSuite) -> str:
        """Generate complete test file from test suite"""
        test_cases_code = []

        for test_case in test_suite.test_cases:
            test_code = self._generate_test_case(test_case)
            test_cases_code.append(test_code)

        return self.templates["file"].format(
            base_url=test_suite.test_cases[0].endpoint.split('/')[0] if test_suite.test_cases else "",
            suite_name="API Tests",
            helper_functions=test_suite.helper_functions,
            setup_code=test_suite.setup_code,
            teardown_code=test_suite.teardown_code,
            test_cases="\n".join(test_cases_code)
        )

    def _generate_test_case(self, test_case: TestCase) -> str:
        """Generate code for a single test case"""
        template_type = "positive_test" if test_case.category == "positive" else "negative_test"

        # Prepare axios body based on method
        if self.framework == TestFramework.JEST:
            if test_case.method.lower() in ['post', 'put', 'patch']:
                axios_body = f"requestBody,"
            else:
                axios_body = ""
        else:
            axios_body = ""

        test_body = self.templates[template_type].format(
            method=test_case.method.lower(),
            endpoint=test_case.endpoint,
            request_body=json.dumps(test_case.body) if test_case.body else "null",
            headers=json.dumps(test_case.headers),
            expected_status=test_case.expected_status,
            assertions="\n    ".join(test_case.assertions),
            axios_body=axios_body
        )

        return self.templates["test_case"].format(
            test_name=test_case.name.replace(" ", "_").lower(),
            test_description=test_case.description,
            test_body=test_body
        )


# ============================================================================
# ASSERTION RULE ENGINE
# ============================================================================

class AssertionRuleEngine:
    """Generates precise assertions based on data types and patterns"""

    def __init__(self, framework: TestFramework):
        self.framework = framework
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, callable]:
        """Load assertion rules for different patterns"""
        if self.framework == TestFramework.JEST:
            return {
                "email": lambda field: f"expect({field}).toMatch(/^[\\w.-]+@[\\w.-]+\\.\\w+$/)",
                "uuid": lambda field: f"expect({field}).toMatch(/^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$/i)",
                "url": lambda field: f"expect({field}).toMatch(/^https?:\\/\\/.+/)",
                "timestamp": lambda field: f"expect(new Date({field}).getTime()).toBeGreaterThan(0)",
                "iso_date": lambda field: f"expect({field}).toMatch(/^\\d{{4}}-\\d{{2}}-\\d{{2}}T\\d{{2}}:\\d{{2}}:\\d{{2}}/)",
                "phone": lambda field: f"expect({field}).toMatch(/^\\+?[1-9]\\d{{1,14}}$/)",
                "jwt": lambda field: f"expect({field}).toMatch(/^[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+$/)",
                "base64": lambda field: f"expect({field}).toMatch(/^[A-Za-z0-9+/]+=*$/)",
                "ipv4": lambda field: f"expect({field}).toMatch(/^(?:[0-9]{{1,3}}\\.)){{3}}[0-9]{{1,3}}$/)",
                "ipv6": lambda field: f"expect({field}).toMatch(/^(?:[0-9a-fA-F]{{1,4}}:)){{7}}[0-9a-fA-F]{{1,4}}$/)",
                "array_length": lambda field, length: f"expect({field}.length).toBe({length})",
                "array_min_length": lambda field, min_len: f"expect({field}.length).toBeGreaterThanOrEqual({min_len})",
                "number_range": lambda field, min_val, max_val: f"expect({field}).toBeGreaterThanOrEqual({min_val}) && expect({field}).toBeLessThanOrEqual({max_val})",
                "string_length": lambda field, length: f"expect({field}.length).toBe({length})",
                "not_empty": lambda field: f"expect({field}).toBeTruthy() && expect({field}.length).toBeGreaterThan(0)",
                "enum": lambda field, values: f"expect({json.dumps(values)}).toContain({field})"
            }
        elif self.framework == TestFramework.PYTEST:
            return {
                "email": lambda field: f"assert re.match(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', {field})",
                "uuid": lambda field: f"assert re.match(r'^[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}}$', {field}, re.I)",
                "url": lambda field: f"assert {field}.startswith(('http://', 'https://'))",
                "timestamp": lambda field: f"assert datetime.fromtimestamp({field}) > datetime(1970, 1, 1)",
                "array_length": lambda field, length: f"assert len({field}) == {length}",
                "not_empty": lambda field: f"assert {field} and len({field}) > 0"
            }
        return {}

    def generate_assertions(self, response_data: Dict[str, Any], path: str = "response.data") -> List[str]:
        """Generate assertions for response data"""
        assertions = []

        for field_name, field_value in response_data.items():
            field_path = f"{path}.{field_name}"

            # Type assertions
            if isinstance(field_value, str):
                assertions.append(self._generate_string_assertion(field_path, field_value))
            elif isinstance(field_value, (int, float)):
                assertions.append(self._generate_number_assertion(field_path, field_value))
            elif isinstance(field_value, bool):
                assertions.append(self._generate_boolean_assertion(field_path, field_value))
            elif isinstance(field_value, list):
                assertions.extend(self._generate_array_assertions(field_path, field_value))
            elif isinstance(field_value, dict):
                # Recursive for nested objects
                assertions.extend(self.generate_assertions(field_value, field_path))
            elif field_value is None:
                assertions.append(self._generate_null_assertion(field_path))

        return assertions

    def _generate_string_assertion(self, field_path: str, value: str) -> str:
        """Generate assertion for string field"""
        if self.framework == TestFramework.JEST:
            # Check for patterns
            if re.match(r'^[\w.-]+@[\w.-]+\.\w+$', value):
                return self.rules["email"](field_path)
            elif re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value, re.I):
                return self.rules["uuid"](field_path)
            elif value.startswith(('http://', 'https://')):
                return self.rules["url"](field_path)
            elif re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                return self.rules["iso_date"](field_path)
            elif re.match(r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$', value):
                return self.rules["jwt"](field_path)
            else:
                return f"expect(typeof {field_path}).toBe('string')"
        elif self.framework == TestFramework.PYTEST:
            return f"assert isinstance({field_path}, str)"
        return ""

    def _generate_number_assertion(self, field_path: str, value: float) -> str:
        """Generate assertion for number field"""
        if self.framework == TestFramework.JEST:
            return f"expect(typeof {field_path}).toBe('number')"
        elif self.framework == TestFramework.PYTEST:
            return f"assert isinstance({field_path}, (int, float))"
        return ""

    def _generate_boolean_assertion(self, field_path: str, value: bool) -> str:
        """Generate assertion for boolean field"""
        if self.framework == TestFramework.JEST:
            return f"expect({field_path}).toBe({str(value).lower()})"
        elif self.framework == TestFramework.PYTEST:
            return f"assert {field_path} is {value}"
        return ""

    def _generate_array_assertions(self, field_path: str, value: list) -> List[str]:
        """Generate assertions for array field"""
        assertions = []
        if self.framework == TestFramework.JEST:
            assertions.append(f"expect(Array.isArray({field_path})).toBe(true)")
            if value:
                assertions.append(f"expect({field_path}.length).toBeGreaterThan(0)")
        elif self.framework == TestFramework.PYTEST:
            assertions.append(f"assert isinstance({field_path}, list)")
            if value:
                assertions.append(f"assert len({field_path}) > 0")
        return assertions

    def _generate_null_assertion(self, field_path: str) -> str:
        """Generate assertion for null field"""
        if self.framework == TestFramework.JEST:
            return f"expect({field_path}).toBeNull()"
        elif self.framework == TestFramework.PYTEST:
            return f"assert {field_path} is None"
        return ""


# ============================================================================
# API ANALYZER
# ============================================================================

class APIAnalyzer:
    """Analyzes API specifications and requests"""

    def analyze(self, input_data: Dict[str, Any], input_type: str) -> APIAnalysis:
        """Analyze API input and extract key information"""
        if input_type == "curl":
            return self._analyze_curl(input_data)
        elif input_type == "har":
            return self._analyze_har(input_data)
        elif input_type == "openapi":
            return self._analyze_openapi(input_data)
        elif input_type == "graphql":
            return self._analyze_graphql(input_data)
        else:
            return self._analyze_generic(input_data)

    def _analyze_curl(self, curl_data: Dict[str, Any]) -> APIAnalysis:
        """Analyze parsed cURL command"""
        return APIAnalysis(
            api_type="rest",
            auth_type=self._detect_auth_type(curl_data.get("headers", {})),
            endpoints=[curl_data.get("url", "")],
            base_url=self._extract_base_url(curl_data.get("url", "")),
            headers=curl_data.get("headers", {}),
            request_body=self._parse_body(curl_data.get("body", "")),
            response_example=None,
            response_schema=None,
            method=curl_data.get("method", "GET")
        )

    def _analyze_har(self, har_data: Dict[str, Any]) -> APIAnalysis:
        """Analyze HAR file data"""
        # Implementation for HAR analysis
        pass

    def _analyze_openapi(self, spec: Dict[str, Any]) -> APIAnalysis:
        """Analyze OpenAPI/Swagger specification"""
        # Implementation for OpenAPI analysis
        pass

    def _analyze_graphql(self, graphql_data: Dict[str, Any]) -> APIAnalysis:
        """Analyze GraphQL query/mutation"""
        return APIAnalysis(
            api_type="graphql",
            auth_type=None,
            endpoints=["/graphql"],
            base_url="",
            headers={"Content-Type": "application/json"},
            request_body=graphql_data,
            response_example=None,
            response_schema=None,
            method="POST"
        )

    def _analyze_generic(self, data: Dict[str, Any]) -> APIAnalysis:
        """Generic analysis for unstructured input"""
        return APIAnalysis(
            api_type="rest",
            auth_type=None,
            endpoints=[],
            base_url="",
            headers={},
            request_body=None,
            response_example=None,
            response_schema=None
        )

    def _detect_auth_type(self, headers: Dict[str, str]) -> Optional[str]:
        """Detect authentication type from headers"""
        auth_header = headers.get("Authorization", "").lower()
        if "bearer" in auth_header:
            return "bearer"
        elif "basic" in auth_header:
            return "basic"
        elif "api-key" in headers or "x-api-key" in headers:
            return "api_key"
        return None

    def _extract_base_url(self, url: str) -> str:
        """Extract base URL from full URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _parse_body(self, body_str: str) -> Optional[Dict[str, Any]]:
        """Parse request body string to dict"""
        if not body_str:
            return None
        try:
            return json.loads(body_str)
        except json.JSONDecodeError:
            # Try fixing common issues
            try:
                fixed_body = body_str.replace("'", '"')
                return json.loads(fixed_body)
            except:
                return None


# ============================================================================
# TEST CASE GENERATOR
# ============================================================================

class TestCaseGenerator:
    """Generates test cases based on API analysis"""

    def __init__(self, assertion_engine: AssertionRuleEngine):
        self.assertion_engine = assertion_engine

    def generate_test_cases(
        self,
        api_analysis: APIAnalysis,
        profile: TestProfile,
        captured_response: Optional[Dict[str, Any]] = None
    ) -> List[TestCase]:
        """Generate test cases based on profile and API analysis"""
        test_cases = []
        profile_config = TEST_PROFILES[profile]

        # Generate tests by category
        if "positive" in profile_config["categories"]:
            test_cases.extend(self._generate_positive_tests(api_analysis, captured_response))

        if "negative" in profile_config["categories"]:
            test_cases.extend(self._generate_negative_tests(api_analysis))

        if "security" in profile_config["categories"]:
            test_cases.extend(self._generate_security_tests(api_analysis))

        if "edge" in profile_config["categories"]:
            test_cases.extend(self._generate_edge_tests(api_analysis))

        if "performance" in profile_config["categories"]:
            test_cases.extend(self._generate_performance_tests(api_analysis))

        # Filter by priority
        test_cases = [tc for tc in test_cases if tc.priority in profile_config["priority"]]

        # Limit to profile test count
        max_tests = profile_config["test_count"]["max"]
        return test_cases[:max_tests]

    def _generate_positive_tests(
        self,
        api_analysis: APIAnalysis,
        captured_response: Optional[Dict[str, Any]]
    ) -> List[TestCase]:
        """Generate positive test cases"""
        test_cases = []

        # Basic success test
        assertions = []
        if captured_response:
            assertions = self.assertion_engine.generate_assertions(captured_response)
        else:
            assertions = [
                "expect(response.status).toBe(200)",
                "expect(response.data).toBeDefined()"
            ]

        test_cases.append(TestCase(
            name="should successfully call API with valid data",
            category="positive",
            method=api_analysis.method,
            endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
            headers=api_analysis.headers,
            body=api_analysis.request_body,
            expected_status=200,
            assertions=assertions,
            description="Verify API returns success with valid request",
            priority=1,
            tags=["smoke", "positive"]
        ))

        # Add more positive test variations
        if api_analysis.response_schema:
            test_cases.append(TestCase(
                name="should return response matching schema",
                category="positive",
                method=api_analysis.method,
                endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
                headers=api_analysis.headers,
                body=api_analysis.request_body,
                expected_status=200,
                assertions=[
                    "expect(response.status).toBe(200)",
                    f"const schema = {json.dumps(api_analysis.response_schema)}",
                    "expect(ajv.validate(schema, response.data)).toBe(true)"
                ],
                description="Verify response matches expected schema",
                priority=1,
                tags=["schema", "positive"]
            ))

        return test_cases

    def _generate_negative_tests(self, api_analysis: APIAnalysis) -> List[TestCase]:
        """Generate negative test cases"""
        test_cases = []

        # Missing required fields
        if api_analysis.request_body:
            for field in api_analysis.request_body.keys():
                invalid_body = api_analysis.request_body.copy()
                del invalid_body[field]

                test_cases.append(TestCase(
                    name=f"should fail when {field} is missing",
                    category="negative",
                    method=api_analysis.method,
                    endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
                    headers=api_analysis.headers,
                    body=invalid_body,
                    expected_status=400,
                    assertions=[
                        "expect(error.response.status).toBe(400)",
                        "expect(error.response.data).toHaveProperty('error')"
                    ],
                    description=f"Verify API returns 400 when required field {field} is missing",
                    priority=2,
                    tags=["validation", "negative"]
                ))

        # Invalid data types
        if api_analysis.request_body:
            invalid_type_body = api_analysis.request_body.copy()
            for field, value in invalid_type_body.items():
                if isinstance(value, str):
                    invalid_type_body[field] = 123  # String to number
                    break
                elif isinstance(value, (int, float)):
                    invalid_type_body[field] = "invalid"  # Number to string
                    break

            test_cases.append(TestCase(
                name="should fail with invalid data types",
                category="negative",
                method=api_analysis.method,
                endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
                headers=api_analysis.headers,
                body=invalid_type_body,
                expected_status=400,
                assertions=[
                    "expect(error.response.status).toBe(400)"
                ],
                description="Verify API validates data types",
                priority=2,
                tags=["validation", "negative"]
            ))

        return test_cases

    def _generate_security_tests(self, api_analysis: APIAnalysis) -> List[TestCase]:
        """Generate security test cases"""
        test_cases = []

        # SQL Injection tests
        sql_injection_payloads = [
            "' OR '1'='1",
            "admin'--",
            "1; DROP TABLE users",
            "' UNION SELECT * FROM users--"
        ]

        for payload in sql_injection_payloads:
            if api_analysis.request_body:
                injection_body = api_analysis.request_body.copy()
                # Inject into first string field
                for field, value in injection_body.items():
                    if isinstance(value, str):
                        injection_body[field] = payload
                        break

                test_cases.append(TestCase(
                    name=f"should prevent SQL injection with payload: {payload[:20]}",
                    category="security",
                    method=api_analysis.method,
                    endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
                    headers=api_analysis.headers,
                    body=injection_body,
                    expected_status=400,
                    assertions=[
                        "expect(error.response.status).toBe(400)",
                        "expect(error.response.data).not.toContain('SQL')",
                        "expect(error.response.data).not.toContain('syntax')"
                    ],
                    description=f"Verify API prevents SQL injection",
                    priority=1,
                    tags=["security", "injection", "sql"]
                ))

        # XSS tests
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert('XSS')",
            "<svg/onload=alert('XSS')>"
        ]

        for payload in xss_payloads[:2]:  # Limit to 2 for brevity
            if api_analysis.request_body:
                xss_body = api_analysis.request_body.copy()
                for field, value in xss_body.items():
                    if isinstance(value, str):
                        xss_body[field] = payload
                        break

                test_cases.append(TestCase(
                    name=f"should prevent XSS with script tags",
                    category="security",
                    method=api_analysis.method,
                    endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
                    headers=api_analysis.headers,
                    body=xss_body,
                    expected_status=400,
                    assertions=[
                        "expect(error.response.status).toBe(400)",
                        "expect(error.response.data).not.toContain('<script>')"
                    ],
                    description="Verify API prevents XSS attacks",
                    priority=1,
                    tags=["security", "xss"]
                ))

        # Authentication tests
        test_cases.append(TestCase(
            name="should fail without authentication",
            category="security",
            method=api_analysis.method,
            endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
            headers={},  # No auth headers
            body=api_analysis.request_body,
            expected_status=401,
            assertions=[
                "expect(error.response.status).toBe(401)",
                "expect(error.response.data).toHaveProperty('error')"
            ],
            description="Verify API requires authentication",
            priority=1,
            tags=["security", "auth"]
        ))

        return test_cases

    def _generate_edge_tests(self, api_analysis: APIAnalysis) -> List[TestCase]:
        """Generate edge case tests"""
        test_cases = []

        # Very large payload
        if api_analysis.request_body:
            large_body = api_analysis.request_body.copy()
            for field, value in large_body.items():
                if isinstance(value, str):
                    large_body[field] = "x" * 10000  # 10KB string
                    break

            test_cases.append(TestCase(
                name="should handle very large payloads",
                category="edge",
                method=api_analysis.method,
                endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
                headers=api_analysis.headers,
                body=large_body,
                expected_status=413,  # Payload too large
                assertions=[
                    "expect(error.response.status).toBe(413)"
                ],
                description="Verify API handles large payloads appropriately",
                priority=3,
                tags=["edge", "boundary"]
            ))

        # Special characters
        special_chars_body = api_analysis.request_body.copy() if api_analysis.request_body else {}
        for field in special_chars_body:
            if isinstance(special_chars_body[field], str):
                special_chars_body[field] = "Test™ © ® 🚀 测试"
                break

        test_cases.append(TestCase(
            name="should handle special characters and unicode",
            category="edge",
            method=api_analysis.method,
            endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
            headers=api_analysis.headers,
            body=special_chars_body,
            expected_status=200,
            assertions=[
                "expect(response.status).toBe(200)"
            ],
            description="Verify API handles unicode and special characters",
            priority=3,
            tags=["edge", "unicode"]
        ))

        return test_cases

    def _generate_performance_tests(self, api_analysis: APIAnalysis) -> List[TestCase]:
        """Generate performance test cases"""
        test_cases = []

        # Response time test
        test_cases.append(TestCase(
            name="should respond within acceptable time",
            category="performance",
            method=api_analysis.method,
            endpoint=api_analysis.endpoints[0] if api_analysis.endpoints else "/",
            headers=api_analysis.headers,
            body=api_analysis.request_body,
            expected_status=200,
            assertions=[
                "const startTime = Date.now()",
                "expect(response.status).toBe(200)",
                "const responseTime = Date.now() - startTime",
                "expect(responseTime).toBeLessThan(2000)  // 2 seconds"
            ],
            description="Verify API responds within 2 seconds",
            priority=4,
            tags=["performance", "sla"]
        ))

        return test_cases


# ============================================================================
# INTELLIGENT TEST GENERATOR (Main Orchestrator)
# ============================================================================

class IntelligentTestGenerator:
    """
    Main orchestrator for test generation using multi-stage pipeline
    """

    def __init__(self, framework: TestFramework = TestFramework.JEST):
        self.framework = framework
        self.template_engine = TemplateEngine(framework)
        self.assertion_engine = AssertionRuleEngine(framework)
        self.api_analyzer = APIAnalyzer()
        self.test_generator = TestCaseGenerator(self.assertion_engine)
        self.cache = {}  # Simple in-memory cache

    async def generate(
        self,
        input_data: Dict[str, Any],
        input_type: str,
        profile: TestProfile = TestProfile.FULL_REGRESSION,
        captured_response: Optional[Dict[str, Any]] = None,
        use_ai_enhancement: bool = True  # Changed to TRUE by default - ALWAYS use LLM
    ) -> GeneratedTestSuite:
        """
        Generate complete test suite using multi-stage pipeline
        """
        import time
        start_time = time.time()

        # Stage 1: Analyze API
        api_analysis = self.api_analyzer.analyze(input_data, input_type)

        # Add captured response to analysis if available
        if captured_response:
            api_analysis.response_example = captured_response
            api_analysis.response_schema = self._extract_schema(captured_response)

        # Stage 2: Check cache
        cache_key = self._generate_cache_key(api_analysis, profile)
        if cache_key in self.cache:
            logger.info("Returning cached test suite")
            return self.cache[cache_key]

        # Stage 3: Generate test cases
        test_cases = self.test_generator.generate_test_cases(
            api_analysis,
            profile,
            captured_response
        )

        # Stage 4: AI Enhancement (optional)
        if use_ai_enhancement:
            test_cases = await self._enhance_with_ai(
                test_cases,
                api_analysis,
                ai_provider=getattr(self, 'ai_provider', 'openai'),
                ai_api_key=getattr(self, 'ai_api_key', None),
                temperature=getattr(self, 'temperature', 0.1)
            )

        # Stage 5: Generate helper functions
        helper_functions = self._generate_helper_functions(api_analysis)

        # Stage 6: Generate setup and teardown
        setup_code, teardown_code = self._generate_setup_teardown(api_analysis)

        # Stage 7: Calculate coverage and quality
        coverage_report = self._calculate_coverage(test_cases, api_analysis)
        quality_score = self._calculate_quality_score(test_cases, coverage_report)

        # Stage 8: Create test suite
        test_suite = GeneratedTestSuite(
            framework=self.framework,
            test_cases=test_cases,
            setup_code=setup_code,
            teardown_code=teardown_code,
            helper_functions=helper_functions,
            imports=self._generate_imports(),
            coverage_report=coverage_report,
            quality_score=quality_score,
            generation_time=time.time() - start_time,
            warnings=self._validate_test_suite(test_cases)
        )

        # Stage 9: Cache the result
        self.cache[cache_key] = test_suite

        return test_suite

    def _extract_schema(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON schema from response data"""
        def get_type(value):
            if isinstance(value, bool):
                return "boolean"
            elif isinstance(value, int):
                return "integer"
            elif isinstance(value, float):
                return "number"
            elif isinstance(value, str):
                return "string"
            elif isinstance(value, list):
                return "array"
            elif isinstance(value, dict):
                return "object"
            elif value is None:
                return "null"
            return "unknown"

        def build_schema(data):
            if isinstance(data, dict):
                properties = {}
                required = []
                for key, value in data.items():
                    properties[key] = build_schema(value)
                    if value is not None:
                        required.append(key)
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            elif isinstance(data, list):
                if data:
                    return {
                        "type": "array",
                        "items": build_schema(data[0])
                    }
                return {"type": "array", "items": {}}
            else:
                return {"type": get_type(data)}

        return build_schema(response_data)

    def _generate_cache_key(self, api_analysis: APIAnalysis, profile: TestProfile) -> str:
        """Generate cache key for test suite"""
        key_data = {
            "url": api_analysis.base_url,
            "method": api_analysis.method,
            "profile": profile.value,
            "framework": self.framework.value
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    async def _generate_with_llm(
        self,
        input_data: Dict[str, Any],
        captured_response: Optional[Dict[str, Any]],
        ai_provider: str = "openai",
        ai_api_key: str = None,
        temperature: float = 0.1
    ) -> str:
        """Generate COMPLETE test script using LLM"""
        if not ai_api_key:
            raise ValueError("AI API key is required for test generation")

        try:
            from ai_providers import AIProviderFactory

            # Initialize AI provider
            ai = AIProviderFactory.create_provider(
                provider_name=ai_provider,
                api_key=ai_api_key,
                temperature=temperature
            )

            # Build comprehensive prompt with ALL user inputs
            prompt = f"""Generate a complete, production-ready {self.framework.value} test script for this API:

INPUT DATA:
Method: {input_data.get('method', 'GET')}
URL: {input_data.get('url', '')}
Headers: {json.dumps(input_data.get('headers', {}), indent=2)}
Body: {input_data.get('body', 'null')}

CAPTURED API RESPONSE:
{json.dumps(captured_response, indent=2) if captured_response else 'No response captured'}

REQUIREMENTS:
1. Generate a COMPLETE, RUNNABLE test script
2. Use {self.framework.value} framework syntax
3. Include all necessary imports
4. Add assertions based on the actual API response
5. Include positive, negative, and edge case tests
6. Add proper error handling
7. Make the script production-ready
8. Include schema validation if response was captured
9. Add performance tests
10. Include security tests

Generate the COMPLETE test script now:"""

            system_message = f"You are an expert test automation engineer. Generate a complete, production-ready {self.framework.value} test script."

            # Get complete test script from LLM
            test_script = await ai.generate(
                prompt=prompt,
                system_message=system_message
            )

            return test_script

        except Exception as e:
            import logging
            logging.error(f"LLM generation failed: {str(e)}")
            raise

    def _generate_helper_functions(self, api_analysis: APIAnalysis) -> str:
        """Generate helper functions for tests"""
        if self.framework == TestFramework.JEST:
            return """
// Helper function to validate response structure
const validateResponseStructure = (response, expectedKeys) => {
  expectedKeys.forEach(key => {
    expect(response.data).toHaveProperty(key);
  });
};

// Helper function to generate random data
const generateRandomString = (length) => {
  return Math.random().toString(36).substring(2, 2 + length);
};

// Helper function for delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
"""
        elif self.framework == TestFramework.PYTEST:
            return """
def validate_response_structure(response, expected_keys):
    \"\"\"Validate response has expected keys\"\"\"
    response_data = response.json()
    for key in expected_keys:
        assert key in response_data

def generate_random_string(length):
    \"\"\"Generate random string\"\"\"
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
"""
        return ""

    def _generate_setup_teardown(self, api_analysis: APIAnalysis) -> Tuple[str, str]:
        """Generate setup and teardown code"""
        if self.framework == TestFramework.JEST:
            setup = "// Setup code\n    console.log('Starting test suite');"
            teardown = "// Teardown code\n    console.log('Test suite completed');"
        elif self.framework == TestFramework.PYTEST:
            setup = "# Setup code\n        print('Starting test suite')"
            teardown = "# Teardown code\n        print('Test suite completed')"
        else:
            setup = ""
            teardown = ""

        return setup, teardown

    def _generate_imports(self) -> str:
        """Generate necessary imports for the test file"""
        if self.framework == TestFramework.JEST:
            return """const axios = require('axios');
const Ajv = require('ajv');
const ajv = new Ajv();"""
        elif self.framework == TestFramework.PYTEST:
            return """import pytest
import requests
import json
import re
from jsonschema import validate, ValidationError
from datetime import datetime
import time"""
        return ""

    def _calculate_coverage(
        self,
        test_cases: List[TestCase],
        api_analysis: APIAnalysis
    ) -> Dict[str, Any]:
        """Calculate test coverage metrics"""
        categories = {}
        for tc in test_cases:
            categories[tc.category] = categories.get(tc.category, 0) + 1

        return {
            "total_tests": len(test_cases),
            "categories": categories,
            "endpoints_covered": len(set(tc.endpoint for tc in test_cases)),
            "methods_covered": len(set(tc.method for tc in test_cases)),
            "status_codes_covered": len(set(tc.expected_status for tc in test_cases)),
            "has_positive_tests": "positive" in categories,
            "has_negative_tests": "negative" in categories,
            "has_security_tests": "security" in categories,
            "has_edge_tests": "edge" in categories,
            "has_performance_tests": "performance" in categories
        }

    def _calculate_quality_score(
        self,
        test_cases: List[TestCase],
        coverage_report: Dict[str, Any]
    ) -> float:
        """Calculate quality score for generated tests"""
        score = 0.0

        # Coverage score (40%)
        coverage_score = 0.0
        if coverage_report["has_positive_tests"]:
            coverage_score += 0.2
        if coverage_report["has_negative_tests"]:
            coverage_score += 0.2
        if coverage_report["has_security_tests"]:
            coverage_score += 0.2
        if coverage_report["has_edge_tests"]:
            coverage_score += 0.2
        if coverage_report["has_performance_tests"]:
            coverage_score += 0.2
        score += coverage_score * 0.4

        # Assertion quality (30%)
        avg_assertions = sum(len(tc.assertions) for tc in test_cases) / len(test_cases) if test_cases else 0
        assertion_score = min(avg_assertions / 5, 1.0)  # Normalize to max of 5 assertions
        score += assertion_score * 0.3

        # Test variety (30%)
        unique_tests = len(set(tc.name for tc in test_cases))
        variety_score = min(unique_tests / 20, 1.0)  # Normalize to max of 20 unique tests
        score += variety_score * 0.3

        return round(score, 2)

    def _validate_test_suite(self, test_cases: List[TestCase]) -> List[str]:
        """Validate test suite and return warnings"""
        warnings = []

        if len(test_cases) < 5:
            warnings.append("Test suite has fewer than 5 test cases")

        if not any(tc.category == "positive" for tc in test_cases):
            warnings.append("No positive test cases found")

        if not any(tc.category == "security" for tc in test_cases):
            warnings.append("No security test cases found")

        # Check for duplicate test names
        test_names = [tc.name for tc in test_cases]
        if len(test_names) != len(set(test_names)):
            warnings.append("Duplicate test names detected")

        return warnings

    def generate_executable_code(self, test_suite: GeneratedTestSuite) -> str:
        """Generate executable test code from test suite"""
        return self.template_engine.generate_test_file(test_suite)


# ============================================================================
# SIMPLIFIED AI PROMPT GENERATOR
# ============================================================================

class SmartPromptGenerator:
    """Generate focused, effective prompts for AI enhancement"""

    @staticmethod
    def generate_assertion_prompt(response_data: Dict[str, Any]) -> str:
        """Generate focused prompt for assertions only"""
        return f"""Generate test assertions for this API response:

Response Data:
{json.dumps(response_data, indent=2)}

Requirements:
1. Generate ONLY assertion statements
2. One assertion per line
3. Check exact values where provided
4. Check data types for all fields
5. Use Jest expect() syntax

Format:
expect(response.data.field).toBe(value);
expect(typeof response.data.id).toBe('string');

Output ONLY assertions, no explanations."""

    @staticmethod
    def generate_business_logic_prompt(api_analysis: APIAnalysis) -> str:
        """Generate prompt for business logic tests"""
        return f"""Generate 3 business logic test scenarios for this API:

Endpoint: {api_analysis.endpoints[0] if api_analysis.endpoints else 'unknown'}
Method: {api_analysis.method}

Output format (JSON):
[
  {{
    "scenario": "description",
    "input": {{}},
    "expectedBehavior": "what should happen"
  }}
]

Focus on realistic business scenarios, not technical validations."""

    @staticmethod
    def generate_security_payload_prompt() -> str:
        """Generate prompt for security test payloads"""
        return """List 5 unique security test payloads for API testing.

Categories:
- SQL injection
- XSS
- Command injection
- Path traversal
- LDAP injection

Format (one per line):
payload_string

Output ONLY payloads, no explanations."""


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'IntelligentTestGenerator',
    'TestProfile',
    'TestFramework',
    'GeneratedTestSuite',
    'TestCase',
    'APIAnalysis',
    'AssertionRuleEngine',
    'TemplateEngine',
    'SmartPromptGenerator'
]