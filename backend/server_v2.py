# -*- coding: utf-8 -*-
"""
Enhanced API Autopilot Server V2
Production-ready implementation with all recommended improvements
"""

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, Depends
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any
import uuid
from datetime import datetime, timezone
import aiohttp

# Optional Redis support
try:
    import redis.asyncio as redis
except ImportError:
    redis = None
    logging.warning("Redis not installed - caching disabled. Install with: pip install redis[hiredis]")

# Import our enhanced modules
from test_generator_v2 import (
    IntelligentTestGenerator,
    TestProfile,
    TestFramework,
    SmartPromptGenerator
)
from api_discovery import (
    APIDiscovery,
    APISpecDetector
)
# Note: test_executor module is part of the existing codebase
# We'll keep this import as-is since it references an existing module
from test_executor import (
    TestExecutor,
    TestRunner,
    ExecutionEnvironment
)
from security_scanner import (
    SecurityScanner,
    QualityAnalyzer,
    ComplianceStandard
)
from ai_providers import generate_with_ai
from schema_analyzer import (
    SchemaAnalyzer,
    TestGenerator as SchemaTestGenerator,
    FieldSchema,
    FieldType
)
from load_test_generator import (
    LoadTestGenerator,
    LoadTestConfig,
    LoadTestFramework
)
from security_testing import (
    DASTScanner,
    OAuthJWTTester,
    APIFuzzer,
    DASTestResult,
    OAuthTestResult,
    FuzzTestResult
)
from test_execution_engine import (
    TestExecutor as NewTestExecutor,
    LoadTestExecutor,
    ExecutionStatus,
    TestFrameworkType
)

# Import authentication module
from auth import (
    AuthService,
    UserRegistration,
    UserLogin,
    TokenResponse,
    SavedScript,
    get_current_user_dependency
)

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE & CACHE SETUP
# ============================================================================

# MongoDB connection
try:
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    db_name = os.environ.get('DB_NAME', 'api_autopilot')
    mongo_client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
    db = mongo_client[db_name]
    logger.info(f"MongoDB connection configured: {mongo_url}/{db_name}")
except Exception as e:
    logger.error(f"Failed to configure MongoDB connection: {str(e)}")
    raise

# Redis cache connection
if redis:
    try:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        logger.info(f"Redis cache configured: {redis_url}")
    except Exception as e:
        logger.warning(f"Redis connection failed, using in-memory cache: {str(e)}")
        redis_client = None
else:
    logger.info("Redis module not installed - using in-memory cache")
    redis_client = None

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="API Autopilot V2",
    description="Enhanced AI-powered API test generation platform",
    version="2.0.0"
)

api_router = APIRouter(prefix="/api/v2")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class EnhancedTestGenerationRequest(BaseModel):
    """Enhanced test generation request with all new features"""
    # Input configuration
    input_type: Literal["curl", "har", "text", "graphql", "openapi", "postman", "auto"]
    input_data: str
    api_type: Optional[Literal["rest", "graphql"]] = "rest"

    # Test configuration
    test_profile: Optional[str] = "full_regression"  # quick_smoke, security_audit, etc.
    test_framework: str = "jest"
    module_system: Optional[Literal["commonjs", "esm"]] = "commonjs"

    # AI configuration
    ai_provider: Optional[str] = "openai"
    ai_model: Optional[str] = None
    ai_api_key: Optional[str] = None
    use_ai_enhancement: Optional[bool] = True  # Default to TRUE - always try to use AI
    temperature: Optional[float] = 0.1  # Lower for deterministic output

    # Advanced features
    auto_discover: Optional[bool] = False  # Auto-discover API spec
    auto_capture: Optional[bool] = True  # Execute API to capture response
    example_response: Optional[str] = None  # Example response JSON
    verify_ssl: Optional[bool] = True

    # Execution options
    execute_tests: Optional[bool] = False  # Run tests after generation
    generate_coverage: Optional[bool] = False  # Generate coverage report
    security_scan: Optional[bool] = True  # Run security scan
    compliance_check: Optional[str] = None  # Check specific compliance (owasp_top_10, pci_dss, etc.)


class EnhancedTestGenerationResponse(BaseModel):
    """Enhanced response with comprehensive metrics"""
    # Core response
    test_script: str
    framework: str
    success: bool

    # Quality metrics
    quality_score: Optional[float] = None
    security_score: Optional[float] = None
    test_count: Optional[int] = None
    coverage_report: Optional[Dict[str, Any]] = None

    # Execution results (if requested)
    execution_result: Optional[Dict[str, Any]] = None

    # Security & compliance
    security_issues: Optional[List[Dict[str, Any]]] = None
    compliance_report: Optional[Dict[str, Any]] = None

    # Metadata
    generation_time: Optional[float] = None
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    api_spec: Optional[Dict[str, Any]] = None
    generation_id: Optional[str] = None  # For tracking background generation


class APIDiscoveryRequest(BaseModel):
    """API discovery request"""
    base_url: str
    discover_spec: Optional[bool] = True
    crawl_html: Optional[bool] = False
    import_curl: Optional[str] = None


class TestExecutionRequest(BaseModel):
    """Test execution request"""
    test_code: str
    framework: str
    api_base_url: Optional[str] = None
    environment: Optional[str] = "local"  # local, docker, sandbox
    timeout: Optional[int] = 60


class SecurityScanRequest(BaseModel):
    """Security scan request"""
    test_code: str
    check_compliance: Optional[List[str]] = None  # List of compliance standards


class LoadTestGenerationRequest(BaseModel):
    """Load test generation request"""
    input_type: Literal["curl", "har", "openapi", "postman"]
    input_data: str
    framework: Literal["k6", "jmeter", "gatling", "artillery", "locust"] = "k6"
    scenario: Literal["smoke", "load", "stress", "spike", "soak"] = "load"
    vus: Optional[int] = 100
    duration: Optional[str] = "30s"
    ramp_up: Optional[str] = "10s"
    thresholds: Optional[Dict[str, str]] = None


class DASTScanRequest(BaseModel):
    """DAST scan request for an API endpoint"""
    url: str
    method: Optional[str] = "GET"
    headers: Optional[Dict[str, str]] = None
    body: Optional[Any] = None


class OAuthTestRequest(BaseModel):
    """OAuth/JWT testing request"""
    config: Dict[str, Any]  # OAuth configuration with endpoints, credentials, etc.
    test_jwt_security: Optional[bool] = True


class APIFuzzRequest(BaseModel):
    """API fuzzing request"""
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    body: Optional[Any] = None
    schema: Optional[Dict[str, Any]] = None  # Optional schema for guided fuzzing
    iterations: Optional[int] = 100


class TestExecutionRequest(BaseModel):
    """Request to execute generated test scripts"""
    test_code: str
    framework: str
    timeout: Optional[int] = 300
    environment: Optional[Dict[str, str]] = None
    api_base_url: Optional[str] = None  # Add missing field


class LoadTestExecutionRequest(BaseModel):
    """Request to execute load test scripts"""
    test_script: str
    framework: str
    duration: Optional[str] = "30s"
    vus: Optional[int] = 10


# ============================================================================
# CORE SERVICES
# ============================================================================

class APIAutopilotService:
    """Main service orchestrating all features"""

    def __init__(self, db=None):
        self.test_generator = IntelligentTestGenerator(TestFramework.JEST)
        self.api_discovery = APIDiscovery()
        self.spec_detector = APISpecDetector()
        self.test_runner = TestRunner()
        self.security_scanner = SecurityScanner()
        self.quality_analyzer = QualityAnalyzer()
        self.prompt_generator = SmartPromptGenerator()
        self.schema_analyzer = SchemaAnalyzer()
        self.schema_test_generator = SchemaTestGenerator()
        self.load_test_generator = LoadTestGenerator()
        self.dast_scanner = DASTScanner()
        self.oauth_tester = OAuthJWTTester()
        self.api_fuzzer = APIFuzzer()
        self.test_executor = NewTestExecutor()  # Use the new executor from test_execution_engine
        self.load_test_executor = LoadTestExecutor()
        self.cache = {}
        self.execution_sessions = {}  # Track active executions
        # Initialize auth service if database is provided
        self.auth_service = AuthService(db) if db is not None else None

    async def generate_tests_v2(self, request: EnhancedTestGenerationRequest) -> EnhancedTestGenerationResponse:
        """Generate tests with all enhanced features"""
        import time
        start_time = time.time()

        # Step 1: Parse input based on type
        if request.input_type == "auto":
            # Auto-detect input type
            spec_type, parsed_data = self.spec_detector.detect_spec_type(request.input_data)
            input_type = spec_type.value.lower()
        else:
            input_type = request.input_type
            parsed_data = self._parse_input(request.input_data, input_type)

        # Step 2: Auto-discover API spec if requested
        api_spec = None
        if request.auto_discover and "url" in parsed_data:
            base_url = self._extract_base_url(parsed_data["url"])
            api_spec = await self.api_discovery.discover_api_spec(base_url)

        # Step 3: Capture real API response if requested
        captured_response = None
        if request.auto_capture and parsed_data.get("url"):
            captured_response = await self._execute_api_call(parsed_data, request.verify_ssl)

        # Step 3.5: Analyze schema from response or contract
        inferred_schema = None
        schema_test_cases = []

        # If we have a captured response, infer schema from it
        if captured_response and captured_response.get("body"):
            logger.info("Inferring schema from captured API response")
            inferred_schema = self.schema_analyzer.infer_schema_from_response(captured_response["body"])

            # Generate comprehensive test cases based on inferred schema
            schema_test_cases = self.schema_test_generator.generate_test_cases(
                inferred_schema,
                include_security=True,
                include_boundary=True
            )
            logger.info(f"Generated {len(schema_test_cases)} schema-based test cases")

        # If input is OpenAPI/Swagger, parse the contract
        elif request.input_type in ["openapi", "swagger", "text"]:
            try:
                # Try to parse as OpenAPI/Swagger
                if request.input_type == "text":
                    # Check if text contains OpenAPI/Swagger spec
                    import yaml
                    try:
                        spec_data = json.loads(request.input_data)
                    except:
                        try:
                            spec_data = yaml.safe_load(request.input_data)
                        except:
                            spec_data = None

                    if spec_data and ("openapi" in spec_data or "swagger" in spec_data):
                        logger.info("Detected OpenAPI/Swagger specification in text input")
                        endpoint_schemas = self.schema_analyzer.parse_openapi_schema(spec_data)

                        # Get the first endpoint's request schema
                        if endpoint_schemas:
                            first_endpoint = list(endpoint_schemas.keys())[0]
                            inferred_schema = endpoint_schemas[first_endpoint]

                            # Generate test cases from contract schema
                            schema_test_cases = self.schema_test_generator.generate_test_cases(
                                inferred_schema,
                                include_security=True,
                                include_boundary=True
                            )
                            logger.info(f"Generated {len(schema_test_cases)} test cases from contract")
                else:
                    # Direct OpenAPI/Swagger input
                    spec_data = json.loads(request.input_data)
                    endpoint_schemas = self.schema_analyzer.parse_openapi_schema(spec_data)

                    if endpoint_schemas:
                        first_endpoint = list(endpoint_schemas.keys())[0]
                        inferred_schema = endpoint_schemas[first_endpoint]

                        schema_test_cases = self.schema_test_generator.generate_test_cases(
                            inferred_schema,
                            include_security=True,
                            include_boundary=True
                        )
                        logger.info(f"Generated {len(schema_test_cases)} test cases from OpenAPI spec")
            except Exception as e:
                logger.warning(f"Could not parse contract specification: {str(e)}")

        # Step 4: Set up test generator with appropriate framework
        framework = TestFramework[request.test_framework.upper()]
        self.test_generator.framework = framework

        # Pass AI credentials if provided
        if request.ai_api_key:
            self.test_generator.ai_provider = request.ai_provider or 'openai'
            self.test_generator.ai_api_key = request.ai_api_key
            self.test_generator.temperature = request.temperature or 0.1

        # Step 5: Check if we should use LLM
        test_suite = None  # Initialize test_suite
        used_llm = False

        if request.ai_api_key:
            # USE LLM TO GENERATE COMPLETE TEST SCRIPT
            try:
                from ai_providers import AIProviderFactory
                logger.info(f"Using AI provider: {request.ai_provider or 'openai'} with temperature: {request.temperature or 0.1}")

                # Initialize AI provider
                ai = AIProviderFactory.create_provider(
                    provider_name=request.ai_provider or 'openai',
                    api_key=request.ai_api_key,
                    temperature=request.temperature or 0.1
                )

                # Build comprehensive LLM prompt
                is_graphql = 'graphql' in parsed_data.get('url', '').lower() or 'query' in parsed_data.get('body', '').lower()

                # Detect authentication headers
                auth_headers = []
                headers = parsed_data.get('headers', {})
                auth_header_names = ['authorization', 'x-token', 'token', 'sessionid', 'x-api-key', 'api-key', 'x-auth-token', 'x-session-id', 'session-id']
                for header_name in headers.keys():
                    if header_name.lower() in auth_header_names:
                        auth_headers.append(header_name)

                auth_note = f"\nIMPORTANT: This API uses authentication via headers: {', '.join(auth_headers)}. Generate comprehensive auth tests!" if auth_headers else ""

                # Build schema information section if available
                schema_info = ""
                if inferred_schema:
                    schema_details = []
                    for field_name, field_schema in inferred_schema.items():
                        field_info = f"  - {field_name}: {field_schema.type.value}"
                        if field_schema.required:
                            field_info += " (REQUIRED)"
                        if field_schema.min_value is not None or field_schema.max_value is not None:
                            field_info += f" [range: {field_schema.min_value}-{field_schema.max_value}]"
                        if field_schema.min_length is not None or field_schema.max_length is not None:
                            field_info += f" [length: {field_schema.min_length}-{field_schema.max_length}]"
                        if field_schema.enum_values:
                            field_info += f" [enum: {', '.join(str(v) for v in field_schema.enum_values)}]"
                        if field_schema.format:
                            field_info += f" [format: {field_schema.format}]"
                        schema_details.append(field_info)

                    schema_info = f"\n\nINFERRED SCHEMA (use for precise type testing):\n" + "\n".join(schema_details)

                # Add schema test cases if available
                schema_tests_info = ""
                if schema_test_cases:
                    test_summaries = []
                    for test_case in schema_test_cases[:20]:  # Limit to first 20 for prompt
                        test_summaries.append(f"  - {test_case.name}: {test_case.description}")

                    schema_tests_info = f"\n\nSCHEMA-BASED TEST CASES TO IMPLEMENT:\n" + "\n".join(test_summaries)

                # Build the prompt string separately to avoid f-string nesting issues
                graphql_section = ""
                if is_graphql:
                    graphql_section = """
10. GRAPHQL SPECIFIC (if applicable):
   - Query depth limiting
   - Alias batching attacks
   - Introspection query security
   - Fragment spreading
   - Variable validation
   - Mutation atomicity
"""

                # Build the JSON examples as separate strings to avoid f-string nesting issues
                json_field_examples = """
   - Field type validation:
     * String field with number: {"name": 123}
     * Number field with string: {"age": "twenty"}
     * Boolean field with string: {"active": "true"}
     * Array field with object: {"items": {}}
     * Object field with array: {"user": []}
   - String field edge cases:
     * Empty string: {"name": ""}
     * Only spaces: {"name": "   "}
     * Very long string: {"name": "(very long string - 10000 a's)"}
     * Special chars: {"name": "'; DROP TABLE--"}
     * Unicode: {"name": "José 北京 🚀"}
   - Number field edge cases:
     * Zero: {"count": 0}
     * Negative: {"count": -1}
     * Decimal in int field: {"count": 3.14}
     * Very large: {"count": 999999999999}
     * String number: {"count": "123"}"""

                json_body_examples = """
- Example for a field "username" (string):
  * Valid: {"username": "john_doe"}
  * Empty: {"username": ""}
  * Special chars: {"username": "john'; DROP TABLE users--"}
  * Unicode: {"username": "José 北京 🚀"}
  * Too long: {"username": "(very long string - 10000 characters)"}
  * Wrong type: {"username": 123}
  * Null: {"username": null}
- Example for a field "age" (integer):
  * Valid: {"age": 25}
  * Zero: {"age": 0}
  * Negative: {"age": -1}
  * Max: {"age": 999999999999}
  * Decimal: {"age": 25.5}
  * String: {"age": "25"}
  * Null: {"age": null}"""

                prompt = f"""Generate a comprehensive, production-ready {request.test_framework} test suite for this API:

CRITICAL REQUIREMENTS:
- DO NOT USE NOCK OR ANY MOCKING LIBRARY
- USE ACTUAL API CALLS WITH AXIOS (const axios = require('axios'))
- Make real HTTP requests to the actual API endpoint
- Use simple Jest assertions or Joi for schema validation (const Joi = require('joi'))
- Never mock responses - use real API calls

API CONTEXT:
Type: {'GraphQL' if is_graphql else 'REST API'}
Method: {parsed_data.get('method', 'GET')}
Endpoint: {parsed_data.get('url', '')}
Headers: {json.dumps(parsed_data.get('headers', {}), indent=2)}{auth_note}
Request Body: {parsed_data.get('body', 'null')}

ACTUAL API RESPONSE (use for accurate assertions):
{json.dumps(captured_response, indent=2) if captured_response else 'Make actual API call to get response'}{schema_info}{schema_tests_info}

COMPREHENSIVE TEST REQUIREMENTS - MUST INCLUDE ALL:

1. SCHEMA VALIDATION TESTS:
   - Validate response structure with explicit assertions
   - Test for required fields: expect(response.data.field).toBeDefined()
   - Test field types: expect(typeof response.data.id).toBe('number')
   - Test string patterns: expect(response.data.url).toMatch(/^https?:\/\//)
   - Example:
     expect(response.data).toBeDefined();
     expect(response.data.id).toBeDefined();
     expect(typeof response.data.id).toBe('number');
     expect(typeof response.data.name).toBe('string');

2. POSITIVE TESTS (Happy Path):
   - Successful request with valid data
   - Explicit schema validation with Jest assertions
   - Data type validation for all fields
   - Response time validation (<2s)
   - Headers validation (Content-Type, CORS, etc.)

3. NEGATIVE TESTS (Error Handling):
   - 400 Bad Request - malformed payload
   - 404 Not Found - invalid endpoint/resource
   - 422 Unprocessable Entity - validation errors
   - Network timeout scenarios
   - Empty/null payload handling
   - Invalid data type testing

4. AUTHENTICATION TESTS:
   - 401 Unauthorized - missing auth token/header
   - 401 Unauthorized - invalid/expired token
   - 401 Unauthorized - malformed token (wrong format)
   - 403 Forbidden - valid token but insufficient permissions
   - Token refresh scenarios (if applicable)
   - Test with common auth headers: Authorization, x-token, token, sessionID, x-api-key, api-key, x-auth-token
   - Bearer token validation (if Authorization header present)

5. AUTHORIZATION TESTS:
   - Role-based access control (admin vs user)
   - Resource ownership validation
   - Scope-based permissions
   - Cross-tenant access prevention

6. SECURITY TESTS:
   - SQL Injection: ' OR '1'='1, admin'--, 1; DROP TABLE users
   - NoSQL Injection (for MongoDB): {{"$ne": null}}, {{"$gt": ""}}
   - XSS Prevention: <script>alert('XSS')</script>, javascript:alert(1)
   - Command Injection: ; ls -la, | whoami, && cat /etc/passwd
   - Path Traversal: ../../etc/passwd, ..\\..\\windows\\system32
   - XXE (if XML): <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
   - LDAP Injection: *, )(cn=*), )(|(cn=*))
   - Header Injection: \\r\\n\\r\\n<script>alert(1)</script>

7. EDGE CASES & BOUNDARIES:
   - Maximum payload size (>1MB)
   - Minimum values (0, -1, negative numbers)
   - Maximum values (MAX_INT, MAX_SAFE_INTEGER)
   - Empty strings vs null vs undefined vs ""
   - Special characters in STRING fields:
     * Quotes: " ' \" \'
     * Newlines and tabs: \n \r \t
     * Unicode: émojis 🚀🔥, Chinese 中文, Arabic العربية, RTL text
     * HTML/XML: <div>test</div>, &lt;&gt;&amp;
     * JSON special: {{"nested": "json"}}, [1,2,3]
     * Path separators: / \ ../
     * SQL chars: ' OR '1'='1
   - Special values for INTEGER fields:
     * Zero: 0
     * Negative: -1, -999999
     * MAX_SAFE_INTEGER: 9007199254740991
     * MIN_SAFE_INTEGER: -9007199254740991
     * Floating as int: 1.5, 3.14
     * Scientific notation: 1e10, 1.23e-4
     * Infinity, -Infinity, NaN
     * String numbers: "123", "0", "-1"
     * Hex/Octal: 0xFF, 0o77
   - Extremely long strings (>10000 chars)
   - Decimal precision edge cases
   - Date boundaries (past, future, invalid formats)

8. RATE LIMITING & PERFORMANCE:
   - Rate limit detection (429 Too Many Requests)
   - Concurrent request handling
   - Load testing pattern (if applicable)
   - Response time under load
   - Circuit breaker testing

9. DATA VALIDATION FOR JSON BODY FIELDS:
   - Required field validation (missing required fields)
{json_field_examples}
   - Format validation (email, URL, UUID, date)
   - Enum/choice field validation
   - Nested object validation
   - Array length constraints

{graphql_section}
TEST STRUCTURE:
- Use {request.test_framework} best practices
- Include setup/teardown for test isolation
- Use descriptive test names
- Add comments for complex assertions
- Include helper functions for reusability
- Mock external dependencies where needed
- Use environment variables for sensitive data

JSON BODY TESTING SPECIFICS:
- For each field in the request body, test:
  * Correct type with valid value
  * Wrong type (string instead of number, etc.)
  * Special characters in strings
  * Boundary values for numbers
  * Null, undefined, and empty values
{json_body_examples}

IMPORTANT:
- Generate ONLY the test code, no explanations
- Do NOT wrap the code in markdown blocks (no ```)
- Do NOT include ```javascript``` or any language tags
- Start directly with the code (imports first)
- Make it immediately runnable
- Include ALL necessary imports
- Use the actual API response data for accurate assertions
- Each test category MUST have at least 2-3 test cases
- NEVER USE NOCK - Use axios for actual API calls
- Use simple Jest assertions: expect(response.data.field).toBeDefined()
- For validation, use explicit checks: expect(typeof response.data.url).toBe('string')
- For complex validation, use Joi: const Joi = require('joi');
- Make real HTTP requests - DO NOT mock anything

Generate the COMPLETE test script now - pure code only, no markdown, no nock, use axios and simple assertions"""

                system_message = f"""You are an expert QA engineer specializing in comprehensive API testing, security testing, and test automation.
Your expertise includes: OWASP Top 10, authentication/authorization testing, injection attacks, performance testing, and {request.test_framework} best practices.
Generate a complete, production-ready test suite that would pass a security audit and provide maximum code coverage.
CRITICAL RULES:
1. Use simple Jest assertions: expect(response.data).toBeDefined(), expect(response.data.field).toBe(value)
2. For type checking: expect(typeof field).toBe('string')
3. For URL validation: expect(field).toMatch(/^https?:\\/\\//)
4. For complex validation, use Joi: const schema = Joi.object({{...}}).unknown(true); expect(schema.validate(data).error).toBeUndefined()
5. Output ONLY pure code without any markdown formatting, no ``` blocks, no language tags."""

                # Get complete test script from LLM
                logger.info("Calling LLM to generate test script...")
                test_code = await ai.generate(
                    prompt=prompt,
                    system_message=system_message
                )
                logger.info(f"LLM returned {len(test_code)} characters of test code")

                # Clean up the response (remove ALL markdown code blocks)
                import re

                # Remove markdown code blocks with language specifiers
                # Matches ```javascript, ```js, ```python, ```typescript, etc.
                test_code = re.sub(r'^```\w*\n', '', test_code, flags=re.MULTILINE)
                test_code = re.sub(r'^```$', '', test_code, flags=re.MULTILINE)
                
                # Post-process: Remove any legacy validation libraries that cause issues
                if 'Ajv' in test_code or 'ajv' in test_code:
                    logger.warning("Detected legacy validation library in generated code, removing it...")
                    # Remove legacy library imports
                    test_code = re.sub(r"const Ajv = require\(['\"]ajv['\"]\);?\s*", "", test_code)
                    test_code = re.sub(r"const ajv = new Ajv\(\);?\s*", "", test_code)
                    # Remove problematic format validators from schemas
                    test_code = re.sub(r",?\s*format:\s*['\"][^'\"]+['\"]", "", test_code)
                    # Remove compile and validation calls - replace with simple assertions
                    test_code = re.sub(r"const validateResponse = ajv\.compile\(responseSchema\);?\s*", "", test_code)
                    test_code = re.sub(r"expect\(validateResponse\([^)]+\)\)\.toBe\(true\);?", 
                                     "expect(response.data).toBeDefined(); // Simplified validation", test_code)
                    logger.info("Removed legacy validation code and format validators")

                # Also remove inline code blocks if they wrap the entire response
                if test_code.startswith("```"):
                    lines = test_code.split('\n')
                    # Remove first line if it starts with ```
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # Remove last line if it's just ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    test_code = '\n'.join(lines)

                # Clean up any leading/trailing whitespace
                test_code = test_code.strip()

                # Post-process to remove nock if it was generated despite instructions
                if 'nock' in test_code.lower():
                    logger.warning("Detected nock in generated code, removing it...")

                    # Remove nock import/require statements
                    test_code = re.sub(r"^.*require\(['\"]nock['\"]\).*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^.*from ['\"]nock['\"].*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^.*import.*nock.*$", "", test_code, flags=re.MULTILINE)

                    # Remove nock usage (mock definitions)
                    test_code = re.sub(r"^\s*nock\([^)]*\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*\.persist\(\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*\.get\([^)]*\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*\.post\([^)]*\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*\.put\([^)]*\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*\.delete\([^)]*\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*\.reply\([^)]*\)[^;]*;?\s*$", "", test_code, flags=re.MULTILINE)

                    # Clean up nock.cleanAll() calls
                    test_code = re.sub(r"^\s*nock\.cleanAll\(\);?\s*$", "", test_code, flags=re.MULTILINE)
                    test_code = re.sub(r"^\s*nock\.isDone\(\);?\s*$", "", test_code, flags=re.MULTILINE)

                    # Remove empty lines created by the above replacements
                    test_code = re.sub(r'\n\s*\n', '\n\n', test_code)

                    # If axios is not imported but nock was removed, add axios import
                    if 'axios' not in test_code and request.test_framework in ['jest', 'mocha', 'cypress']:
                        test_code = "const axios = require('axios');\n" + test_code

                    logger.info("Removed nock references from generated code")

                used_llm = True

            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                # Fallback to template generation
                profile = TestProfile[request.test_profile.upper()]
                test_suite = await self.test_generator.generate(
                    input_data=parsed_data,
                    input_type=input_type,
                    profile=profile,
                    captured_response=captured_response.get("body") if captured_response else None,
                    use_ai_enhancement=False
                )
                test_code = self.test_generator.generate_executable_code(test_suite)
        else:
            # No API key - use template-based generation
            profile = TestProfile[request.test_profile.upper()]
            test_suite = await self.test_generator.generate(
                input_data=parsed_data,
                input_type=input_type,
                profile=profile,
                captured_response=captured_response.get("body") if captured_response else None,
                use_ai_enhancement=False
            )
            test_code = self.test_generator.generate_executable_code(test_suite)

        # Step 7: Run security scan if requested
        security_issues = None
        security_score = None
        if request.security_scan:
            scan_result = await self.security_scanner.scan_test_code(test_code)
            security_score = scan_result.security_score
            security_issues = [
                {
                    "severity": issue.severity,
                    "category": issue.category.value,
                    "description": issue.description,
                    "line": issue.line_number
                }
                for issue in scan_result.issues
            ]

        # Step 8: Check compliance if requested
        compliance_report = None
        if request.compliance_check:
            standard = ComplianceStandard[request.compliance_check.upper()]
            compliance = await self.security_scanner.check_compliance(test_code, standard)
            compliance_report = {
                "standard": compliance.standard.value,
                "is_compliant": compliance.is_compliant,
                "score": compliance.compliance_score,
                "recommendations": compliance.recommendations
            }

        # Step 9: Analyze quality
        quality_metrics = self.quality_analyzer.analyze_test_quality(test_code, request.test_framework)

        # Step 10: Execute tests if requested
        execution_result = None
        if request.execute_tests:
            execution_report = await self.test_runner.run_and_report(
                test_code,
                request.test_framework,
                api_base_url=parsed_data.get("url"),
                generate_coverage=request.generate_coverage
            )
            execution_result = execution_report

        # Step 11: Build response
        generation_time = time.time() - start_time

        # Build appropriate response based on generation method
        if used_llm:
            # For LLM-generated tests, estimate metrics
            # Count tests based on framework
            if request.test_framework in ['jest', 'mocha', 'cypress']:
                test_count = test_code.count('it(') + test_code.count('test(')
            elif request.test_framework in ['pytest', 'requests']:
                test_count = test_code.count('def test_')
            elif request.test_framework in ['testng', 'junit', 'restassured']:
                test_count = test_code.count('@Test')
            elif request.test_framework == 'behave':
                test_count = test_code.count('Scenario:')
            else:
                # Fallback: try all patterns
                test_count = test_code.count('it(') + test_code.count('test(') + test_code.count('def test_') + test_code.count('@Test')
            coverage_report = {
                "total_tests": test_count,
                "categories": {
                    "positive": max(1, test_count // 3),
                    "negative": max(1, test_count // 3),
                    "edge": test_count - (2 * (test_count // 3))
                },
                "endpoints_covered": 1,
                "methods_covered": 1,
                "status_codes_covered": 3,
                "has_positive_tests": True,
                "has_negative_tests": test_count > 1,
                "has_security_tests": 'security' in test_code.lower() or 'injection' in test_code.lower(),
                "has_edge_tests": test_count > 2,
                "has_performance_tests": 'performance' in test_code.lower() or 'timeout' in test_code.lower()
            }
            warnings = []
            if test_count < 3:
                warnings.append("Consider adding more test cases for comprehensive coverage")
            elif test_count < 8:
                warnings.append("Test suite is minimal. Consider adding more negative, security, and edge case tests for better coverage")
        else:
            # Use test_suite data for template-generated tests
            test_count = test_suite.test_cases.__len__() if test_suite else 0
            coverage_report = test_suite.coverage_report if test_suite else {}
            warnings = test_suite.warnings if test_suite else []

        return EnhancedTestGenerationResponse(
            test_script=test_code,
            framework=request.test_framework,
            success=True,
            quality_score=quality_metrics["quality_score"],
            security_score=security_score,
            test_count=test_count,
            coverage_report=coverage_report,
            execution_result=execution_result,
            security_issues=security_issues,
            compliance_report=compliance_report,
            generation_time=generation_time,
            warnings=warnings,
            suggestions=quality_metrics.get("suggestions", []),
            api_spec=api_spec.__dict__ if api_spec else None
        )

    def _parse_input(self, input_data: str, input_type: str) -> Dict[str, Any]:
        """Parse input based on type"""
        if input_type == "curl":
            return self._parse_curl(input_data)
        elif input_type == "har":
            return self._parse_har(input_data)
        elif input_type == "graphql":
            return self._parse_graphql(input_data)
        elif input_type in ["openapi", "swagger"]:
            return json.loads(input_data)
        elif input_type == "postman":
            return json.loads(input_data)
        else:
            return {"description": input_data}

    def _parse_curl(self, curl_command: str) -> Dict[str, Any]:
        """Parse cURL command with improved handling"""
        import re

        # Remove line continuations and extra whitespace
        curl_command = curl_command.replace('\\\n', ' ').replace('\\', '')

        # Extract URL - handle --location flag properly
        # First try to find URL after --location flag
        url = ""
        location_match = re.search(r'--location\s+[\'"]([^\'"]+)[\'"]', curl_command)
        if location_match:
            url = location_match.group(1)
        else:
            # Fallback to standard curl URL parsing
            url_match = re.search(r'curl\s+(?:--location\s+)?(?:-X\s+\w+\s+)?[\'"]?([^\s\'"]+)', curl_command)
            if url_match and not url_match.group(1).startswith('-'):
                url = url_match.group(1)

        # Extract method
        method_match = re.search(r'-X\s+(\w+)', curl_command)
        method = method_match.group(1) if method_match else "GET"

        # If --data is present but no -X, default to POST
        if '--data' in curl_command and not method_match:
            method = "POST"

        # Extract headers
        headers = {}
        header_matches = re.finditer(r'(?:-H|--header)\s+[\'"]([^:]+):\s*([^\'"]+)[\'"]', curl_command)
        for match in header_matches:
            headers[match.group(1).strip()] = match.group(2).strip()

        # Extract body - handle various data formats
        body = ""
        # Try --data-raw first
        body_match = re.search(r"--data-raw\s+'([^']+)'", curl_command, re.DOTALL)
        if not body_match:
            body_match = re.search(r'--data-raw\s+"((?:[^"\\]|\\.)*)"', curl_command, re.DOTALL)

        # Then try --data
        if not body_match:
            body_match = re.search(r"--data\s+'([^']+)'", curl_command, re.DOTALL)
        if not body_match:
            body_match = re.search(r'--data\s+"((?:[^"\\]|\\.)*)"', curl_command, re.DOTALL)

        if body_match:
            body = body_match.group(1).strip()

        return {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body
        }

    def _parse_har(self, har_data: str) -> Dict[str, Any]:
        """Parse HAR file"""
        har_json = json.loads(har_data)
        entries = har_json.get("log", {}).get("entries", [])

        if entries:
            request = entries[0].get("request", {})
            return {
                "method": request.get("method", "GET"),
                "url": request.get("url", ""),
                "headers": {h["name"]: h["value"] for h in request.get("headers", [])},
                "body": request.get("postData", {}).get("text", "")
            }

        return {}

    def _parse_graphql(self, graphql_input: str) -> Dict[str, Any]:
        """Parse GraphQL query/mutation"""
        try:
            graphql_json = json.loads(graphql_input)
            return graphql_json
        except json.JSONDecodeError:
            return {"query": graphql_input}

    def _extract_base_url(self, url: str) -> str:
        """Extract base URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def _execute_api_call(self, request_data: Dict[str, Any], verify_ssl: bool) -> Optional[Dict[str, Any]]:
        """Execute API call to capture response"""
        try:
            method = request_data.get("method", "GET")
            url = request_data.get("url", "")
            headers = request_data.get("headers", {})
            body = request_data.get("body", "")

            # Parse body if it's a string
            json_body = None
            if body:
                try:
                    json_body = json.loads(body)
                except:
                    # Try fixing single quotes
                    try:
                        fixed_body = body.replace("'", '"')
                        json_body = json.loads(fixed_body)
                    except:
                        json_body = None

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    timeout=aiohttp.ClientTimeout(total=30),
                    ssl=verify_ssl
                ) as response:
                    try:
                        response_body = await response.json()
                    except:
                        response_body = await response.text()

                    return {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "body": response_body
                    }
        except Exception as e:
            logger.error(f"Failed to execute API call: {str(e)}")
            return None


# Initialize service with database
service = APIAutopilotService(db=db)

# Store for background generation results
generation_results = {}

async def complete_test_generation_async(
    request: EnhancedTestGenerationRequest,
    generation_id: str,
    service: APIAutopilotService,
    redis_client
):
    """Complete test generation in background"""
    try:
        logger.info(f"Starting background generation for {generation_id}")
        response = await service.generate_tests_v2(request)

        # Store result
        generation_results[generation_id] = response.dict()

        # Also cache in Redis if available
        if redis_client:
            try:
                await redis_client.setex(
                    f"generation:{generation_id}",
                    3600,  # 1 hour TTL
                    json.dumps(response.dict())
                )
            except Exception as e:
                logger.warning(f"Failed to cache background result: {str(e)}")

        logger.info(f"Completed background generation for {generation_id}")
    except Exception as e:
        logger.error(f"Background generation failed for {generation_id}: {str(e)}")
        generation_results[generation_id] = {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(registration: UserRegistration):
    """Register a new user"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        token_response = await service.auth_service.register_user(registration)
        return token_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(login_data: UserLogin):
    """Login and get JWT token"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        token_response = await service.auth_service.login_user(login_data)
        return token_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@api_router.get("/auth/me")
async def get_current_user(current_user: dict = Depends(get_current_user_dependency)):
    """Get current user information"""
    return {
        "success": True,
        "user": current_user
    }

# ============================================================================
# PROTECTED USER ENDPOINTS
# ============================================================================

@api_router.post("/user/scripts")
async def save_user_script(
    script: SavedScript,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Save a test script for the authenticated user"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        result = await service.auth_service.save_script(current_user["id"], script)

        # Also return the saved script for immediate display
        if result.get("success"):
            # Include the saved script in the response
            saved_script = {
                "id": script.id,
                "name": script.name,
                "script": script.script,
                "framework": script.framework,
                "timestamp": script.timestamp
            }
            result["script"] = saved_script

        return result
    except Exception as e:
        logger.error(f"Error saving script: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save script")

@api_router.get("/user/scripts")
async def get_user_scripts(current_user: dict = Depends(get_current_user_dependency)):
    """Get all saved scripts for the authenticated user"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        scripts = await service.auth_service.get_user_scripts(current_user["id"])
        return {
            "success": True,
            "scripts": scripts
        }
    except Exception as e:
        logger.error(f"Error fetching scripts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch scripts")

@api_router.get("/user/scripts/{script_id}")
async def get_script_content(
    script_id: str,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Get full content of a specific script"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        script = await service.auth_service.get_script_content(current_user["id"], script_id)
        return {
            "success": True,
            "script": script
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching script content: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch script")

@api_router.put("/user/scripts/{script_id}")
async def update_user_script(
    script_id: str,
    script: SavedScript,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Update a user's script (e.g., rename)"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        # Update the script in the database
        from bson import ObjectId
        result = await db.scripts.update_one(
            {"_id": ObjectId(script_id), "user_id": current_user["id"]},
            {"$set": {"name": script.name, "updated_at": datetime.now(timezone.utc)}}
        )

        if result.modified_count > 0:
            return {"success": True, "message": "Script updated"}
        else:
            raise HTTPException(status_code=404, detail="Script not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating script: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update script")

@api_router.delete("/user/scripts/{script_id}")
async def delete_user_script(
    script_id: str,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Delete a user's script"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        result = await service.auth_service.delete_script(current_user["id"], script_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting script: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete script")

@api_router.post("/user/api-keys/{provider}")
async def save_api_key(
    provider: str,
    api_key: str,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Save an encrypted API key for the user"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        result = await service.auth_service.save_api_key(current_user["id"], provider, api_key)
        return result
    except Exception as e:
        logger.error(f"Error saving API key: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save API key")

@api_router.get("/user/api-keys/{provider}")
async def get_api_key(
    provider: str,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Get decrypted API key for the user"""
    try:
        if not service.auth_service:
            raise HTTPException(status_code=500, detail="Authentication service not available")

        api_key = await service.auth_service.get_api_key(current_user["id"], provider)
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")

        return {
            "success": True,
            "provider": provider,
            "api_key": api_key
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching API key: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch API key")

# ============================================================================
# DASHBOARD ENDPOINTS (for user statistics and data)
# ============================================================================

@api_router.get("/user/stats")
async def get_user_stats(current_user: dict = Depends(get_current_user_dependency)):
    """Get user statistics for dashboard"""
    try:
        # For enhanced auth service, we need to import and use it
        from auth_enhanced import EnhancedAuthService

        # Initialize enhanced service if available
        enhanced_service = EnhancedAuthService(db)

        # Get user statistics
        stats = await enhanced_service.get_user_statistics(current_user["id"])

        return {
            "totalTests": stats.total_tests,
            "testsThisMonth": stats.tests_this_month,
            "totalExecutions": stats.total_executions,
            "successRate": stats.success_rate,
            "savedScripts": stats.saved_scripts,
            "scriptsShared": stats.scripts_shared,
            "apiKeysStored": stats.api_keys_stored,
            "frameworkUsage": stats.framework_usage,
            "lastActivity": stats.last_activity.isoformat() if stats.last_activity else None
        }
    except Exception as e:
        logger.error(f"Error fetching user stats: {str(e)}")
        # Return default values if there's an error
        return {
            "totalTests": 0,
            "testsThisMonth": 0,
            "totalExecutions": 0,
            "successRate": 0,
            "savedScripts": 0,
            "scriptsShared": 0,
            "apiKeysStored": 0,
            "frameworkUsage": [],
            "lastActivity": None
        }

@api_router.get("/user/execution-history")
async def get_user_execution_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Get execution history for dashboard"""
    try:
        from auth_enhanced import EnhancedAuthService

        enhanced_service = EnhancedAuthService(db)
        history = await enhanced_service.get_execution_history(current_user["id"], limit=limit)

        return {
            "history": [
                {
                    "id": h.id,
                    "scriptName": h.script_name,
                    "framework": h.framework,
                    "status": h.status,
                    "totalTests": h.total_tests,
                    "passedTests": h.passed_tests,
                    "failedTests": h.failed_tests,
                    "duration": h.duration,
                    "timestamp": h.timestamp.isoformat() if h.timestamp else None
                }
                for h in history
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching execution history: {str(e)}")
        return {"history": []}

@api_router.get("/user/preferences")
async def get_user_preferences(current_user: dict = Depends(get_current_user_dependency)):
    """Get user preferences for dashboard"""
    try:
        from auth_enhanced import EnhancedAuthService

        enhanced_service = EnhancedAuthService(db)
        preferences = await enhanced_service.get_user_preferences(current_user["id"])

        return {
            "preferences": {
                "defaultFramework": preferences.default_framework,
                "defaultAiProvider": preferences.default_ai_provider,
                "defaultAiModel": preferences.default_ai_model,
                "theme": preferences.theme,
                "emailNotifications": preferences.email_notifications,
                "twoFactorEnabled": preferences.two_factor_enabled
            }
        }
    except Exception as e:
        logger.error(f"Error fetching user preferences: {str(e)}")
        return {
            "preferences": {
                "defaultFramework": "jest",
                "defaultAiProvider": "openai",
                "defaultAiModel": "gpt-4o",
                "theme": "light",
                "emailNotifications": True,
                "twoFactorEnabled": False
            }
        }

@api_router.put("/user/profile")
async def update_user_profile(
    profile_data: dict,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Update user profile information"""
    try:
        # Update user profile in database
        from bson import ObjectId

        result = await db.users.update_one(
            {"_id": ObjectId(current_user["id"])},
            {
                "$set": {
                    "name": profile_data.get("name", current_user.get("name")),
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

        if result.modified_count > 0:
            # Get updated user
            updated_user = await db.users.find_one({"_id": ObjectId(current_user["id"])})
            return {
                "success": True,
                "user": {
                    "id": str(updated_user["_id"]),
                    "email": updated_user["email"],
                    "name": updated_user.get("name", ""),
                    "created_at": updated_user.get("created_at")
                }
            }

        return {"success": False, "message": "No changes made"}

    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@api_router.get("/user/api-keys")
async def get_all_api_keys(current_user: dict = Depends(get_current_user_dependency)):
    """Get all encrypted API keys for dashboard"""
    try:
        from bson import ObjectId

        user = await db.users.find_one({"_id": ObjectId(current_user["id"])})
        api_keys = {}

        if user and "api_keys" in user:
            # Return masked versions of API keys
            for provider, encrypted_key in user["api_keys"].items():
                # Show only last 4 characters
                api_keys[provider] = {
                    "key": "••••••••" + encrypted_key[-4:] if len(encrypted_key) > 4 else "••••••••",
                    "hasKey": True
                }

        return {"apiKeys": api_keys}

    except Exception as e:
        logger.error(f"Error fetching API keys: {str(e)}")
        return {"apiKeys": {}}

@api_router.get("/user/sessions")
async def get_user_sessions(current_user: dict = Depends(get_current_user_dependency)):
    """Get active sessions for security dashboard"""
    try:
        from auth_enhanced import EnhancedAuthService

        enhanced_service = EnhancedAuthService(db)
        sessions = await enhanced_service.get_user_sessions(current_user["id"])

        return {
            "sessions": [
                {
                    "id": s.id,
                    "device": s.device,
                    "ipAddress": s.ip_address,
                    "location": s.location,
                    "createdAt": s.created_at.isoformat() if s.created_at else None,
                    "lastActive": s.last_active.isoformat() if s.last_active else None,
                    "current": s.is_current
                }
                for s in sessions
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching sessions: {str(e)}")
        return {"sessions": []}

@api_router.delete("/user/sessions/{session_id}")
async def revoke_user_session(
    session_id: str,
    current_user: dict = Depends(get_current_user_dependency)
):
    """Revoke a specific session"""
    try:
        from auth_enhanced import EnhancedAuthService

        enhanced_service = EnhancedAuthService(db)
        result = await enhanced_service.revoke_session(current_user["id"], session_id)

        return {"success": result}

    except Exception as e:
        logger.error(f"Error revoking session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to revoke session")

@api_router.post("/auth/reset-password-request")
async def request_password_reset(email: str):
    """Request password reset email"""
    try:
        from auth_enhanced import EnhancedAuthService

        enhanced_service = EnhancedAuthService(db)
        await enhanced_service.request_password_reset(email)

        return {
            "success": True,
            "message": "If the email exists, a password reset link has been sent"
        }

    except Exception as e:
        logger.error(f"Error requesting password reset: {str(e)}")
        # Always return success to prevent email enumeration
        return {
            "success": True,
            "message": "If the email exists, a password reset link has been sent"
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@api_router.post("/generate-tests", response_model=EnhancedTestGenerationResponse)
async def generate_tests_v2(request: EnhancedTestGenerationRequest, background_tasks: BackgroundTasks):
    """Generate tests with enhanced features"""
    try:
        # Check cache first
        if redis_client:
            try:
                cache_key = f"test:{json.dumps(request.dict(), sort_keys=True)}"
                cached = await redis_client.get(cache_key)
                if cached:
                    logger.info("Returning cached test suite")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache read failed, continuing without cache: {str(e)}")

        # Set timeout for long-running generations
        import asyncio
        try:
            # Generate tests with timeout
            response = await asyncio.wait_for(
                service.generate_tests_v2(request),
                timeout=120.0  # 2 minutes timeout
            )
        except asyncio.TimeoutError:
            # Return partial result and continue in background
            logger.warning("Test generation timed out, continuing in background")
            generation_id = str(uuid.uuid4())

            # Start background generation
            background_tasks.add_task(
                complete_test_generation_async,
                request,
                generation_id,
                service,
                redis_client
            )

            return EnhancedTestGenerationResponse(
                test_script="// Test generation in progress...\n// Please wait, this may take a few minutes for complex APIs.\n// Generation ID: " + generation_id,
                framework=request.test_framework,
                success=True,
                warnings=["Test generation is taking longer than expected. Processing in background. Check back in a moment."],
                generation_id=generation_id
            )

        # Cache the result
        if redis_client and response.success:
            try:
                await redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(response.dict())
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {str(e)}")

        return response

    except Exception as e:
        logger.error(f"Error generating tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/generation-status/{generation_id}")
async def get_generation_status(generation_id: str):
    """Check status of background test generation"""
    # Check in-memory store first
    if generation_id in generation_results:
        return generation_results[generation_id]

    # Check Redis
    if redis_client:
        try:
            cached = await redis_client.get(f"generation:{generation_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Failed to check Redis for generation status: {str(e)}")

    return {
        "success": False,
        "status": "pending",
        "message": "Generation still in progress or not found"
    }


@api_router.post("/discover-api")
async def discover_api(request: APIDiscoveryRequest):
    """Discover API specification from URL"""
    try:
        discovery = APIDiscovery()

        # Discover API spec
        api_spec = None
        if request.discover_spec:
            api_spec = await discovery.discover_api_spec(request.base_url)

        # Crawl HTML for endpoints
        endpoints = []
        if request.crawl_html:
            endpoints = await discovery.discover_from_html(request.base_url)

        # Import from cURL
        imported_endpoint = None
        if request.import_curl:
            imported_endpoint = discovery.import_from_curl(request.import_curl)

        return {
            "api_spec": api_spec.__dict__ if api_spec else None,
            "discovered_endpoints": endpoints,
            "imported_endpoint": imported_endpoint.__dict__ if imported_endpoint else None
        }

    except Exception as e:
        logger.error(f"Error discovering API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Removed duplicate /execute-tests endpoint - using execute_tests_v2 below


@api_router.post("/security-scan")
async def security_scan(request: SecurityScanRequest):
    """Perform security scan on test code"""
    try:
        scanner = SecurityScanner()

        # Run security scan
        scan_result = await scanner.scan_test_code(request.test_code)

        # Check compliance if requested
        compliance_reports = []
        if request.check_compliance:
            for standard_name in request.check_compliance:
                try:
                    standard = ComplianceStandard[standard_name.upper()]
                    report = await scanner.check_compliance(request.test_code, standard)
                    compliance_reports.append({
                        "standard": standard.value,
                        "is_compliant": report.is_compliant,
                        "score": report.compliance_score,
                        "recommendations": report.recommendations
                    })
                except KeyError:
                    logger.warning(f"Unknown compliance standard: {standard_name}")

        return {
            "security_score": scan_result.security_score,
            "total_issues": scan_result.total_issues,
            "critical_issues": scan_result.critical_issues,
            "high_issues": scan_result.high_issues,
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category.value,
                    "description": issue.description,
                    "line": issue.line_number,
                    "recommendation": issue.recommendation
                }
                for issue in scan_result.issues[:10]  # Limit to first 10
            ],
            "recommendations": scan_result.recommendations,
            "compliance_reports": compliance_reports
        }

    except Exception as e:
        logger.error(f"Error performing security scan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/generate-load-tests")
async def generate_load_tests(request: LoadTestGenerationRequest):
    """Generate load test scripts for various frameworks"""
    try:
        # Parse input data
        if request.input_type == "curl":
            api_data = service._parse_curl(request.input_data)
        elif request.input_type == "har":
            api_data = service._parse_har(request.input_data)
        elif request.input_type in ["openapi", "postman"]:
            api_data = json.loads(request.input_data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported input type")

        # Configure load test
        config = LoadTestConfig(
            framework=LoadTestFramework[request.framework.upper()],
            scenario=request.scenario,  # Use scenario name directly
            vus=request.vus,
            duration=request.duration,
            ramp_up=request.ramp_up,
            thresholds=request.thresholds or {}
        )

        # Generate load test script
        load_test_script = service.load_test_generator.generate(api_data, config)

        return {
            "success": True,
            "framework": request.framework,
            "scenario": request.scenario,
            "script": load_test_script,
            "config": {
                "vus": config.vus,
                "duration": config.duration,
                "ramp_up": config.ramp_up,
                "thresholds": config.thresholds
            }
        }

    except Exception as e:
        logger.error(f"Error generating load tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/dast-scan")
async def run_dast_scan(request: DASTScanRequest):
    """Run DAST scan on an API endpoint"""
    try:
        # Run DAST scan
        results = await service.dast_scanner.scan_endpoint(
            url=request.url,
            method=request.method,
            headers=request.headers,
            body=request.body
        )

        # Format results
        vulnerabilities = []
        for result in results:
            vulnerabilities.append({
                "vulnerability": result.vulnerability,
                "severity": result.severity,
                "payload": result.payload,
                "response": result.response,
                "recommendation": result.recommendation
            })

        # Calculate security score
        critical = sum(1 for r in results if r.severity == "CRITICAL")
        high = sum(1 for r in results if r.severity == "HIGH")
        medium = sum(1 for r in results if r.severity == "MEDIUM")
        low = sum(1 for r in results if r.severity == "LOW")

        score = max(0, 100 - (critical * 30) - (high * 20) - (medium * 10) - (low * 5))

        return {
            "success": True,
            "url": request.url,
            "method": request.method,
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total": len(results),
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low
            },
            "security_score": score
        }

    except Exception as e:
        logger.error(f"Error running DAST scan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/oauth-test")
async def test_oauth_flow(request: OAuthTestRequest):
    """Test OAuth/JWT authentication flows"""
    try:
        # Run OAuth flow tests
        oauth_results = await service.oauth_tester.test_oauth_flow(request.config)

        # Test JWT security if requested
        jwt_results = []
        if request.test_jwt_security and request.config.get("sample_jwt"):
            jwt_results = await service.oauth_tester.test_jwt_security(
                request.config["sample_jwt"],
                request.config.get("jwt_secret")
            )

        # Format results
        oauth_tests = []
        for result in oauth_results:
            oauth_tests.append({
                "test_name": result.test_name,
                "passed": result.passed,
                "message": result.message,
                "details": result.details
            })

        jwt_tests = []
        for result in jwt_results:
            jwt_tests.append({
                "test_name": result.test_name,
                "passed": result.passed,
                "message": result.message,
                "details": result.details
            })

        # Calculate success rate
        total_tests = len(oauth_tests) + len(jwt_tests)
        passed_tests = sum(1 for t in oauth_tests if t["passed"]) + sum(1 for t in jwt_tests if t["passed"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        return {
            "success": True,
            "oauth_tests": oauth_tests,
            "jwt_tests": jwt_tests,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": success_rate
            }
        }

    except Exception as e:
        logger.error(f"Error testing OAuth flow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/api-fuzz")
async def fuzz_api_endpoint(request: APIFuzzRequest):
    """Fuzz an API endpoint with schema-aware payloads"""
    try:
        # Run fuzzing
        results = await service.api_fuzzer.fuzz_endpoint(
            url=request.url,
            method=request.method,
            headers=request.headers,
            body=request.body,
            schema=request.schema,
            iterations=request.iterations
        )

        # Analyze results
        issues = []
        for result in results:
            if result.is_anomaly:
                issues.append({
                    "payload": result.payload,
                    "status_code": result.status_code,
                    "response_time": result.response_time,
                    "error": result.error,
                    "anomaly_type": result.anomaly_type
                })

        # Calculate robustness score
        error_rate = (len(issues) / len(results) * 100) if results else 0
        robustness_score = max(0, 100 - error_rate)

        return {
            "success": True,
            "total_tests": len(results),
            "anomalies_found": len(issues),
            "robustness_score": robustness_score,
            "issues": issues[:50],  # Limit to first 50 issues
            "summary": {
                "crash_errors": sum(1 for i in issues if "500" in str(i.get("status_code", ""))),
                "validation_errors": sum(1 for i in issues if i.get("anomaly_type") == "validation_error"),
                "timeout_errors": sum(1 for i in issues if i.get("anomaly_type") == "timeout"),
                "unexpected_responses": sum(1 for i in issues if i.get("anomaly_type") == "unexpected_response")
            }
        }

    except Exception as e:
        logger.error(f"Error fuzzing API: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/execute-tests")
async def execute_tests_v2(request: TestExecutionRequest):
    """Execute generated test scripts and stream results"""
    try:
        execution_id = str(uuid.uuid4())
        results = []
        logger.info(f"Starting test execution for framework: {request.framework}, execution_id: {execution_id}")

        # Execute tests and collect results
        async for update in service.test_executor.execute_tests(
            test_code=request.test_code,
            framework=request.framework,
            execution_id=execution_id,
            timeout=request.timeout,
            environment=request.environment,
            stream_output=True
        ):
            logger.info(f"Received update type: {update.get('type')}")
            results.append(update)

            # If it's the final result, return it
            if update.get("type") == "result":
                logger.info("Found result, returning success")
                return {
                    "success": True,
                    "execution_id": execution_id,
                    **update["data"]
                }
            elif update.get("type") == "error":
                logger.error(f"Execution error: {update.get('message')}")
                return {
                    "success": False,
                    "execution_id": execution_id,
                    "error": update.get("message", "Unknown error")
                }

        # If no result found, return error
        logger.warning(f"No result found after {len(results)} updates")
        return {
            "success": False,
            "execution_id": execution_id,
            "error": f"Test execution completed without results. Received {len(results)} updates."
        }

    except Exception as e:
        logger.error(f"Error executing tests: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/execute-load-tests")
async def execute_load_tests(request: LoadTestExecutionRequest):
    """Execute load test scripts with metrics streaming"""
    try:
        execution_id = str(uuid.uuid4())
        results = []

        # Execute load tests and collect metrics
        async for update in service.load_test_executor.execute_load_test(
            test_script=request.test_script,
            framework=request.framework,
            duration=request.duration,
            vus=request.vus,
            stream_output=True
        ):
            results.append(update)

            # If it's the final result, return it
            if update.get("type") == "result":
                return {
                    "success": True,
                    "execution_id": execution_id,
                    **update["data"]
                }

        return {
            "success": False,
            "execution_id": execution_id,
            "error": "Load test execution completed without results"
        }

    except Exception as e:
        logger.error(f"Error executing load tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.websocket("/ws/execute-tests")
async def websocket_test_execution_v2(websocket: WebSocket):
    """WebSocket endpoint for real-time test execution"""
    await websocket.accept()
    execution_id = str(uuid.uuid4())

    try:
        # Store websocket session
        service.execution_sessions[execution_id] = websocket

        while True:
            # Receive execution request
            data = await websocket.receive_json()

            if data.get("action") == "execute":
                # Stream execution updates
                async for update in service.test_executor.execute_tests(
                    test_code=data["test_code"],
                    framework=data["framework"],
                    execution_id=execution_id,
                    timeout=data.get("timeout", 300),
                    stream_output=True
                ):
                    await websocket.send_json(update)

            elif data.get("action") == "execute_load":
                # Stream load test updates
                async for update in service.load_test_executor.execute_load_test(
                    test_script=data["test_script"],
                    framework=data["framework"],
                    duration=data.get("duration", "30s"),
                    vus=data.get("vus", 10),
                    stream_output=True
                ):
                    await websocket.send_json(update)

            elif data.get("action") == "cancel":
                # Cancel execution
                cancelled = await service.test_executor.cancel_execution(execution_id)
                await websocket.send_json({
                    "type": "cancelled",
                    "execution_id": execution_id,
                    "success": cancelled
                })

    except WebSocketDisconnect:
        # Client disconnected - this is normal, just log it
        logger.info(f"WebSocket disconnected for execution {execution_id}")
    except Exception as e:
        # Try to send error if connection is still open
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "execution_id": execution_id
            })
        except (WebSocketDisconnect, RuntimeError):
            # Connection already closed, can't send error
            logger.warning(f"Could not send error to client for execution {execution_id}: {str(e)}")
    finally:
        # Clean up session
        if execution_id in service.execution_sessions:
            del service.execution_sessions[execution_id]
        # Try to close if not already closed
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            # Already closed, no problem
            pass


@api_router.get("/execution-logs/{execution_id}")
async def get_execution_logs(execution_id: str):
    """Get logs for a specific test execution"""
    try:
        logs = await service.test_executor.get_execution_logs(execution_id)
        return {
            "success": True,
            "execution_id": execution_id,
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/cancel-execution/{execution_id}")
async def cancel_execution(execution_id: str):
    """Cancel a running test execution"""
    try:
        cancelled = await service.test_executor.cancel_execution(execution_id)
        return {
            "success": cancelled,
            "execution_id": execution_id,
            "message": "Execution cancelled" if cancelled else "Execution not found"
        }
    except Exception as e:
        logger.error(f"Error cancelling execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/test-profiles")
async def get_test_profiles():
    """Get available test profiles"""
    return {
        "profiles": [
            {
                "id": "quick_smoke",
                "name": "Quick Smoke",
                "description": "Basic positive and auth tests (5-10 tests)",
                "categories": ["positive", "auth"],
                "test_count": "5-10"
            },
            {
                "id": "security_audit",
                "name": "Security Audit",
                "description": "Comprehensive security testing (30-50 tests)",
                "categories": ["security", "injection", "auth", "encryption"],
                "test_count": "30-50"
            },
            {
                "id": "full_regression",
                "name": "Full Regression",
                "description": "Complete test coverage (50-100 tests)",
                "categories": ["positive", "negative", "security", "edge", "performance"],
                "test_count": "50-100"
            },
            {
                "id": "ci_cd",
                "name": "CI/CD Pipeline",
                "description": "Balanced tests for continuous integration (15-20 tests)",
                "categories": ["positive", "critical_negative", "auth"],
                "test_count": "15-20"
            }
        ]
    }


@api_router.get("/frameworks")
async def get_frameworks():
    """Get supported test frameworks"""
    return {
        "frameworks": [
            {"id": "jest", "name": "Jest", "language": "JavaScript", "type": "unit"},
            {"id": "mocha", "name": "Mocha", "language": "JavaScript", "type": "unit"},
            {"id": "cypress", "name": "Cypress", "language": "JavaScript", "type": "e2e"},
            {"id": "pytest", "name": "pytest", "language": "Python", "type": "unit"},
            {"id": "requests", "name": "Requests + unittest", "language": "Python", "type": "unit"},
            {"id": "testng", "name": "TestNG", "language": "Java", "type": "unit"},
            {"id": "junit", "name": "JUnit", "language": "Java", "type": "unit"}
        ]
    }


@api_router.get("/compliance-standards")
async def get_compliance_standards():
    """Get available compliance standards"""
    return {
        "standards": [
            {"id": "owasp_top_10", "name": "OWASP Top 10", "description": "Web application security risks"},
            {"id": "pci_dss", "name": "PCI DSS", "description": "Payment card industry standards"},
            {"id": "gdpr", "name": "GDPR", "description": "General Data Protection Regulation"},
            {"id": "hipaa", "name": "HIPAA", "description": "Health Insurance Portability and Accountability Act"},
            {"id": "soc2", "name": "SOC 2", "description": "Service Organization Control 2"},
            {"id": "iso_27001", "name": "ISO 27001", "description": "Information security management"},
            {"id": "nist", "name": "NIST", "description": "National Institute of Standards and Technology"},
            {"id": "cwe_top_25", "name": "CWE Top 25", "description": "Most dangerous software weaknesses"}
        ]
    }


# Removed duplicate /ws/test-execution endpoint - using websocket_test_execution_v2 above


@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if Redis is actually working
    redis_working = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_working = True
        except:
            redis_working = False

    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "mongodb": mongo_client is not None,
            "redis": redis_working
        }
    }


# ============================================================================
# APP CONFIGURATION
# ============================================================================

# Include router
app.include_router(api_router)

# Configure CORS
cors_origins = os.environ.get('CORS_ORIGINS', '*')
if cors_origins == '*':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(','),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("API Autopilot V2 starting up...")

    # Test MongoDB connection
    try:
        await mongo_client.admin.command('ping')
        logger.info("MongoDB connection successful")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")

    # Test Redis connection
    if redis_client:
        try:
            await redis_client.ping()
            logger.info("Redis connection successful")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API Autopilot V2 shutting down...")

    # Close MongoDB connection
    mongo_client.close()

    # Close Redis connection
    if redis_client:
        await redis_client.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )