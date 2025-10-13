# -*- coding: utf-8 -*-
"""
Enhanced API Autopilot Server V2
Production-ready implementation with all recommended improvements
"""

from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket
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


# ============================================================================
# CORE SERVICES
# ============================================================================

class APIAutopilotService:
    """Main service orchestrating all features"""

    def __init__(self):
        self.test_generator = IntelligentTestGenerator(TestFramework.JEST)
        self.api_discovery = APIDiscovery()
        self.spec_detector = APISpecDetector()
        self.test_runner = TestRunner()
        self.security_scanner = SecurityScanner()
        self.quality_analyzer = QualityAnalyzer()
        self.prompt_generator = SmartPromptGenerator()
        self.cache = {}

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

                prompt = f"""Generate a comprehensive, production-ready {request.test_framework} test suite for this API:

API CONTEXT:
Type: {'GraphQL' if is_graphql else 'REST API'}
Method: {parsed_data.get('method', 'GET')}
Endpoint: {parsed_data.get('url', '')}
Headers: {json.dumps(parsed_data.get('headers', {}), indent=2)}{auth_note}
Request Body: {parsed_data.get('body', 'null')}

ACTUAL API RESPONSE (use for accurate assertions):
{json.dumps(captured_response, indent=2) if captured_response else 'No response captured - create realistic mock data'}

COMPREHENSIVE TEST REQUIREMENTS - MUST INCLUDE ALL:

1. POSITIVE TESTS (Happy Path):
   - Successful request with valid data
   - Response schema validation (use actual response structure)
   - Data type validation for all fields
   - Response time validation (<2s)
   - Headers validation (Content-Type, CORS, etc.)

2. NEGATIVE TESTS (Error Handling):
   - 400 Bad Request - malformed payload
   - 404 Not Found - invalid endpoint/resource
   - 422 Unprocessable Entity - validation errors
   - Network timeout scenarios
   - Empty/null payload handling
   - Invalid data type testing

3. AUTHENTICATION TESTS:
   - 401 Unauthorized - missing auth token/header
   - 401 Unauthorized - invalid/expired token
   - 401 Unauthorized - malformed token (wrong format)
   - 403 Forbidden - valid token but insufficient permissions
   - Token refresh scenarios (if applicable)
   - Test with common auth headers: Authorization, x-token, token, sessionID, x-api-key, api-key, x-auth-token
   - Bearer token validation (if Authorization header present)

4. AUTHORIZATION TESTS:
   - Role-based access control (admin vs user)
   - Resource ownership validation
   - Scope-based permissions
   - Cross-tenant access prevention

5. SECURITY TESTS:
   - SQL Injection: ' OR '1'='1, admin'--, 1; DROP TABLE users
   - NoSQL Injection (for MongoDB): {{"$ne": null}}, {{"$gt": ""}}
   - XSS Prevention: <script>alert('XSS')</script>, javascript:alert(1)
   - Command Injection: ; ls -la, | whoami, && cat /etc/passwd
   - Path Traversal: ../../etc/passwd, ..\\..\\windows\\system32
   - XXE (if XML): <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
   - LDAP Injection: *, )(cn=*), )(|(cn=*))
   - Header Injection: \\r\\n\\r\\n<script>alert(1)</script>

6. EDGE CASES & BOUNDARIES:
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

7. RATE LIMITING & PERFORMANCE:
   - Rate limit detection (429 Too Many Requests)
   - Concurrent request handling
   - Load testing pattern (if applicable)
   - Response time under load
   - Circuit breaker testing

8. DATA VALIDATION FOR JSON BODY FIELDS:
   - Required field validation (missing required fields)
   - Field type validation:
     * String field with number: {{"name": 123}}
     * Number field with string: {{"age": "twenty"}}
     * Boolean field with string: {{"active": "true"}}
     * Array field with object: {{"items": {{}}}}
     * Object field with array: {{"user": []}}
   - String field edge cases:
     * Empty string: {{"name": ""}}
     * Only spaces: {{"name": "   "}}
     * Very long string: {{"name": "(very long string - 10000 a's)"}}
     * Special chars: {{"name": "'; DROP TABLE--"}}
     * Unicode: {{"name": "José 北京 🚀"}}
   - Number field edge cases:
     * Zero: {{"count": 0}}
     * Negative: {{"count": -1}}
     * Decimal in int field: {{"count": 3.14}}
     * Very large: {{"count": 999999999999}}
     * String number: {{"count": "123"}}
   - Format validation (email, URL, UUID, date)
   - Enum/choice field validation
   - Nested object validation
   - Array length constraints

{'''
9. GRAPHQL SPECIFIC (if applicable):
   - Query depth limiting
   - Alias batching attacks
   - Introspection query security
   - Fragment spreading
   - Variable validation
   - Mutation atomicity
''' if is_graphql else ''}

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
- Example for a field "username" (string):
  * Valid: {{"username": "john_doe"}}
  * Empty: {{"username": ""}}
  * Special chars: {{"username": "john'; DROP TABLE users--"}}
  * Unicode: {{"username": "José 北京 🚀"}}
  * Too long: {{"username": "(very long string - 10000 characters)"}}
  * Wrong type: {{"username": 123}}
  * Null: {{"username": null}}
- Example for a field "age" (integer):
  * Valid: {{"age": 25}}
  * Zero: {{"age": 0}}
  * Negative: {{"age": -1}}
  * Max: {{"age": 999999999}}
  * Decimal: {{"age": 25.5}}
  * String: {{"age": "25"}}
  * Null: {{"age": null}}

IMPORTANT:
- Generate ONLY the test code, no explanations
- Do NOT wrap the code in markdown blocks (no ```)
- Do NOT include ```javascript``` or any language tags
- Start directly with the code (imports first)
- Make it immediately runnable
- Include ALL necessary imports
- Use the actual API response data for accurate assertions
- Each test category MUST have at least 2-3 test cases

Generate the COMPLETE test script now - pure code only, no markdown"""

                system_message = f"""You are an expert QA engineer specializing in comprehensive API testing, security testing, and test automation.
Your expertise includes: OWASP Top 10, authentication/authorization testing, injection attacks, performance testing, and {request.test_framework} best practices.
Generate a complete, production-ready test suite that would pass a security audit and provide maximum code coverage.
IMPORTANT: Output ONLY pure code without any markdown formatting, no ``` blocks, no language tags."""

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


# Initialize service
service = APIAutopilotService()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@api_router.post("/generate-tests", response_model=EnhancedTestGenerationResponse)
async def generate_tests_v2(request: EnhancedTestGenerationRequest):
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

        # Generate tests
        response = await service.generate_tests_v2(request)

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


@api_router.post("/execute-tests")
async def execute_tests(request: TestExecutionRequest):
    """Execute generated tests in real-time"""
    try:
        runner = TestRunner()

        # Run tests and get report
        report = await runner.run_and_report(
            test_code=request.test_code,
            framework=request.framework,
            api_base_url=request.api_base_url,
            generate_coverage=True
        )

        return report

    except Exception as e:
        logger.error(f"Error executing tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


@api_router.websocket("/ws/test-execution")
async def websocket_test_execution(websocket: WebSocket):
    """WebSocket for real-time test execution updates"""
    await websocket.accept()

    try:
        while True:
            # Receive test execution request
            data = await websocket.receive_json()

            # Execute tests and stream results
            runner = TestRunner()

            # Send progress updates
            await websocket.send_json({
                "type": "progress",
                "message": "Starting test execution..."
            })

            # Run tests
            report = await runner.run_and_report(
                test_code=data["test_code"],
                framework=data["framework"],
                api_base_url=data.get("api_base_url")
            )

            # Send results
            await websocket.send_json({
                "type": "result",
                "data": report
            })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()


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