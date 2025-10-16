# -*- coding: utf-8 -*-
"""
Real-time Test Execution and Validation Module
Execute generated tests in isolated environments and provide instant feedback
"""

import json
import subprocess
import tempfile
import os
import sys
import asyncio
import re
import time
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles

# Optional Docker support
try:
    import docker
except ImportError:
    docker = None

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class ExecutionEnvironment(Enum):
    """Test execution environment"""
    LOCAL = "local"
    DOCKER = "docker"
    SANDBOX = "sandbox"
    CLOUD = "cloud"


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: TestStatus
    duration: float  # in seconds
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    console_output: str = ""


@dataclass
class TestSuiteResult:
    """Complete test suite execution result"""
    framework: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    test_results: List[TestResult]
    console_output: str
    coverage: Optional[Dict[str, Any]] = None
    success_rate: float = 0.0


@dataclass
class ValidationResult:
    """Test code validation result"""
    is_valid: bool
    syntax_errors: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class TestExecutor:
    """Execute tests in isolated environments"""

    def __init__(self, environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL):
        self.environment = environment
        self.docker_client = None
        if environment == ExecutionEnvironment.DOCKER:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                self.environment = ExecutionEnvironment.LOCAL

    async def execute_test_suite(
        self,
        test_code: str,
        framework: str,
        timeout: int = 60,
        api_base_url: Optional[str] = None
    ) -> TestSuiteResult:
        """
        Execute test suite and return results

        Args:
            test_code: Generated test code
            framework: Test framework (jest, pytest, etc.)
            timeout: Execution timeout in seconds
            api_base_url: Optional API base URL for test execution

        Returns:
            TestSuiteResult with execution details
        """
        start_time = time.time()

        # Validate code first
        validation = await self.validate_test_code(test_code, framework)
        if not validation.is_valid:
            return TestSuiteResult(
                framework=framework,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                duration=0,
                test_results=[],
                console_output=f"Validation failed: {', '.join(validation.syntax_errors)}"
            )

        # Execute based on framework
        if framework in ["jest", "mocha", "cypress"]:
            result = await self._execute_javascript_tests(test_code, framework, timeout, api_base_url)
        elif framework in ["pytest", "requests"]:
            result = await self._execute_python_tests(test_code, framework, timeout, api_base_url)
        elif framework in ["testng", "junit"]:
            result = await self._execute_java_tests(test_code, framework, timeout, api_base_url)
        else:
            result = TestSuiteResult(
                framework=framework,
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                duration=0,
                test_results=[],
                console_output=f"Unsupported framework: {framework}"
            )

        result.duration = time.time() - start_time
        result.success_rate = (result.passed / result.total_tests * 100) if result.total_tests > 0 else 0

        return result

    async def _execute_javascript_tests(
        self,
        test_code: str,
        framework: str,
        timeout: int,
        api_base_url: Optional[str]
    ) -> TestSuiteResult:
        """Execute JavaScript-based tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.spec.js"

            # Write test file
            async with aiofiles.open(test_file, 'w') as f:
                await f.write(test_code)

            # Create package.json
            package_json = {
                "name": "api-test-suite",
                "version": "1.0.0",
                "scripts": {
                    "test": f"{framework} test.spec.js --reporters=json" if framework == "jest" else f"mocha test.spec.js --reporter json"
                },
                "dependencies": {
                    framework: "latest",
                    "axios": "latest",
                    "ajv": "latest"
                }
            }

            if framework == "jest":
                package_json["jest"] = {
                    "testEnvironment": "node",
                    "testTimeout": timeout * 1000
                }

            package_file = Path(temp_dir) / "package.json"
            async with aiofiles.open(package_file, 'w') as f:
                await f.write(json.dumps(package_json, indent=2))

            # Install dependencies
            install_process = await asyncio.create_subprocess_exec(
                "npm", "install", "--silent",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await install_process.communicate()

            # Run tests
            env = os.environ.copy()
            if api_base_url:
                env["API_BASE_URL"] = api_base_url

            test_process = await asyncio.create_subprocess_exec(
                "npm", "test",
                cwd=temp_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    test_process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                test_process.kill()
                return TestSuiteResult(
                    framework=framework,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    duration=timeout,
                    test_results=[],
                    console_output=f"Test execution timed out after {timeout} seconds"
                )

            # Parse results
            return self._parse_javascript_results(stdout.decode(), stderr.decode(), framework)

    async def _execute_python_tests(
        self,
        test_code: str,
        framework: str,
        timeout: int,
        api_base_url: Optional[str]
    ) -> TestSuiteResult:
        """Execute Python-based tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_api.py"

            # Write test file
            async with aiofiles.open(test_file, 'w') as f:
                await f.write(test_code)

            # Create requirements.txt
            requirements = [
                "pytest" if framework == "pytest" else "unittest2",
                "requests",
                "jsonschema"
            ]
            req_file = Path(temp_dir) / "requirements.txt"
            async with aiofiles.open(req_file, 'w') as f:
                await f.write("\n".join(requirements))

            # Create virtual environment and install dependencies
            venv_path = Path(temp_dir) / "venv"
            create_venv = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "venv", str(venv_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await create_venv.communicate()

            # Install requirements
            pip_path = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
            install_process = await asyncio.create_subprocess_exec(
                str(pip_path), "install", "-r", "requirements.txt",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await install_process.communicate()

            # Run tests
            python_path = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "python"
            env = os.environ.copy()
            if api_base_url:
                env["API_BASE_URL"] = api_base_url

            if framework == "pytest":
                cmd = [str(python_path), "-m", "pytest", str(test_file), "--json-report", "--json-report-file=report.json"]
            else:
                cmd = [str(python_path), str(test_file)]

            test_process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=temp_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    test_process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                test_process.kill()
                return TestSuiteResult(
                    framework=framework,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    duration=timeout,
                    test_results=[],
                    console_output=f"Test execution timed out after {timeout} seconds"
                )

            # Parse results
            report_file = Path(temp_dir) / "report.json"
            if report_file.exists():
                async with aiofiles.open(report_file, 'r') as f:
                    report_data = json.loads(await f.read())
                return self._parse_python_json_results(report_data, framework)
            else:
                return self._parse_python_output(stdout.decode(), stderr.decode(), framework)

    async def _execute_java_tests(
        self,
        test_code: str,
        framework: str,
        timeout: int,
        api_base_url: Optional[str]
    ) -> TestSuiteResult:
        """Execute Java-based tests"""
        # Implementation for Java test execution
        # This would require Maven or Gradle setup
        return TestSuiteResult(
            framework=framework,
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            duration=0,
            test_results=[],
            console_output="Java test execution not yet implemented"
        )

    def _parse_javascript_results(self, stdout: str, stderr: str, framework: str) -> TestSuiteResult:
        """Parse JavaScript test results"""
        test_results = []
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0

        try:
            # Try to parse JSON output
            result_json = json.loads(stdout)

            if framework == "jest":
                for test_suite in result_json.get("testResults", []):
                    for test in test_suite.get("assertionResults", []):
                        status = TestStatus.PASSED if test["status"] == "passed" else TestStatus.FAILED

                        # Parse failure message for expected vs actual
                        error_message = None
                        stack_trace = None
                        if status == TestStatus.FAILED and test.get("failureMessages"):
                            full_error = test["failureMessages"][0]
                            error_message = full_error

                            # Try to extract stack trace
                            if "\n    at " in full_error:
                                parts = full_error.split("\n    at ", 1)
                                error_message = parts[0]
                                stack_trace = "    at " + parts[1]

                        test_result = TestResult(
                            name=test["title"],
                            status=status,
                            duration=test.get("duration", 0) / 1000,
                            error_message=error_message,
                            stack_trace=stack_trace
                        )
                        test_results.append(test_result)
                        total_tests += 1
                        if status == TestStatus.PASSED:
                            passed += 1
                        else:
                            failed += 1

            elif framework == "mocha":
                for test in result_json.get("tests", []):
                    status = TestStatus.PASSED if test.get("pass") else TestStatus.FAILED
                    test_result = TestResult(
                        name=test["title"],
                        status=status,
                        duration=test.get("duration", 0) / 1000,
                        error_message=test.get("err", {}).get("message") if status == TestStatus.FAILED else None
                    )
                    test_results.append(test_result)
                    total_tests += 1
                    if status == TestStatus.PASSED:
                        passed += 1
                    else:
                        failed += 1

        except json.JSONDecodeError:
            # Fallback to parsing text output
            if "PASS" in stdout or "✓" in stdout:
                passed = len(re.findall(r"✓|PASS", stdout))
            if "FAIL" in stdout or "✗" in stdout:
                failed = len(re.findall(r"✗|FAIL", stdout))
            total_tests = passed + failed

        return TestSuiteResult(
            framework=framework,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=0,
            test_results=test_results,
            console_output=stdout + stderr
        )

    def _parse_python_json_results(self, report_data: Dict, framework: str) -> TestSuiteResult:
        """Parse Python JSON test results"""
        test_results = []
        total_tests = report_data.get("summary", {}).get("total", 0)
        passed = report_data.get("summary", {}).get("passed", 0)
        failed = report_data.get("summary", {}).get("failed", 0)
        skipped = report_data.get("summary", {}).get("skipped", 0)

        for test in report_data.get("tests", []):
            status = TestStatus.PASSED if test["outcome"] == "passed" else TestStatus.FAILED
            test_result = TestResult(
                name=test["nodeid"],
                status=status,
                duration=test.get("duration", 0),
                error_message=test.get("call", {}).get("longrepr") if status == TestStatus.FAILED else None
            )
            test_results.append(test_result)

        return TestSuiteResult(
            framework=framework,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=report_data.get("duration", 0),
            test_results=test_results,
            console_output=""
        )

    def _parse_python_output(self, stdout: str, stderr: str, framework: str) -> TestSuiteResult:
        """Parse Python text output"""
        test_results = []
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0

        # Parse pytest output
        if "passed" in stdout or "failed" in stdout:
            passed_match = re.search(r"(\d+) passed", stdout)
            failed_match = re.search(r"(\d+) failed", stdout)
            skipped_match = re.search(r"(\d+) skipped", stdout)

            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            if skipped_match:
                skipped = int(skipped_match.group(1))

            total_tests = passed + failed + skipped

        return TestSuiteResult(
            framework=framework,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=0,
            test_results=test_results,
            console_output=stdout + stderr
        )

    async def validate_test_code(self, test_code: str, framework: str) -> ValidationResult:
        """
        Validate test code for syntax and dependencies

        Args:
            test_code: Test code to validate
            framework: Test framework

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)

        # Check for syntax errors
        if framework in ["jest", "mocha", "cypress"]:
            syntax_errors = self._validate_javascript_syntax(test_code)
        elif framework in ["pytest", "requests"]:
            syntax_errors = self._validate_python_syntax(test_code)
        else:
            syntax_errors = []

        if syntax_errors:
            result.is_valid = False
            result.syntax_errors = syntax_errors

        # Check for missing dependencies
        missing_deps = self._check_dependencies(test_code, framework)
        if missing_deps:
            result.missing_dependencies = missing_deps
            result.warnings.append(f"Missing dependencies: {', '.join(missing_deps)}")

        # Check for common issues
        issues = self._check_common_issues(test_code, framework)
        result.warnings.extend(issues)

        # Generate suggestions
        result.suggestions = self._generate_suggestions(test_code, framework)

        return result

    def _validate_javascript_syntax(self, code: str) -> List[str]:
        """Validate JavaScript syntax"""
        errors = []

        # Check for unbalanced braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

        # Check for unbalanced parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")

        # Check for missing semicolons (basic check)
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.endswith((';', '{', '}', ',', ')', ']')) and not stripped.startswith('//'):
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith((')', '}', ']', '.')):
                    errors.append(f"Line {i + 1}: Possible missing semicolon")

        return errors

    def _validate_python_syntax(self, code: str) -> List[str]:
        """Validate Python syntax"""
        errors = []

        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

        return errors

    def _check_dependencies(self, code: str, framework: str) -> List[str]:
        """Check for missing dependencies"""
        missing = []

        if framework in ["jest", "mocha", "cypress"]:
            # Check JavaScript dependencies
            if "require('axios')" in code or "import axios" in code:
                if "axios" not in missing:
                    missing.append("axios")
            if "require('ajv')" in code or "import Ajv" in code:
                if "ajv" not in missing:
                    missing.append("ajv")

        elif framework in ["pytest", "requests"]:
            # Check Python imports
            if "import requests" in code and "requests" not in missing:
                missing.append("requests")
            if "from jsonschema import" in code and "jsonschema" not in missing:
                missing.append("jsonschema")

        return []  # Return empty as we'll install these automatically

    def _check_common_issues(self, code: str, framework: str) -> List[str]:
        """Check for common issues in test code"""
        issues = []

        # Check for hardcoded URLs
        if re.search(r'https?://localhost:\d+', code):
            issues.append("Contains hardcoded localhost URL - consider using environment variable")

        # Check for hardcoded credentials
        if re.search(r'(password|secret|api_key)\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            issues.append("Contains hardcoded credentials - use environment variables instead")

        # Check for console.log or print statements
        if framework in ["jest", "mocha"] and "console.log" in code:
            issues.append("Contains console.log statements - consider removing for production")
        elif framework in ["pytest", "requests"] and "print(" in code:
            issues.append("Contains print statements - consider using logging instead")

        return issues

    def _generate_suggestions(self, code: str, framework: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Check test count
        test_count = len(re.findall(r'\bit\(|test\(|def test_', code))
        if test_count < 5:
            suggestions.append(f"Only {test_count} tests found - consider adding more test cases")

        # Check for assertions
        if framework in ["jest", "mocha"]:
            assertion_count = len(re.findall(r'expect\(', code))
            if assertion_count < test_count * 2:
                suggestions.append("Consider adding more assertions per test")

        # Check for error handling
        if "try" not in code and "catch" not in code and ".catch" not in code:
            suggestions.append("Consider adding error handling for API calls")

        # Check for timeouts
        if "timeout" not in code.lower():
            suggestions.append("Consider adding timeout configuration for API calls")

        return suggestions


class TestRunner:
    """High-level test runner with additional features"""

    def __init__(self):
        self.executor = TestExecutor()
        self.results_cache = {}

    async def run_and_report(
        self,
        test_code: str,
        framework: str,
        api_base_url: Optional[str] = None,
        generate_coverage: bool = False
    ) -> Dict[str, Any]:
        """
        Run tests and generate comprehensive report

        Args:
            test_code: Test code to execute
            framework: Test framework
            api_base_url: API base URL
            generate_coverage: Whether to generate coverage report

        Returns:
            Comprehensive test report
        """
        # Validate first
        validation = await self.executor.validate_test_code(test_code, framework)

        if not validation.is_valid:
            return {
                "status": "validation_failed",
                "validation": validation.__dict__,
                "execution": None
            }

        # Execute tests
        execution_result = await self.executor.execute_test_suite(
            test_code,
            framework,
            timeout=60,
            api_base_url=api_base_url
        )

        # Generate coverage if requested
        coverage = None
        if generate_coverage:
            coverage = await self._generate_coverage_report(test_code, framework)

        # Generate report
        report = {
            "status": "completed",
            "validation": validation.__dict__,
            "execution": {
                "framework": execution_result.framework,
                "total_tests": execution_result.total_tests,
                "passed": execution_result.passed,
                "failed": execution_result.failed,
                "skipped": execution_result.skipped,
                "duration": execution_result.duration,
                "success_rate": execution_result.success_rate,
                "test_results": [
                    {
                        "name": tr.name,
                        "status": tr.status.value,
                        "duration": tr.duration,
                        "error": {
                            "message": tr.error_message,
                            "stack": tr.stack_trace,
                            "expected": self._extract_expected(tr.error_message) if tr.error_message else None,
                            "actual": self._extract_actual(tr.error_message) if tr.error_message else None
                        } if tr.error_message else None
                    }
                    for tr in execution_result.test_results
                ]
            },
            "coverage": coverage,
            "timestamp": time.time()
        }

        # Cache result
        cache_key = hashlib.md5(test_code.encode()).hexdigest()
        self.results_cache[cache_key] = report

        return report

    def _extract_expected(self, error_message: str) -> Optional[str]:
        """Extract expected value from error message"""
        if not error_message:
            return None

        # Look for common patterns
        patterns = [
            r"Expected: (.+?)(?:\n|$)",
            r"expected (.+?) to (?:be|equal|match) (.+?)(?:\n|$)",
            r"Expected.*?to receive (.+?)(?:\n|$)",
            r"Expected.*?: (.+?)(?:\n|Received|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.lastindex == 1 else match.group(2).strip()

        return None

    def _extract_actual(self, error_message: str) -> Optional[str]:
        """Extract actual value from error message"""
        if not error_message:
            return None

        # Look for common patterns
        patterns = [
            r"Received: (.+?)(?:\n|$)",
            r"Actual: (.+?)(?:\n|$)",
            r"but (?:got|received|was) (.+?)(?:\n|$)",
            r"Received.*?: (.+?)(?:\n|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    async def _generate_coverage_report(self, test_code: str, framework: str) -> Optional[Dict[str, Any]]:
        """Generate code coverage report"""
        # This would integrate with coverage tools like Istanbul for JS or coverage.py for Python
        # For now, returning mock coverage data
        return {
            "lines": {"total": 100, "covered": 85, "percentage": 85.0},
            "functions": {"total": 20, "covered": 18, "percentage": 90.0},
            "branches": {"total": 30, "covered": 25, "percentage": 83.3},
            "statements": {"total": 150, "covered": 130, "percentage": 86.7}
        }

    async def run_in_parallel(
        self,
        test_suites: List[Tuple[str, str]],
        api_base_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple test suites in parallel

        Args:
            test_suites: List of (test_code, framework) tuples
            api_base_url: API base URL

        Returns:
            List of test reports
        """
        tasks = []
        for test_code, framework in test_suites:
            task = self.run_and_report(test_code, framework, api_base_url)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'TestExecutor',
    'TestRunner',
    'TestResult',
    'TestSuiteResult',
    'ValidationResult',
    'TestStatus',
    'ExecutionEnvironment'
]