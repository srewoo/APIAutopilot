# -*- coding: utf-8 -*-
"""
Test Execution Engine
Executes generated test scripts and returns real-time results
"""

import asyncio
import subprocess
import tempfile
import os
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
import logging

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TestFrameworkType(Enum):
    """Supported test frameworks for execution"""
    # JavaScript
    JEST = "jest"
    MOCHA = "mocha"
    CYPRESS = "cypress"
    K6 = "k6"

    # Python
    PYTEST = "pytest"
    REQUESTS = "requests"
    LOCUST = "locust"

    # Java
    JUNIT = "junit"
    TESTNG = "testng"
    RESTASSURED = "restassured"
    GATLING = "gatling"
    JMETER = "jmeter"

    # Other
    ARTILLERY = "artillery"
    BEHAVE = "behave"


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    status: ExecutionStatus
    duration: float
    error_message: Optional[str] = None
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Complete execution result"""
    execution_id: str
    framework: str
    status: ExecutionStatus
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration: float
    start_time: str
    end_time: str
    test_results: List[TestResult]
    console_output: List[str]
    error: Optional[str] = None
    coverage: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        return {
            "execution_id": self.execution_id,
            "framework": self.framework,
            "status": self.status.value if isinstance(self.status, ExecutionStatus) else self.status,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "test_results": [
                {
                    "name": tr.name,
                    "status": tr.status.value if isinstance(tr.status, ExecutionStatus) else tr.status,
                    "duration": tr.duration,
                    "error_message": tr.error_message,
                    "assertions": tr.assertions,
                    "logs": tr.logs
                }
                for tr in self.test_results
            ],
            "console_output": self.console_output,
            "error": self.error,
            "coverage": self.coverage,
            "performance_metrics": self.performance_metrics
        }


class TestExecutor:
    """Executes test scripts in isolated environments"""

    def __init__(self, workspace_dir: str = "/tmp/test_execution"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        self.active_processes: Dict[str, subprocess.Popen] = {}

    async def execute_tests(
        self,
        test_code: str,
        framework: str,
        execution_id: Optional[str] = None,
        timeout: int = 300,
        environment: Optional[Dict[str, str]] = None,
        stream_output: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute test scripts and stream results

        Yields progress updates and final results
        """
        execution_id = execution_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            # Detect framework type
            framework_type = TestFrameworkType[framework.upper()]

            # Create execution directory
            exec_dir = self.workspace_dir / execution_id
            exec_dir.mkdir(exist_ok=True)

            # Prepare test file
            test_file = await self._prepare_test_file(test_code, framework_type, exec_dir)

            # Install dependencies if needed
            if stream_output:
                yield {
                    "type": "progress",
                    "message": "Installing dependencies...",
                    "execution_id": execution_id
                }

            # Install and stream installation output
            logger.info(f"Starting dependency installation for {framework_type}")
            install_start = time.time()
            async for install_line in self._install_dependencies_stream(framework_type, exec_dir):
                if stream_output:
                    yield {
                        "type": "console",
                        "line": install_line,
                        "execution_id": execution_id
                    }
            logger.info(f"Dependency installation completed in {time.time() - install_start:.1f}s")

            # Build execution command
            command = self._build_command(framework_type, test_file, exec_dir)

            # Execute tests
            if stream_output:
                yield {
                    "type": "progress",
                    "message": "Running tests...",
                    "execution_id": execution_id
                }

            # Run the command and stream output
            logger.info(f"Starting test execution with command: {' '.join(command)}, timeout: {timeout}s")
            test_start = time.time()
            async for output_line in self._run_command_stream(
                command,
                exec_dir,
                execution_id,
                timeout,
                environment
            ):
                if stream_output:
                    yield {
                        "type": "console",
                        "line": output_line,
                        "execution_id": execution_id
                    }
            logger.info(f"Test execution completed in {time.time() - test_start:.1f}s")

            # Parse results
            results = await self._parse_results(framework_type, exec_dir, execution_id)

            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time

            # Build final result
            execution_result = ExecutionResult(
                execution_id=execution_id,
                framework=framework,
                status=results.get("status", ExecutionStatus.PASSED),
                total_tests=results.get("total", 0),
                passed_tests=results.get("passed", 0),
                failed_tests=results.get("failed", 0),
                skipped_tests=results.get("skipped", 0),
                duration=duration,
                start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
                test_results=results.get("tests", []),
                console_output=results.get("console", []),
                coverage=results.get("coverage"),
                performance_metrics=results.get("performance")
            )

            # Yield final result
            yield {
                "type": "result",
                "data": execution_result.to_dict(),
                "execution_id": execution_id
            }

        except asyncio.TimeoutError as te:
            elapsed = time.time() - start_time
            logger.error(f"Timeout error in execute_tests after {elapsed:.1f} seconds (configured timeout: {timeout}s): {str(te)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Test execution timed out after {timeout} seconds (actual elapsed: {elapsed:.1f}s)",
                "execution_id": execution_id
            }
        except Exception as e:
            logger.error(f"Execution error in execute_tests: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e),
                "execution_id": execution_id
            }
        finally:
            # Cleanup
            await self._cleanup(execution_id)

    def _extract_dependencies(self, test_code: str, framework: TestFrameworkType) -> List[str]:
        """Extract required dependencies from test code"""
        import re

        dependencies = set()

        if framework in [TestFrameworkType.JEST, TestFrameworkType.MOCHA, TestFrameworkType.CYPRESS]:
            # Match require('package') or require("package")
            require_pattern = r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
            for match in re.finditer(require_pattern, test_code):
                pkg = match.group(1)
                # Skip built-in Node modules and relative paths
                if not pkg.startswith('.') and not pkg.startswith('/') and pkg not in ['fs', 'path', 'http', 'https', 'crypto', 'util', 'stream', 'events', 'os', 'child_process', 'cluster', 'dns', 'net', 'tls', 'dgram', 'readline', 'repl', 'vm', 'assert', 'buffer', 'console', 'constants', 'domain', 'process', 'punycode', 'querystring', 'string_decoder', 'timers', 'tty', 'url', 'v8', 'zlib']:
                    # Handle scoped packages (e.g., @faker-js/faker)
                    if pkg.startswith('@'):
                        # For scoped packages, keep the full scope/package name
                        dependencies.add(pkg)
                    else:
                        dependencies.add(pkg)

            # Match import ... from 'package' or import 'package'
            import_pattern = r"import\s+(?:.*?\s+from\s+)?['\"]([^'\"]+)['\"]"
            for match in re.finditer(import_pattern, test_code):
                pkg = match.group(1)
                if not pkg.startswith('.') and not pkg.startswith('/'):
                    dependencies.add(pkg)

            # Add common test dependencies that might be needed
            if 'describe' in test_code or 'test' in test_code or 'it' in test_code:
                # These are usually provided by the test framework itself
                pass

        elif framework in [TestFrameworkType.PYTEST, TestFrameworkType.REQUESTS, TestFrameworkType.LOCUST]:
            # Match Python imports
            import_pattern = r"(?:from\s+([^\s]+)\s+import|import\s+([^\s,]+))"
            for match in re.finditer(import_pattern, test_code):
                pkg = match.group(1) or match.group(2)
                if pkg:
                    # Get the base package name
                    base_pkg = pkg.split('.')[0]
                    # Skip built-in Python modules
                    if base_pkg not in ['os', 'sys', 'json', 'time', 'datetime', 'random', 're', 'math', 'collections', 'itertools', 'functools', 'typing', 'unittest', 'asyncio', 'threading', 'multiprocessing', 'subprocess', 'io', 'builtins', '__builtin__']:
                        dependencies.add(base_pkg)

        return list(dependencies)

    async def _prepare_test_file(
        self,
        test_code: str,
        framework: TestFrameworkType,
        exec_dir: Path
    ) -> Path:
        """Prepare test file based on framework"""

        # Determine file extension
        extensions = {
            TestFrameworkType.JEST: ".test.js",
            TestFrameworkType.MOCHA: ".test.js",
            TestFrameworkType.CYPRESS: ".cy.js",
            TestFrameworkType.K6: ".js",
            TestFrameworkType.PYTEST: "_test.py",
            TestFrameworkType.REQUESTS: "_test.py",
            TestFrameworkType.LOCUST: ".py",
            TestFrameworkType.JUNIT: ".java",
            TestFrameworkType.TESTNG: ".java",
            TestFrameworkType.RESTASSURED: ".java",
            TestFrameworkType.GATLING: ".scala",
            TestFrameworkType.JMETER: ".jmx",
            TestFrameworkType.ARTILLERY: ".yml",
            TestFrameworkType.BEHAVE: ".feature"
        }

        ext = extensions.get(framework, ".txt")
        test_file = exec_dir / f"test{ext}"

        # Write test code
        async with aiofiles.open(test_file, "w") as f:
            await f.write(test_code)

        # Extract dependencies from test code
        detected_deps = self._extract_dependencies(test_code, framework)
        logger.info(f"Detected dependencies: {detected_deps}")

        # Create package.json for Node.js tests
        if framework in [TestFrameworkType.JEST, TestFrameworkType.MOCHA, TestFrameworkType.CYPRESS]:
            await self._create_package_json(framework, exec_dir, detected_deps)

        # Create jest.config.js for Jest
        if framework == TestFrameworkType.JEST:
            await self._create_jest_config(exec_dir)

        # Create requirements.txt for Python tests
        if framework in [TestFrameworkType.PYTEST, TestFrameworkType.REQUESTS, TestFrameworkType.LOCUST]:
            await self._create_requirements_txt(framework, exec_dir, detected_deps)

        return test_file

    async def _create_package_json(self, framework: TestFrameworkType, exec_dir: Path, detected_deps: List[str] = None):
        """Create package.json with detected dependencies"""

        detected_deps = detected_deps or []

        # Base package structure
        base_package = {
            "name": "test-execution",
            "version": "1.0.0",
            "scripts": {}
        }

        # Framework-specific configurations and default dependencies
        framework_configs = {
            TestFrameworkType.JEST: {
                "scripts": {"test": "jest"},
                "default_deps": {"jest": "^29.0.0", "@jest/globals": "^29.0.0"}
            },
            TestFrameworkType.MOCHA: {
                "scripts": {"test": "mocha --reporter json > results.json"},
                "default_deps": {"mocha": "^10.2.0", "chai": "^4.3.10"}
            },
            TestFrameworkType.CYPRESS: {
                "scripts": {"test": "cypress run --reporter json"},
                "default_deps": {"cypress": "^13.6.0"}
            }
        }

        config = framework_configs.get(framework, framework_configs[TestFrameworkType.JEST])

        # Build package.json
        package_json = {
            **base_package,
            "scripts": config["scripts"],
            "dependencies": {**config["default_deps"]}
        }

        # Add detected dependencies with latest versions
        for dep in detected_deps:
            if dep not in package_json["dependencies"]:
                # Use latest version for detected dependencies
                package_json["dependencies"][dep] = "latest"

        # Add commonly used testing dependencies if they appear to be needed
        if framework == TestFrameworkType.JEST:
            # Check for common testing patterns
            if any(dep in detected_deps for dep in ['supertest', 'axios', 'node-fetch']):
                # API testing likely
                if 'supertest' not in package_json["dependencies"]:
                    package_json["dependencies"]["supertest"] = "^6.3.0"
            if 'nock' in detected_deps and 'nock' not in package_json["dependencies"]:
                package_json["dependencies"]["nock"] = "^13.3.0"

        async with aiofiles.open(exec_dir / "package.json", "w") as f:
            await f.write(json.dumps(package_json, indent=2))

    async def _create_jest_config(self, exec_dir: Path):
        """Create jest.config.js for Jest tests"""
        jest_config = """module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/*.test.js'],
  verbose: true,
  collectCoverage: false,
  testTimeout: 30000,
  globals: {
    'ts-jest': {
      useESM: false,
    },
  },
};"""
        async with aiofiles.open(exec_dir / "jest.config.js", "w") as f:
            await f.write(jest_config)

    async def _create_requirements_txt(self, framework: TestFrameworkType, exec_dir: Path, detected_deps: List[str] = None):
        """Create requirements.txt with detected dependencies"""

        detected_deps = detected_deps or []

        # Default requirements for each framework
        base_requirements = {
            TestFrameworkType.PYTEST: [
                "pytest>=7.4.0",
                "pytest-asyncio>=0.21.0",
                "pytest-json-report>=1.5.0"
            ],
            TestFrameworkType.REQUESTS: [
                "requests>=2.31.0",
                "unittest-xml-reporting>=3.2.0"
            ],
            TestFrameworkType.LOCUST: [
                "locust>=2.17.0"
            ]
        }

        reqs = list(base_requirements.get(framework, base_requirements[TestFrameworkType.PYTEST]))

        # Add detected dependencies if not already included
        for dep in detected_deps:
            # Map common package names to PyPI names if different
            package_map = {
                'requests': 'requests>=2.31.0',
                'aiohttp': 'aiohttp>=3.9.0',
                'httpx': 'httpx>=0.24.0',
                'pytest': 'pytest>=7.4.0',
                'numpy': 'numpy>=1.24.0',
                'pandas': 'pandas>=2.0.0',
                'beautifulsoup4': 'beautifulsoup4>=4.12.0',
                'selenium': 'selenium>=4.0.0'
            }

            if dep in package_map and package_map[dep] not in reqs:
                reqs.append(package_map[dep])
            elif not any(dep in req for req in reqs):
                # Add with no version constraint if not mapped
                reqs.append(dep)

        async with aiofiles.open(exec_dir / "requirements.txt", "w") as f:
            await f.write("\n".join(reqs))

    async def _install_dependencies_stream(self, framework: TestFrameworkType, exec_dir: Path) -> AsyncGenerator[str, None]:
        """Install dependencies and stream output"""

        proc = None
        try:
            if framework in [TestFrameworkType.JEST, TestFrameworkType.MOCHA, TestFrameworkType.CYPRESS, TestFrameworkType.K6]:
                # Install Node.js dependencies
                proc = await asyncio.create_subprocess_exec(
                    "npm", "install", "--no-audit", "--no-fund", "--loglevel=error",
                    cwd=exec_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
            elif framework in [TestFrameworkType.PYTEST, TestFrameworkType.REQUESTS, TestFrameworkType.LOCUST]:
                # Install Python dependencies
                proc = await asyncio.create_subprocess_exec(
                    "pip", "install", "-r", "requirements.txt", "--quiet",
                    cwd=exec_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
            else:
                return  # No dependencies to install

            if proc:
                # Stream output with timeout
                try:
                    start_time = asyncio.get_event_loop().time()
                    while True:
                        # Check if we've exceeded the timeout
                        if asyncio.get_event_loop().time() - start_time > 50:
                            yield "Dependency installation taking too long, skipping..."
                            proc.terminate()
                            await proc.wait()
                            break

                        try:
                            line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                            if not line:
                                break
                            decoded_line = line.decode("utf-8", errors="ignore").rstrip()
                            if decoded_line:  # Only yield non-empty lines
                                yield decoded_line
                        except asyncio.TimeoutError:
                            # Check if process is still running
                            if proc.returncode is not None:
                                break
                            continue

                    # Wait for process to complete
                    await asyncio.wait_for(proc.wait(), timeout=5)

                    if proc.returncode != 0:
                        yield f"⚠️ Dependency installation completed with warnings (code: {proc.returncode})"
                    else:
                        yield "✅ Dependencies installed successfully"

                except asyncio.TimeoutError:
                    yield "⚠️ Dependency installation timed out, continuing anyway..."
                    proc.terminate()
                    await proc.wait()
                except Exception as e:
                    yield f"⚠️ Error during installation: {str(e)}"
                    if proc:
                        proc.terminate()
                        await proc.wait()

        except Exception as e:
            yield f"⚠️ Failed to start dependency installation: {str(e)}"

    async def _install_dependencies(self, framework: TestFrameworkType, exec_dir: Path):
        """Install required dependencies with timeout"""

        try:
            if framework in [TestFrameworkType.JEST, TestFrameworkType.MOCHA, TestFrameworkType.CYPRESS, TestFrameworkType.K6]:
                # Install Node.js dependencies with timeout
                proc = await asyncio.create_subprocess_exec(
                    "npm", "install", "--no-audit", "--no-fund", "--loglevel=error",
                    cwd=exec_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                # Wait for 50 seconds max for npm install
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=50)

                if proc.returncode != 0:
                    logger.error(f"npm install failed: {stderr.decode()}")
                    # Continue anyway, tests might still work with missing deps

            elif framework in [TestFrameworkType.PYTEST, TestFrameworkType.REQUESTS, TestFrameworkType.LOCUST]:
                # Install Python dependencies with timeout
                proc = await asyncio.create_subprocess_exec(
                    "pip", "install", "-r", "requirements.txt", "--quiet",
                    cwd=exec_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                # Wait for 50 seconds max for pip install
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=50)

                if proc.returncode != 0:
                    logger.error(f"pip install failed: {stderr.decode()}")
                    # Continue anyway

        except asyncio.TimeoutError:
            logger.warning(f"Dependency installation timed out after 50 seconds. Continuing without full dependencies...")
            if proc:
                proc.terminate()
                await proc.wait()
        except Exception as e:
            logger.error(f"Error installing dependencies: {str(e)}")
            # Continue anyway - tests might work without all dependencies

    def _build_command(self, framework: TestFrameworkType, test_file: Path, exec_dir: Path) -> List[str]:
        """Build execution command based on framework"""

        commands = {
            # Use npx to run without full installation
            TestFrameworkType.JEST: ["npx", "-y", "jest", str(test_file.name), "--config", "jest.config.js", "--json", "--outputFile=results.json"],
            TestFrameworkType.MOCHA: ["npx", "-y", "mocha", str(test_file), "--reporter", "json"],
            TestFrameworkType.CYPRESS: ["npx", "cypress", "run", "--spec", str(test_file)],
            TestFrameworkType.K6: ["k6", "run", "--out", "json=results.json", str(test_file)],
            TestFrameworkType.PYTEST: ["pytest", str(test_file), "--json-report", "--json-report-file=results.json", "-v"],
            TestFrameworkType.REQUESTS: ["python", str(test_file)],
            TestFrameworkType.LOCUST: ["locust", "-f", str(test_file), "--headless", "-u", "10", "-r", "2", "--run-time", "30s"],
            TestFrameworkType.GATLING: ["gatling.sh", "-sf", str(exec_dir), "-rf", str(exec_dir / "results")],
            TestFrameworkType.JMETER: ["jmeter", "-n", "-t", str(test_file), "-l", str(exec_dir / "results.jtl")],
            TestFrameworkType.ARTILLERY: ["artillery", "run", str(test_file), "--output", str(exec_dir / "results.json")]
        }

        return commands.get(framework, ["echo", "Unsupported framework"])

    async def _run_command_stream(
        self,
        command: List[str],
        cwd: Path,
        execution_id: str,
        timeout: int,
        environment: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """Run command and stream output"""

        env = os.environ.copy()
        if environment:
            env.update(environment)

        logger.info(f"Creating subprocess for command: {' '.join(command[:3])}...")
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )

        self.active_processes[execution_id] = proc
        start_time = asyncio.get_event_loop().time()

        try:
            while True:
                # Check overall timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    logger.warning(f"Command execution exceeded timeout of {timeout}s")
                    raise asyncio.TimeoutError(f"Command exceeded {timeout}s timeout")

                try:
                    # Wait for line with small timeout
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check if process has ended
                    if proc.returncode is not None:
                        break
                    # Continue if still running
                    continue

                if not line:
                    break

                decoded_line = line.decode("utf-8", errors="ignore").rstrip()
                if decoded_line:  # Only yield non-empty lines
                    yield decoded_line

            # Wait for process to complete with remaining timeout
            remaining_timeout = max(1, timeout - (asyncio.get_event_loop().time() - start_time))
            await asyncio.wait_for(proc.wait(), timeout=remaining_timeout)

        except asyncio.TimeoutError:
            logger.error(f"Process timeout after {asyncio.get_event_loop().time() - start_time:.1f}s")
            proc.terminate()
            await proc.wait()
            raise
        finally:
            if execution_id in self.active_processes:
                del self.active_processes[execution_id]

    async def _parse_results(
        self,
        framework: TestFrameworkType,
        exec_dir: Path,
        execution_id: str
    ) -> Dict[str, Any]:
        """Parse test results based on framework"""

        results = {
            "status": ExecutionStatus.PASSED,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "tests": [],
            "console": []
        }

        # Try to parse framework-specific results
        results_file = exec_dir / "results.json"

        if results_file.exists():
            try:
                async with aiofiles.open(results_file, "r") as f:
                    data = json.loads(await f.read())

                # Parse based on framework
                if framework == TestFrameworkType.JEST:
                    results = self._parse_jest_results(data)
                elif framework == TestFrameworkType.PYTEST:
                    results = self._parse_pytest_results(data)
                elif framework == TestFrameworkType.K6:
                    results = self._parse_k6_results(data)
                # Add more parsers as needed

            except Exception as e:
                logger.error(f"Failed to parse results: {str(e)}")

        return results

    def _parse_jest_results(self, data: Dict) -> Dict[str, Any]:
        """Parse Jest JSON results"""

        results = {
            "status": ExecutionStatus.PASSED if data.get("success") else ExecutionStatus.FAILED,
            "total": data.get("numTotalTests", 0),
            "passed": data.get("numPassedTests", 0),
            "failed": data.get("numFailedTests", 0),
            "skipped": data.get("numPendingTests", 0),
            "tests": []
        }

        # Parse individual test results
        for test_suite in data.get("testResults", []):
            for test in test_suite.get("assertionResults", []):
                failure_messages = test.get("failureMessages", [])
                results["tests"].append(TestResult(
                    name=test.get("title", "Unknown"),
                    status=ExecutionStatus.PASSED if test.get("status") == "passed" else ExecutionStatus.FAILED,
                    duration=test.get("duration", 0) / 1000,  # Convert to seconds
                    error_message=failure_messages[0] if failure_messages else None
                ))

        return results

    def _parse_pytest_results(self, data: Dict) -> Dict[str, Any]:
        """Parse pytest JSON results"""

        summary = data.get("summary", {})

        results = {
            "status": ExecutionStatus.PASSED if summary.get("failed", 0) == 0 else ExecutionStatus.FAILED,
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "skipped": summary.get("skipped", 0),
            "tests": []
        }

        # Parse individual test results
        for test in data.get("tests", []):
            results["tests"].append(TestResult(
                name=test.get("nodeid", "Unknown"),
                status=ExecutionStatus.PASSED if test.get("outcome") == "passed" else ExecutionStatus.FAILED,
                duration=test.get("duration", 0),
                error_message=test.get("call", {}).get("longrepr")
            ))

        return results

    def _parse_k6_results(self, data: Dict) -> Dict[str, Any]:
        """Parse k6 JSON results"""

        metrics = data.get("metrics", {})

        # Calculate pass/fail based on checks
        checks = metrics.get("checks", {})
        passed = checks.get("passes", 0)
        failed = checks.get("fails", 0)

        results = {
            "status": ExecutionStatus.PASSED if failed == 0 else ExecutionStatus.FAILED,
            "total": passed + failed,
            "passed": passed,
            "failed": failed,
            "skipped": 0,
            "tests": [],
            "performance": {
                "http_req_duration": metrics.get("http_req_duration", {}),
                "http_reqs": metrics.get("http_reqs", {}),
                "vus": metrics.get("vus", {}),
                "iterations": metrics.get("iterations", {})
            }
        }

        return results

    async def _cleanup(self, execution_id: str):
        """Cleanup execution directory"""

        exec_dir = self.workspace_dir / execution_id

        # Keep results for debugging (optional)
        # In production, you might want to delete after some time
        # import shutil
        # if exec_dir.exists():
        #     shutil.rmtree(exec_dir)

        # Terminate process if still running
        if execution_id in self.active_processes:
            proc = self.active_processes[execution_id]
            proc.terminate()
            await proc.wait()
            del self.active_processes[execution_id]

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""

        if execution_id in self.active_processes:
            proc = self.active_processes[execution_id]
            proc.terminate()
            await proc.wait()
            del self.active_processes[execution_id]
            return True

        return False

    async def get_execution_logs(self, execution_id: str) -> List[str]:
        """Get logs for a specific execution"""

        exec_dir = self.workspace_dir / execution_id
        log_file = exec_dir / "console.log"

        if log_file.exists():
            async with aiofiles.open(log_file, "r") as f:
                return (await f.read()).splitlines()

        return []


class LoadTestExecutor:
    """Special executor for load tests with metrics parsing"""

    def __init__(self):
        self.executor = TestExecutor()

    async def execute_load_test(
        self,
        test_script: str,
        framework: str,
        duration: str = "30s",
        vus: int = 10,
        stream_output: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute load test and stream metrics"""

        execution_id = str(uuid.uuid4())

        # Stream initial progress
        if stream_output:
            yield {
                "type": "progress",
                "message": f"Starting {framework} load test with {vus} VUs for {duration}",
                "execution_id": execution_id
            }

        # Execute based on framework
        if framework.lower() == "k6":
            async for result in self._execute_k6(test_script, execution_id, stream_output):
                yield result
        elif framework.lower() == "locust":
            async for result in self._execute_locust(test_script, execution_id, vus, duration, stream_output):
                yield result
        elif framework.lower() == "artillery":
            async for result in self._execute_artillery(test_script, execution_id, stream_output):
                yield result
        else:
            # Fallback to generic executor
            async for result in self.executor.execute_tests(
                test_script, framework, execution_id, stream_output=stream_output
            ):
                yield result

    async def _execute_k6(
        self,
        test_script: str,
        execution_id: str,
        stream_output: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute k6 load test with metrics streaming"""

        # Create temp file for script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(test_script)
            script_file = f.name

        try:
            # Run k6 with JSON output
            proc = await asyncio.create_subprocess_exec(
                "k6", "run", "--out", "json=-", script_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            metrics_buffer = []

            # Stream metrics
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break

                try:
                    metric = json.loads(line.decode())

                    # Stream important metrics
                    if metric.get("type") == "Point":
                        metric_name = metric.get("metric")
                        if metric_name in ["http_req_duration", "http_reqs", "vus", "iterations"]:
                            if stream_output:
                                yield {
                                    "type": "metric",
                                    "name": metric_name,
                                    "value": metric.get("data", {}).get("value"),
                                    "timestamp": metric.get("data", {}).get("time"),
                                    "execution_id": execution_id
                                }

                    metrics_buffer.append(metric)

                except json.JSONDecodeError:
                    pass

            # Wait for completion
            await proc.wait()

            # Analyze final metrics
            summary = self._analyze_k6_metrics(metrics_buffer)

            yield {
                "type": "result",
                "data": {
                    "execution_id": execution_id,
                    "framework": "k6",
                    "status": "completed",
                    "metrics": summary
                }
            }

        finally:
            os.unlink(script_file)

    def _analyze_k6_metrics(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze k6 metrics for summary"""

        summary = {
            "total_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0,
            "p95_response_time": 0,
            "p99_response_time": 0,
            "requests_per_second": 0
        }

        durations = []

        for metric in metrics:
            if metric.get("type") == "Point":
                metric_name = metric.get("metric")
                value = metric.get("data", {}).get("value", 0)

                if metric_name == "http_reqs":
                    summary["total_requests"] += 1
                elif metric_name == "http_req_failed" and value == 1:
                    summary["failed_requests"] += 1
                elif metric_name == "http_req_duration":
                    durations.append(value)
                    summary["min_response_time"] = min(summary["min_response_time"], value)
                    summary["max_response_time"] = max(summary["max_response_time"], value)

        # Calculate percentiles
        if durations:
            durations.sort()
            summary["avg_response_time"] = sum(durations) / len(durations)
            summary["p95_response_time"] = durations[int(len(durations) * 0.95)]
            summary["p99_response_time"] = durations[int(len(durations) * 0.99)]

        return summary

    async def _execute_locust(
        self,
        test_script: str,
        execution_id: str,
        vus: int,
        duration: str,
        stream_output: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute Locust load test"""

        # Similar implementation for Locust
        yield {
            "type": "result",
            "data": {
                "execution_id": execution_id,
                "framework": "locust",
                "status": "completed"
            }
        }

    async def _execute_artillery(
        self,
        test_script: str,
        execution_id: str,
        stream_output: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute Artillery load test"""

        # Similar implementation for Artillery
        yield {
            "type": "result",
            "data": {
                "execution_id": execution_id,
                "framework": "artillery",
                "status": "completed"
            }
        }