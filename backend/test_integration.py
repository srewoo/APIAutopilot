#!/usr/bin/env python3
"""
Integration test script to verify all components are working correctly
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    errors = []

    try:
        # Test core modules
        from test_generator_v2 import IntelligentTestGenerator, TestProfile, TestFramework
        print("✓ test_generator_v2 imports successfully")
    except ImportError as e:
        errors.append(f"✗ test_generator_v2: {e}")

    try:
        from api_discovery import APIDiscovery, APISpecDetector
        print("✓ api_discovery imports successfully")
    except ImportError as e:
        errors.append(f"✗ api_discovery: {e}")

    try:
        from test_executor import TestExecutor, TestRunner
        print("✓ test_executor imports successfully")
    except ImportError as e:
        errors.append(f"✗ test_executor: {e}")

    try:
        from security_scanner import SecurityScanner, QualityAnalyzer, ComplianceStandard
        print("✓ security_scanner imports successfully")
    except ImportError as e:
        errors.append(f"✗ security_scanner: {e}")

    try:
        from schema_analyzer import SchemaAnalyzer, TestGenerator as SchemaTestGenerator
        print("✓ schema_analyzer imports successfully")
    except ImportError as e:
        errors.append(f"✗ schema_analyzer: {e}")

    try:
        from load_test_generator import LoadTestGenerator, LoadTestConfig, LoadTestFramework
        print("✓ load_test_generator imports successfully")
    except ImportError as e:
        errors.append(f"✗ load_test_generator: {e}")

    try:
        from security_testing import DASTScanner, OAuthJWTTester, APIFuzzer
        print("✓ security_testing imports successfully")
    except ImportError as e:
        errors.append(f"✗ security_testing: {e}")

    try:
        from test_execution_engine import TestExecutor as NewTestExecutor, LoadTestExecutor, ExecutionStatus
        print("✓ test_execution_engine imports successfully")
    except ImportError as e:
        errors.append(f"✗ test_execution_engine: {e}")

    return errors

def test_server_endpoints():
    """Test if server file defines all expected endpoints"""
    server_file = Path("server_v2.py")

    if not server_file.exists():
        return ["Server file not found"]

    content = server_file.read_text()

    required_endpoints = [
        "/generate-tests",
        "/generate-load-tests",
        "/dast-scan",
        "/oauth-test",
        "/api-fuzz",
        "/execute-tests",
        "/execute-load-tests",
        "/ws/execute-tests",
        "/generation-status/",
        "/execution-logs/",
        "/cancel-execution/"
    ]

    errors = []
    for endpoint in required_endpoints:
        if endpoint not in content:
            errors.append(f"✗ Missing endpoint: {endpoint}")
        else:
            print(f"✓ Endpoint found: {endpoint}")

    return errors

def test_models():
    """Test if all request/response models are defined"""
    try:
        from server_v2 import (
            EnhancedTestGenerationRequest,
            EnhancedTestGenerationResponse,
            LoadTestGenerationRequest,
            DASTScanRequest,
            OAuthTestRequest,
            APIFuzzRequest,
            TestExecutionRequest,
            LoadTestExecutionRequest
        )
        print("✓ All request/response models import successfully")

        # Check if generation_id field exists
        response = EnhancedTestGenerationResponse(
            test_script="test",
            framework="jest",
            success=True,
            generation_id="test-id"
        )
        if response.generation_id == "test-id":
            print("✓ generation_id field works correctly")
        else:
            return ["generation_id field not working"]

    except Exception as e:
        return [f"✗ Model error: {e}"]

    return []

def test_service_initialization():
    """Test if APIAutopilotService initializes correctly"""
    try:
        from server_v2 import APIAutopilotService
        service = APIAutopilotService()

        # Check all components are initialized
        components = [
            'test_generator',
            'api_discovery',
            'spec_detector',
            'test_runner',
            'security_scanner',
            'quality_analyzer',
            'schema_analyzer',
            'schema_test_generator',
            'load_test_generator',
            'dast_scanner',
            'oauth_tester',
            'api_fuzzer',
            'test_executor',
            'load_test_executor'
        ]

        errors = []
        for component in components:
            if not hasattr(service, component):
                errors.append(f"✗ Missing component: {component}")
            else:
                print(f"✓ Component initialized: {component}")

        return errors

    except Exception as e:
        return [f"✗ Service initialization error: {e}"]

def check_frontend_files():
    """Check if all frontend files exist"""
    frontend_files = [
        "../frontend/src/App.js",
        "../frontend/src/AppEnhanced.js",
        "../frontend/src/AppWithExecution.js",
        "../frontend/src/AppUltimate.js",
        "../frontend/src/ScriptEditor.js",
        "../frontend/src/TestExecutionPanel.js",
        "../frontend/package.json"
    ]

    errors = []
    for file_path in frontend_files:
        path = Path(file_path)
        if not path.exists():
            errors.append(f"✗ Missing frontend file: {file_path}")
        else:
            print(f"✓ Frontend file exists: {path.name}")

    # Check if Monaco Editor is in package.json
    package_json = Path("../frontend/package.json")
    if package_json.exists():
        content = package_json.read_text()
        if "@monaco-editor/react" in content:
            print("✓ Monaco Editor is in package.json")
        else:
            errors.append("✗ Monaco Editor not found in package.json")

    return errors

def main():
    print("=" * 60)
    print("API AUTOPILOT INTEGRATION TEST")
    print("=" * 60)

    all_errors = []

    print("\n1. Testing Python Module Imports...")
    print("-" * 40)
    errors = test_imports()
    all_errors.extend(errors)

    print("\n2. Testing Server Endpoints...")
    print("-" * 40)
    errors = test_server_endpoints()
    all_errors.extend(errors)

    print("\n3. Testing Data Models...")
    print("-" * 40)
    errors = test_models()
    all_errors.extend(errors)

    print("\n4. Testing Service Initialization...")
    print("-" * 40)
    errors = test_service_initialization()
    all_errors.extend(errors)

    print("\n5. Checking Frontend Files...")
    print("-" * 40)
    errors = check_frontend_files()
    all_errors.extend(errors)

    print("\n" + "=" * 60)
    if all_errors:
        print(f"❌ TESTS FAILED - {len(all_errors)} issues found:")
        print("-" * 40)
        for error in all_errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("-" * 40)
        print("The API Autopilot application is ready to use!")
        print("\nTo start the application:")
        print("  Backend:  cd backend && python3 server_v2.py")
        print("  Frontend: cd frontend && npm install && npm start")

    print("=" * 60)

if __name__ == "__main__":
    main()