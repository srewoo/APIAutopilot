import requests
import sys
import json
from datetime import datetime

class APIAutopilotTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        if headers is None:
            headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:300]}")
                self.failed_tests.append({
                    "test": name,
                    "expected": expected_status,
                    "actual": response.status_code,
                    "response": response.text[:300]
                })

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append({
                "test": name,
                "error": str(e)
            })
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )

    def test_generate_tests_missing_data(self):
        """Test generate-tests endpoint with missing data"""
        return self.run_test(
            "Generate Tests - Missing Input Data",
            "POST",
            "generate-tests",
            422,  # FastAPI validation error
            data={
                "input_type": "curl",
                "ai_provider": "openai",
                "ai_api_key": "test-key",
                "test_framework": "jest"
                # Missing input_data
            }
        )

    def test_generate_tests_invalid_input_type(self):
        """Test generate-tests endpoint with invalid input type"""
        return self.run_test(
            "Generate Tests - Invalid Input Type",
            "POST",
            "generate-tests",
            422,  # FastAPI validation error
            data={
                "input_type": "invalid",
                "input_data": "test data",
                "ai_provider": "openai",
                "ai_api_key": "test-key",
                "test_framework": "jest"
            }
        )

    def test_generate_tests_invalid_provider(self):
        """Test generate-tests endpoint with invalid AI provider"""
        return self.run_test(
            "Generate Tests - Invalid AI Provider",
            "POST",
            "generate-tests",
            422,  # FastAPI validation error
            data={
                "input_type": "curl",
                "input_data": "curl -X GET https://api.example.com/users",
                "ai_provider": "invalid",
                "ai_api_key": "test-key",
                "test_framework": "jest"
            }
        )

    def test_generate_tests_invalid_framework(self):
        """Test generate-tests endpoint with invalid framework"""
        return self.run_test(
            "Generate Tests - Invalid Framework",
            "POST",
            "generate-tests",
            422,  # FastAPI validation error
            data={
                "input_type": "curl",
                "input_data": "curl -X GET https://api.example.com/users",
                "ai_provider": "openai",
                "ai_api_key": "test-key",
                "test_framework": "invalid"
            }
        )

    def test_generate_tests_curl_parsing(self):
        """Test generate-tests endpoint with valid cURL but invalid API key"""
        return self.run_test(
            "Generate Tests - cURL Parsing (Invalid API Key)",
            "POST",
            "generate-tests",
            200,  # Should return 200 but with success=false
            data={
                "input_type": "curl",
                "input_data": "curl -X GET https://api.example.com/users -H 'Authorization: Bearer token'",
                "api_type": "rest",
                "ai_provider": "openai",
                "ai_api_key": "invalid-key",
                "test_framework": "jest"
            }
        )

    def test_generate_tests_har_parsing(self):
        """Test generate-tests endpoint with HAR data"""
        har_data = {
            "log": {
                "entries": [
                    {
                        "request": {
                            "method": "GET",
                            "url": "https://api.example.com/users",
                            "headers": [
                                {"name": "Content-Type", "value": "application/json"}
                            ],
                            "postData": {"text": ""}
                        }
                    }
                ]
            }
        }
        
        return self.run_test(
            "Generate Tests - HAR Parsing (Invalid API Key)",
            "POST",
            "generate-tests",
            200,  # Should return 200 but with success=false
            data={
                "input_type": "har",
                "input_data": json.dumps(har_data),
                "api_type": "rest",
                "ai_provider": "openai",
                "ai_api_key": "invalid-key",
                "test_framework": "pytest"
            }
        )

    def test_generate_tests_text_input(self):
        """Test generate-tests endpoint with text input"""
        return self.run_test(
            "Generate Tests - Text Input (Invalid API Key)",
            "POST",
            "generate-tests",
            200,  # Should return 200 but with success=false
            data={
                "input_type": "text",
                "input_data": "POST /api/users - Creates a new user with name and email",
                "api_type": "rest",
                "ai_provider": "openai",
                "ai_api_key": "invalid-key",
                "test_framework": "mocha"
            }
        )

    def test_generate_tests_graphql_input(self):
        """Test generate-tests endpoint with GraphQL input"""
        graphql_query = """
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                name
                email
            }
        }
        """
        
        return self.run_test(
            "Generate Tests - GraphQL Input (Invalid API Key)",
            "POST",
            "generate-tests",
            200,  # Should return 200 but with success=false
            data={
                "input_type": "graphql",
                "input_data": graphql_query,
                "api_type": "graphql",
                "ai_provider": "openai",
                "ai_api_key": "invalid-key",
                "test_framework": "jest"
            }
        )

    def test_generate_tests_graphql_json_input(self):
        """Test generate-tests endpoint with GraphQL JSON input"""
        graphql_json = {
            "query": "mutation CreateUser($input: UserInput!) { createUser(input: $input) { id name email } }",
            "variables": {"input": {"name": "John Doe", "email": "john@example.com"}},
            "operationName": "CreateUser"
        }
        
        return self.run_test(
            "Generate Tests - GraphQL JSON Input (Invalid API Key)",
            "POST",
            "generate-tests",
            200,  # Should return 200 but with success=false
            data={
                "input_type": "graphql",
                "input_data": json.dumps(graphql_json),
                "api_type": "graphql",
                "ai_provider": "anthropic",
                "ai_api_key": "invalid-key",
                "test_framework": "pytest"
            }
        )

    def test_api_type_parameter_validation(self):
        """Test that api_type parameter is required and validated"""
        return self.run_test(
            "Generate Tests - Missing API Type",
            "POST",
            "generate-tests",
            422,  # FastAPI validation error
            data={
                "input_type": "curl",
                "input_data": "curl -X GET https://api.example.com/users",
                "ai_provider": "openai",
                "ai_api_key": "test-key",
                "test_framework": "jest"
                # Missing api_type
            }
        )

    def test_api_type_invalid_value(self):
        """Test api_type parameter with invalid value"""
        return self.run_test(
            "Generate Tests - Invalid API Type",
            "POST",
            "generate-tests",
            422,  # FastAPI validation error
            data={
                "input_type": "curl",
                "input_data": "curl -X GET https://api.example.com/users",
                "api_type": "invalid",
                "ai_provider": "openai",
                "ai_api_key": "test-key",
                "test_framework": "jest"
            }
        )

    def test_status_endpoints(self):
        """Test status check endpoints"""
        # Test POST status
        success, response = self.run_test(
            "Create Status Check",
            "POST",
            "status",
            200,
            data={"client_name": "test_client"}
        )
        
        if success:
            # Test GET status
            self.run_test(
                "Get Status Checks",
                "GET",
                "status",
                200
            )

    def test_curl_parsing_function(self):
        """Test cURL parsing with various formats"""
        test_cases = [
            {
                "name": "Simple GET cURL",
                "curl": "curl https://api.example.com/users",
                "expected_method": "GET"
            },
            {
                "name": "POST with data",
                "curl": "curl -X POST https://api.example.com/users -H 'Content-Type: application/json' -d '{\"name\":\"John\"}'",
                "expected_method": "POST"
            },
            {
                "name": "cURL with headers",
                "curl": "curl -H 'Authorization: Bearer token' https://api.example.com/users",
                "expected_method": "GET"
            }
        ]
        
        for case in test_cases:
            success, response = self.run_test(
                f"cURL Parsing - {case['name']}",
                "POST",
                "generate-tests",
                200,
                data={
                    "input_type": "curl",
                    "input_data": case["curl"],
                    "ai_provider": "openai",
                    "ai_api_key": "invalid-key",
                    "test_framework": "jest"
                }
            )

def main():
    print("🚀 Starting API Autopilot Backend Tests")
    print("=" * 60)
    
    tester = APIAutopilotTester()
    
    # Run all tests
    tester.test_root_endpoint()
    tester.test_generate_tests_missing_data()
    tester.test_generate_tests_invalid_input_type()
    tester.test_generate_tests_invalid_provider()
    tester.test_generate_tests_invalid_framework()
    tester.test_generate_tests_curl_parsing()
    tester.test_generate_tests_har_parsing()
    tester.test_generate_tests_text_input()
    tester.test_status_endpoints()
    tester.test_curl_parsing_function()
    
    # Print results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS")
    print(f"Tests passed: {tester.tests_passed}/{tester.tests_run}")
    print(f"Success rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for failure in tester.failed_tests:
            print(f"  - {failure.get('test', 'Unknown')}")
            if 'error' in failure:
                print(f"    Error: {failure['error']}")
            else:
                print(f"    Expected: {failure.get('expected')}, Got: {failure.get('actual')}")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())