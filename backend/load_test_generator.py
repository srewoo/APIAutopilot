# -*- coding: utf-8 -*-
"""
Load Test Generation Module
Generates load testing scripts for k6, JMeter, and Gatling
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime

class LoadTestFramework(Enum):
    """Supported load testing frameworks"""
    K6 = "k6"
    JMETER = "jmeter"
    GATLING = "gatling"
    ARTILLERY = "artillery"
    LOCUST = "locust"

@dataclass
class LoadTestConfig:
    """Configuration for load test generation"""
    framework: LoadTestFramework
    duration: int = 60  # seconds
    virtual_users: int = 10
    ramp_up_time: int = 10  # seconds
    iterations: Optional[int] = None
    think_time: int = 1  # seconds between requests

    # Thresholds
    max_response_time: int = 2000  # milliseconds
    error_rate_threshold: float = 0.01  # 1%
    requests_per_second: Optional[int] = None

    # Scenarios
    scenarios: List[str] = None  # smoke, load, stress, spike, soak

    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = ["load"]

@dataclass
class LoadTestScenario:
    """Load test scenario definition"""
    name: str
    stages: List[Dict[str, Any]]
    description: str

class LoadTestGenerator:
    """Generates load testing scripts for various frameworks"""

    def __init__(self):
        self.scenarios = {
            "smoke": LoadTestScenario(
                name="smoke",
                description="Minimal load to verify system works",
                stages=[
                    {"duration": "30s", "target": 1}
                ]
            ),
            "load": LoadTestScenario(
                name="load",
                description="Average expected load",
                stages=[
                    {"duration": "2m", "target": 10},
                    {"duration": "5m", "target": 10},
                    {"duration": "2m", "target": 0}
                ]
            ),
            "stress": LoadTestScenario(
                name="stress",
                description="Beyond normal load to find breaking point",
                stages=[
                    {"duration": "2m", "target": 10},
                    {"duration": "5m", "target": 50},
                    {"duration": "2m", "target": 100},
                    {"duration": "5m", "target": 100},
                    {"duration": "2m", "target": 0}
                ]
            ),
            "spike": LoadTestScenario(
                name="spike",
                description="Sudden increase in traffic",
                stages=[
                    {"duration": "10s", "target": 5},
                    {"duration": "30s", "target": 5},
                    {"duration": "10s", "target": 100},
                    {"duration": "3m", "target": 100},
                    {"duration": "10s", "target": 5},
                    {"duration": "30s", "target": 5},
                    {"duration": "10s", "target": 0}
                ]
            ),
            "soak": LoadTestScenario(
                name="soak",
                description="Extended period to detect memory leaks",
                stages=[
                    {"duration": "2m", "target": 20},
                    {"duration": "3h", "target": 20},
                    {"duration": "2m", "target": 0}
                ]
            )
        }

    def generate(self, api_data: Dict[str, Any], config: LoadTestConfig) -> str:
        """
        Generate load test script based on framework

        Args:
            api_data: API request data (method, url, headers, body)
            config: Load test configuration

        Returns:
            Generated load test script
        """
        if config.framework == LoadTestFramework.K6:
            return self._generate_k6_script(api_data, config)
        elif config.framework == LoadTestFramework.JMETER:
            return self._generate_jmeter_script(api_data, config)
        elif config.framework == LoadTestFramework.GATLING:
            return self._generate_gatling_script(api_data, config)
        elif config.framework == LoadTestFramework.ARTILLERY:
            return self._generate_artillery_script(api_data, config)
        elif config.framework == LoadTestFramework.LOCUST:
            return self._generate_locust_script(api_data, config)
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")

    def _generate_k6_script(self, api_data: Dict[str, Any], config: LoadTestConfig) -> str:
        """Generate k6 load test script"""

        # Prepare headers
        headers_str = json.dumps(api_data.get("headers", {}), indent=8)

        # Prepare body
        body = api_data.get("body", "")
        if body and isinstance(body, dict):
            body_str = f"JSON.stringify({json.dumps(body, indent=8)})"
        elif body:
            body_str = f"'{body}'"
        else:
            body_str = "null"

        # Get scenario stages
        stages = []
        for scenario_name in config.scenarios:
            if scenario_name in self.scenarios:
                stages.extend(self.scenarios[scenario_name].stages)

        stages_str = json.dumps(stages, indent=4)

        script = f"""import http from 'k6/http';
import {{ check, sleep }} from 'k6';
import {{ Rate }} from 'k6/metrics';
import {{ randomIntBetween }} from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {{
    stages: {stages_str},
    thresholds: {{
        'http_req_duration': ['p(95)<{config.max_response_time}'], // 95% of requests must complete below {config.max_response_time}ms
        'http_req_failed': ['rate<{config.error_rate_threshold}'], // Error rate must be below {config.error_rate_threshold*100}%
        'errors': ['rate<{config.error_rate_threshold}'], // Custom error rate
    }},
}};

// API Configuration
const BASE_URL = __ENV.BASE_URL || '{api_data.get("url", "http://localhost:8000")}';
const API_ENDPOINT = '{api_data.get("url", "")}';
const METHOD = '{api_data.get("method", "GET")}';

// Request headers
const headers = {headers_str};

// Request body
const requestBody = {body_str};

// Setup function - runs once per VU
export function setup() {{
    console.log('Load test starting...');
    console.log('Target URL:', API_ENDPOINT);
    console.log('Method:', METHOD);
    console.log('Virtual Users:', options.stages[options.stages.length - 2].target);

    // Optionally perform authentication here
    // const authResponse = http.post(`${{BASE_URL}}/auth/login`, ...);
    // return {{ token: authResponse.json('token') }};
    return {{}};
}}

// Main test function - runs continuously for each VU
export default function(data) {{
    // Prepare request options
    const params = {{
        headers: headers,
        timeout: '30s',
        tags: {{
            name: 'API_Test',
            scenario: '{config.scenarios[0] if config.scenarios else "load"}'
        }}
    }};

    // Make the request
    let response;

    switch(METHOD.toUpperCase()) {{
        case 'GET':
            response = http.get(API_ENDPOINT, params);
            break;
        case 'POST':
            response = http.post(API_ENDPOINT, requestBody, params);
            break;
        case 'PUT':
            response = http.put(API_ENDPOINT, requestBody, params);
            break;
        case 'PATCH':
            response = http.patch(API_ENDPOINT, requestBody, params);
            break;
        case 'DELETE':
            response = http.del(API_ENDPOINT, params);
            break;
        default:
            response = http.request(METHOD, API_ENDPOINT, requestBody, params);
    }}

    // Verify response
    const success = check(response, {{
        'status is 200': (r) => r.status === 200,
        'status is not 500': (r) => r.status !== 500,
        'response time < {config.max_response_time}ms': (r) => r.timings.duration < {config.max_response_time},
        'response has body': (r) => r.body && r.body.length > 0,
    }});

    // Track errors
    errorRate.add(!success);

    // Log errors for debugging
    if (!success) {{
        console.error(`Request failed: ${{response.status}} - ${{response.body}}`);
    }}

    // Think time between requests
    sleep(randomIntBetween({config.think_time}, {config.think_time + 2}));
}}

// Teardown function - runs once after all VUs finish
export function teardown(data) {{
    console.log('Load test completed');
}}

// Custom scenarios for different load patterns
export function smokeTest() {{
    const response = http.get(API_ENDPOINT, {{ headers }});
    check(response, {{
        'smoke test passed': (r) => r.status === 200
    }});
}}

export function stressTest() {{
    const responses = http.batch([
        ['GET', API_ENDPOINT, null, {{ headers }}],
        ['GET', API_ENDPOINT, null, {{ headers }}],
        ['GET', API_ENDPOINT, null, {{ headers }}],
    ]);

    responses.forEach(response => {{
        check(response, {{
            'batch request successful': (r) => r.status === 200
        }});
    }});
}}

// Run with: k6 run load_test.js
// Run with dashboard: k6 run --out dashboard load_test.js
// Run with cloud: k6 cloud load_test.js
"""
        return script

    def _generate_jmeter_script(self, api_data: Dict[str, Any], config: LoadTestConfig) -> str:
        """Generate JMeter JMX test plan"""

        # Create root element
        root = ET.Element("jmeterTestPlan", version="1.2", properties="5.0", jmeter="5.4.1")

        # Add test plan
        test_plan_tree = ET.SubElement(root, "hashTree")
        test_plan = ET.SubElement(test_plan_tree, "TestPlan",
                                  guiclass="TestPlanGui",
                                  testclass="TestPlan",
                                  testname="API Load Test Plan",
                                  enabled="true")

        # Add test plan properties
        ET.SubElement(test_plan, "stringProp", name="TestPlan.comments").text = f"Load test for {api_data.get('url', 'API')}"
        ET.SubElement(test_plan, "boolProp", name="TestPlan.functional_mode").text = "false"
        ET.SubElement(test_plan, "boolProp", name="TestPlan.tearDown_on_shutdown").text = "true"

        # Add thread group
        thread_group_tree = ET.SubElement(test_plan_tree, "hashTree")
        thread_group = ET.SubElement(thread_group_tree, "ThreadGroup",
                                    guiclass="ThreadGroupGui",
                                    testclass="ThreadGroup",
                                    testname="Load Test Thread Group",
                                    enabled="true")

        # Thread group properties
        ET.SubElement(thread_group, "stringProp", name="ThreadGroup.on_sample_error").text = "continue"
        ET.SubElement(thread_group, "stringProp", name="ThreadGroup.num_threads").text = str(config.virtual_users)
        ET.SubElement(thread_group, "stringProp", name="ThreadGroup.ramp_time").text = str(config.ramp_up_time)
        ET.SubElement(thread_group, "boolProp", name="ThreadGroup.scheduler").text = "true"
        ET.SubElement(thread_group, "stringProp", name="ThreadGroup.duration").text = str(config.duration)

        # Add HTTP Request Sampler
        http_sampler_tree = ET.SubElement(thread_group_tree, "hashTree")
        http_sampler = ET.SubElement(http_sampler_tree, "HTTPSamplerProxy",
                                    guiclass="HttpTestSampleGui",
                                    testclass="HTTPSamplerProxy",
                                    testname=f"{api_data.get('method', 'GET')} Request",
                                    enabled="true")

        # Parse URL
        from urllib.parse import urlparse
        parsed_url = urlparse(api_data.get("url", "http://localhost:8000"))

        # HTTP Sampler properties
        ET.SubElement(http_sampler, "stringProp", name="HTTPSampler.domain").text = parsed_url.hostname or "localhost"
        ET.SubElement(http_sampler, "stringProp", name="HTTPSampler.port").text = str(parsed_url.port or (443 if parsed_url.scheme == "https" else 80))
        ET.SubElement(http_sampler, "stringProp", name="HTTPSampler.protocol").text = parsed_url.scheme or "http"
        ET.SubElement(http_sampler, "stringProp", name="HTTPSampler.path").text = parsed_url.path or "/"
        ET.SubElement(http_sampler, "stringProp", name="HTTPSampler.method").text = api_data.get("method", "GET")

        # Add request body if present
        if api_data.get("body"):
            body_str = json.dumps(api_data["body"]) if isinstance(api_data["body"], dict) else api_data["body"]
            ET.SubElement(http_sampler, "stringProp", name="HTTPSampler.postBodyRaw").text = body_str
            ET.SubElement(http_sampler, "boolProp", name="HTTPSampler.postBodyRaw").text = "true"

        # Add headers
        if api_data.get("headers"):
            header_manager_tree = ET.SubElement(http_sampler_tree, "hashTree")
            header_manager = ET.SubElement(header_manager_tree, "HeaderManager",
                                          guiclass="HeaderPanel",
                                          testclass="HeaderManager",
                                          testname="HTTP Header Manager",
                                          enabled="true")

            headers_collection = ET.SubElement(header_manager, "collectionProp", name="HeaderManager.headers")
            for header_name, header_value in api_data["headers"].items():
                header_elem = ET.SubElement(headers_collection, "elementProp", name="", elementType="Header")
                ET.SubElement(header_elem, "stringProp", name="Header.name").text = header_name
                ET.SubElement(header_elem, "stringProp", name="Header.value").text = header_value

        # Add response assertions
        assertion_tree = ET.SubElement(http_sampler_tree, "hashTree")
        response_assertion = ET.SubElement(assertion_tree, "ResponseAssertion",
                                          guiclass="AssertionGui",
                                          testclass="ResponseAssertion",
                                          testname="Response Assertion",
                                          enabled="true")

        ET.SubElement(response_assertion, "stringProp", name="Assertion.test_field").text = "Assertion.response_code"
        ET.SubElement(response_assertion, "stringProp", name="Assertion.assume_success").text = "true"
        ET.SubElement(response_assertion, "intProp", name="Assertion.test_type").text = "8"
        assertions_collection = ET.SubElement(response_assertion, "collectionProp", name="Asserion.test_strings")
        ET.SubElement(assertions_collection, "stringProp").text = "200"

        # Add duration assertion
        duration_assertion = ET.SubElement(assertion_tree, "DurationAssertion",
                                          guiclass="DurationAssertionGui",
                                          testclass="DurationAssertion",
                                          testname="Duration Assertion",
                                          enabled="true")
        ET.SubElement(duration_assertion, "stringProp", name="DurationAssertion.duration").text = str(config.max_response_time)

        # Add listeners for results
        listeners_tree = ET.SubElement(thread_group_tree, "hashTree")

        # View Results Tree
        ET.SubElement(listeners_tree, "ResultCollector",
                     guiclass="ViewResultsFullVisualizer",
                     testclass="ResultCollector",
                     testname="View Results Tree",
                     enabled="true")

        # Summary Report
        ET.SubElement(listeners_tree, "ResultCollector",
                     guiclass="SummaryReport",
                     testclass="ResultCollector",
                     testname="Summary Report",
                     enabled="true")

        # Convert to string
        tree = ET.ElementTree(root)
        import io
        output = io.StringIO()
        tree.write(output, encoding='unicode', xml_declaration=True)
        return output.getvalue()

    def _generate_gatling_script(self, api_data: Dict[str, Any], config: LoadTestConfig) -> str:
        """Generate Gatling Scala simulation"""

        # Prepare headers
        headers = api_data.get("headers", {})
        headers_str = ",\n        ".join([f'"{k}" -> "{v}"' for k, v in headers.items()])

        # Prepare body
        body = api_data.get("body", "")
        if body and isinstance(body, dict):
            body_str = f'.body(StringBody(""""{json.dumps(body)}"""")).asJson'
        elif body:
            body_str = f'.body(StringBody("{body}"))'
        else:
            body_str = ""

        # Method
        method = api_data.get("method", "GET").lower()

        script = f"""package loadtest

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._
import java.util.concurrent.ThreadLocalRandom

class APILoadTestSimulation extends Simulation {{

  // HTTP Configuration
  val httpProtocol = http
    .baseUrl("{api_data.get('url', 'http://localhost:8000')}")
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")
    .userAgentHeader("Gatling Load Test")

  // Headers
  val headers = Map(
        {headers_str}
    )

  // Scenario Definition
  val apiScenario = scenario("API Load Test Scenario")
    .exec(
      http("API Request")
        .{method}("{api_data.get('url', '/')}")
        .headers(headers)
        {body_str}
        .check(status.is(200))
        .check(responseTimeInMillis.lt({config.max_response_time}))
    )
    .pause({config.think_time})

  // Load Injection Profiles
  val loadProfile = {{
    "{config.scenarios[0] if config.scenarios else 'load'}" match {{
      case "smoke" =>
        // Smoke test - minimal load
        constantUsersPerSec(1) during (30 seconds)

      case "load" =>
        // Normal load test
        rampUsers({config.virtual_users}) during ({config.ramp_up_time} seconds),
        constantUsersPerSec({config.virtual_users}) during ({config.duration} seconds)

      case "stress" =>
        // Stress test - increasing load
        rampUsersPerSec(1) to ({config.virtual_users}) during (30 seconds),
        constantUsersPerSec({config.virtual_users}) during (2 minutes),
        rampUsersPerSec({config.virtual_users}) to ({config.virtual_users * 5}) during (30 seconds),
        constantUsersPerSec({config.virtual_users * 5}) during (5 minutes)

      case "spike" =>
        // Spike test - sudden increase
        constantUsersPerSec(5) during (1 minute),
        nothingFor(10 seconds),
        atOnceUsers({config.virtual_users * 10}),
        constantUsersPerSec(5) during (1 minute)

      case _ =>
        // Default load profile
        rampUsers({config.virtual_users}) during ({config.ramp_up_time} seconds)
    }}
  }}

  // Setup
  setUp(
    apiScenario.inject(loadProfile)
  ).protocols(httpProtocol)
    .assertions(
      global.responseTime.max.lt({config.max_response_time}),
      global.responseTime.mean.lt({config.max_response_time // 2}),
      global.successfulRequests.percent.gt({(1 - config.error_rate_threshold) * 100}),
      global.requestsPerSec.gt({config.requests_per_second if config.requests_per_second else 1})
    )
}}

// Run with: gatling:test -Dgatling.simulationClass=loadtest.APILoadTestSimulation
"""
        return script

    def _generate_artillery_script(self, api_data: Dict[str, Any], config: LoadTestConfig) -> str:
        """Generate Artillery YAML configuration"""

        import yaml

        # Build Artillery configuration
        artillery_config = {
            "config": {
                "target": api_data.get("url", "http://localhost:8000"),
                "phases": [],
                "ensure": {
                    "maxErrorRate": config.error_rate_threshold * 100,
                    "p95": config.max_response_time
                }
            },
            "scenarios": [
                {
                    "name": "API Load Test",
                    "flow": []
                }
            ]
        }

        # Add phases based on scenario
        for scenario_name in config.scenarios:
            if scenario_name == "smoke":
                artillery_config["config"]["phases"].append({
                    "duration": 30,
                    "arrivalRate": 1
                })
            elif scenario_name == "load":
                artillery_config["config"]["phases"].extend([
                    {"duration": config.ramp_up_time, "arrivalRate": 1, "rampTo": config.virtual_users},
                    {"duration": config.duration, "arrivalRate": config.virtual_users}
                ])
            elif scenario_name == "stress":
                artillery_config["config"]["phases"].extend([
                    {"duration": 60, "arrivalRate": config.virtual_users},
                    {"duration": 60, "arrivalRate": config.virtual_users * 2},
                    {"duration": 120, "arrivalRate": config.virtual_users * 5}
                ])
            elif scenario_name == "spike":
                artillery_config["config"]["phases"].extend([
                    {"duration": 30, "arrivalRate": 5},
                    {"duration": 10, "arrivalRate": config.virtual_users * 10},
                    {"duration": 30, "arrivalRate": 5}
                ])

        # Add headers if present
        if api_data.get("headers"):
            artillery_config["config"]["http"] = {
                "headers": api_data["headers"]
            }

        # Build request flow
        method = api_data.get("method", "GET").lower()
        request = {method: api_data.get("url", "/")}

        if api_data.get("body"):
            if isinstance(api_data["body"], dict):
                request["json"] = api_data["body"]
            else:
                request["body"] = api_data["body"]

        # Add expectations
        request["expect"] = [
            {"statusCode": 200},
            {"responseTime": {"max": config.max_response_time}}
        ]

        # Add think time
        artillery_config["scenarios"][0]["flow"] = [
            request,
            {"think": config.think_time}
        ]

        # Convert to YAML
        return yaml.dump(artillery_config, default_flow_style=False, sort_keys=False)

    def _generate_locust_script(self, api_data: Dict[str, Any], config: LoadTestConfig) -> str:
        """Generate Locust Python script"""

        # Prepare headers
        headers = api_data.get("headers", {})
        headers_str = json.dumps(headers, indent=8) if headers else "{}"

        # Prepare body
        body = api_data.get("body", "")
        if body and isinstance(body, dict):
            body_str = f"json={json.dumps(body, indent=12)}"
        elif body:
            body_str = f'data="{body}"'
        else:
            body_str = ""

        # Method
        method = api_data.get("method", "GET").lower()

        script = f"""from locust import HttpUser, task, between, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from locust.log import setup_logging
import time
import json
import random

# Setup logging
setup_logging("INFO")

class APILoadTestUser(HttpUser):
    \"\"\"
    Locust user class for API load testing
    \"\"\"

    # Wait time between requests (think time)
    wait_time = between({config.think_time}, {config.think_time + 2})

    # Base URL
    host = "{api_data.get('url', 'http://localhost:8000')}"

    # Headers
    headers = {headers_str}

    def on_start(self):
        \"\"\"Called when a user starts\"\"\"
        # Optional: Perform authentication here
        # self.client.post("/auth/login", json={{"username": "test", "password": "test"}})
        pass

    @task(1)
    def test_api_endpoint(self):
        \"\"\"Main API test task\"\"\"

        # Make the request
        with self.client.{method}(
            "{api_data.get('url', '/')}",
            headers=self.headers,
            {body_str},
            catch_response=True,
            name="API Request"
        ) as response:
            # Validate response
            if response.status_code == 200:
                # Additional validations
                if response.elapsed.total_seconds() * 1000 > {config.max_response_time}:
                    response.failure(f"Response time exceeded {{response.elapsed.total_seconds() * 1000}}ms")
                else:
                    response.success()
            else:
                response.failure(f"Got status code {{response.status_code}}")

    @task(weight=0.1)  # Run occasionally
    def health_check(self):
        \"\"\"Health check task\"\"\"
        self.client.get("/health", name="Health Check")

    def on_stop(self):
        \"\"\"Called when a user stops\"\"\"
        pass


class StagesShape:
    \"\"\"
    Custom load shape for different test scenarios
    \"\"\"

    def __init__(self):
        self.scenario = "{config.scenarios[0] if config.scenarios else 'load'}"

    def tick(self):
        run_time = self.get_run_time()

        if self.scenario == "smoke":
            # Smoke test - minimal load
            if run_time < 30:
                return (1, 1)
            return None

        elif self.scenario == "load":
            # Normal load test
            if run_time < {config.ramp_up_time}:
                user_count = int(run_time * {config.virtual_users} / {config.ramp_up_time})
                return (user_count, user_count)
            elif run_time < {config.duration + config.ramp_up_time}:
                return ({config.virtual_users}, {config.virtual_users})
            else:
                return None

        elif self.scenario == "stress":
            # Stress test - increasing load
            if run_time < 60:
                return ({config.virtual_users}, {config.virtual_users})
            elif run_time < 120:
                return ({config.virtual_users * 2}, {config.virtual_users * 2})
            elif run_time < 300:
                return ({config.virtual_users * 5}, {config.virtual_users * 5})
            else:
                return None

        elif self.scenario == "spike":
            # Spike test
            if run_time < 30:
                return (5, 5)
            elif run_time < 40:
                return ({config.virtual_users * 10}, {config.virtual_users * 10})
            elif run_time < 70:
                return (5, 5)
            else:
                return None

        # Default
        return ({config.virtual_users}, {config.virtual_users})


# Event handlers for custom metrics
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    \"\"\"Track custom metrics\"\"\"
    if response_time > {config.max_response_time}:
        print(f"⚠️ Slow request: {{name}} took {{response_time}}ms")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    \"\"\"Print final statistics\"\"\"
    print("\\n📊 Test Results Summary:")
    print(f"Total Requests: {{environment.stats.total.num_requests}}")
    print(f"Failed Requests: {{environment.stats.total.num_failures}}")
    print(f"Median Response Time: {{environment.stats.total.median_response_time}}ms")
    print(f"95th Percentile: {{environment.stats.total.get_response_time_percentile(0.95)}}ms")
    print(f"RPS: {{environment.stats.total.current_rps}}")


# Run with: locust -f load_test.py --headless -u {config.virtual_users} -r {config.ramp_up_time} -t {config.duration}s
# Run with Web UI: locust -f load_test.py
"""
        return script

    def generate_all_frameworks(self, api_data: Dict[str, Any], base_config: LoadTestConfig) -> Dict[str, str]:
        """Generate load test scripts for all supported frameworks"""

        scripts = {}

        for framework in LoadTestFramework:
            config = LoadTestConfig(
                framework=framework,
                duration=base_config.duration,
                virtual_users=base_config.virtual_users,
                ramp_up_time=base_config.ramp_up_time,
                think_time=base_config.think_time,
                max_response_time=base_config.max_response_time,
                error_rate_threshold=base_config.error_rate_threshold,
                scenarios=base_config.scenarios
            )

            scripts[framework.value] = self.generate(api_data, config)

        return scripts


# Export classes
__all__ = [
    'LoadTestGenerator',
    'LoadTestConfig',
    'LoadTestFramework',
    'LoadTestScenario'
]