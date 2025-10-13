# -*- coding: utf-8 -*-
"""
Framework-specific test templates and code generation patterns
"""

from typing import Dict, Any

class FrameworkTemplates:
    """Templates for different test frameworks"""
    
    @staticmethod
    def get_restassured_instructions() -> str:
        """Get RestAssured-specific generation instructions"""
        return """
FRAMEWORK: RestAssured (Java)

REQUIRED STRUCTURE:
```java
import io.restassured.RestAssured;
import io.restassured.response.Response;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;

public class ApiTests {
    
    private static final String BASE_URL = "API_BASE_URL_HERE";
    
    @BeforeClass
    public void setup() {
        RestAssured.baseURI = BASE_URL;
    }
    
    @Test
    public void testPositiveScenario() {
        given()
            .contentType("application/json")
            .header("Authorization", "Bearer TOKEN")
            .body("{\\"key\\": \\"value\\"}")
        .when()
            .post("/endpoint")
        .then()
            .statusCode(200)
            .body("id", notNullValue())
            .body("status", equalTo("success"));
    }
    
    @Test
    public void testNegativeScenario() {
        given()
            .contentType("application/json")
            .body("{}")
        .when()
            .post("/endpoint")
        .then()
            .statusCode(400)
            .body("error", notNullValue());
    }
}
```

CRITICAL REQUIREMENTS:
1. Use given().when().then() syntax
2. Use Hamcrest matchers (equalTo, notNullValue, hasSize, etc.)
3. Include @BeforeClass setup method
4. Use @Test annotations
5. Proper JSON escaping in body strings
6. Include comprehensive assertions with body() matchers
"""
    
    @staticmethod
    def get_behave_instructions() -> str:
        """Get Behave BDD-specific generation instructions"""
        return """
FRAMEWORK: Behave BDD (Python)

Generate TWO separate files:

=== FILE: features/api_test.feature ===
Feature: API Testing
  As a QA engineer
  I want to test the API
  So that I can ensure it works correctly

  Background:
    Given the API base URL is "BASE_URL_HERE"
    And I have valid authentication credentials

  Scenario: Successful API call with valid data
    Given I have a valid request body
    When I send a POST request to "/endpoint"
    Then the response status code should be 200
    And the response should contain field "id"
    And the response field "status" should equal "success"

  Scenario: API call with missing required field
    Given I have a request body missing "name"
    When I send a POST request to "/endpoint"
    Then the response status code should be 400
    And the response should contain field "error"

  Scenario Outline: API call with invalid data
    Given I have a request body with <field> as <invalid_value>
    When I send a POST request to "/endpoint"
    Then the response status code should be 400
    
    Examples:
      | field  | invalid_value |
      | email  | "invalid"     |
      | age    | "text"        |

=== FILE: features/steps/api_steps.py ===
# -*- coding: utf-8 -*-
from behave import given, when, then
import requests
import json

@given('the API base URL is "{base_url}"')
def step_impl(context, base_url):
    context.base_url = base_url

@given('I have valid authentication credentials')
def step_impl(context):
    context.headers = {
        "Authorization": "Bearer YOUR_TOKEN",
        "Content-Type": "application/json"
    }

@given('I have a valid request body')
def step_impl(context):
    context.request_body = {"key": "value"}

@when('I send a {method} request to "{endpoint}"')
def step_impl(context, method, endpoint):
    url = context.base_url + endpoint
    if method == "POST":
        context.response = requests.post(
            url, 
            headers=context.headers, 
            json=context.request_body
        )

@then('the response status code should be {status_code:d}')
def step_impl(context, status_code):
    assert context.response.status_code == status_code

@then('the response should contain field "{field}"')
def step_impl(context, field):
    response_data = context.response.json()
    assert field in response_data

CRITICAL REQUIREMENTS:
1. Use Gherkin syntax (Feature, Scenario, Given/When/Then)
2. Separate feature file and step definitions
3. Use clear, readable scenario descriptions
4. Include Scenario Outline for data-driven tests
5. Implement all step definitions in Python
6. Use requests library for API calls
"""
    
    @staticmethod
    def get_framework_instructions(framework: str) -> str:
        """Get framework-specific instructions for AI prompt"""
        instructions = {
            "restassured": FrameworkTemplates.get_restassured_instructions(),
            "behave": FrameworkTemplates.get_behave_instructions(),
        }
        return instructions.get(framework, "")
    
    @staticmethod
    def get_framework_language(framework: str) -> str:
        """Get programming language for framework"""
        languages = {
            "jest": "javascript",
            "mocha": "javascript",
            "cypress": "javascript",
            "pytest": "python",
            "requests": "python",
            "testng": "java",
            "junit": "java",
            "restassured": "java",
            "behave": "python",
        }
        return languages.get(framework, "text")
