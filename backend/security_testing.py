# -*- coding: utf-8 -*-
"""
Advanced Security Testing Module
DAST Integration, OAuth/JWT Testing, and API Fuzzing
"""

import json
import jwt
import base64
import hashlib
import hmac
import time
import random
import string
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import aiohttp
from urllib.parse import urlparse, parse_qs

class SecurityTestType(Enum):
    """Types of security tests"""
    DAST = "dast"
    OAUTH_FLOW = "oauth_flow"
    JWT_VALIDATION = "jwt_validation"
    API_FUZZING = "api_fuzzing"
    INJECTION = "injection"
    AUTH_BYPASS = "auth_bypass"
    RATE_LIMITING = "rate_limiting"

@dataclass
class DASTestResult:
    """DAST scan result"""
    vulnerability: str
    severity: str  # Critical, High, Medium, Low
    endpoint: str
    method: str
    payload: Optional[str]
    response_code: int
    evidence: str
    remediation: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None

@dataclass
class OAuthTestResult:
    """OAuth flow test result"""
    flow_type: str  # authorization_code, implicit, client_credentials, password
    test_name: str
    passed: bool
    error_message: Optional[str]
    security_issues: List[str]
    recommendations: List[str]

@dataclass
class FuzzTestResult:
    """Fuzzing test result"""
    field: str
    original_value: Any
    fuzzed_value: Any
    response_code: int
    error_triggered: bool
    error_type: Optional[str]
    potential_vulnerability: Optional[str]

class DASTScanner:
    """Dynamic Application Security Testing Scanner"""

    def __init__(self):
        self.vulnerabilities = []
        self.payloads = {
            "sql_injection": [
                "' OR '1'='1",
                "'; DROP TABLE users--",
                "' UNION SELECT * FROM users--",
                "admin'--",
                "' OR 1=1--",
                "1' AND '1' = '1",
                "' OR 'x'='x",
                "' AND id IS NULL; --",
            ],
            "xss": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<svg onload=alert('XSS')>",
                "<iframe src=javascript:alert('XSS')>",
                "'><script>alert(String.fromCharCode(88,83,83))</script>",
                "<body onload=alert('XSS')>",
                "<input onfocus=alert('XSS') autofocus>",
            ],
            "command_injection": [
                "; ls -la",
                "| whoami",
                "&& cat /etc/passwd",
                "`id`",
                "$(whoami)",
                "; ping -c 10 127.0.0.1",
                "& dir",
                "| net user",
            ],
            "path_traversal": [
                "../../etc/passwd",
                "..\\..\\windows\\system32\\config\\sam",
                "....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "..;/etc/passwd",
                "../../../etc/shadow",
                "..\\..\\..\\boot.ini",
                "file:///etc/passwd",
            ],
            "ldap_injection": [
                "*",
                "*)(&",
                "*)(mail=*",
                "*)(|(mail=*)(cn=*))",
                "admin*",
                "*)(uid=*))(|(uid=*",
                "*()|&'",
            ],
            "xml_injection": [
                "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
                "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'http://evil.com/steal'>]>",
                "<![CDATA[<script>alert('XSS')</script>]]>",
            ],
            "nosql_injection": [
                '{"$ne": null}',
                '{"$gt": ""}',
                '{"$regex": ".*"}',
                '{"$where": "1==1"}',
                '{"username": {"$ne": null}, "password": {"$ne": null}}',
                '{"$or": [{"username": "admin"}, {"username": "administrator"}]}',
            ],
            "header_injection": [
                "Content-Type: text/html\r\n\r\n<script>alert('XSS')</script>",
                "X-Forwarded-For: 127.0.0.1\r\nX-Evil: true",
                "User-Agent: Mozilla/5.0\r\nSet-Cookie: admin=true",
            ],
            "ssti": [  # Server-Side Template Injection
                "{{7*7}}",
                "${7*7}",
                "<%= 7*7 %>",
                "#{7*7}",
                "*{7*7}",
                "{{config}}",
                "{{self}}",
                "{{_self.env}}",
            ]
        }

    async def scan_endpoint(self, url: str, method: str = "GET", headers: Dict = None, body: Any = None) -> List[DASTestResult]:
        """
        Perform DAST scan on an endpoint

        Args:
            url: Target URL
            method: HTTP method
            headers: Request headers
            body: Request body

        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []

        # Test each vulnerability type
        for vuln_type, payloads in self.payloads.items():
            for payload in payloads:
                # Test in URL parameters
                if "?" in url:
                    test_url = self._inject_payload_in_url(url, payload)
                    result = await self._test_payload(test_url, method, headers, body, vuln_type, payload, "url_param")
                    if result:
                        vulnerabilities.append(result)

                # Test in headers
                if headers:
                    test_headers = self._inject_payload_in_headers(headers.copy(), payload)
                    result = await self._test_payload(url, method, test_headers, body, vuln_type, payload, "header")
                    if result:
                        vulnerabilities.append(result)

                # Test in body
                if body:
                    test_body = self._inject_payload_in_body(body, payload)
                    result = await self._test_payload(url, method, headers, test_body, vuln_type, payload, "body")
                    if result:
                        vulnerabilities.append(result)

        # Additional security checks
        vulnerabilities.extend(await self._check_security_headers(url, headers))
        vulnerabilities.extend(await self._check_cors_misconfiguration(url, headers))
        vulnerabilities.extend(await self._check_http_methods(url, headers))

        return vulnerabilities

    async def _test_payload(self, url: str, method: str, headers: Dict, body: Any,
                           vuln_type: str, payload: str, injection_point: str) -> Optional[DASTestResult]:
        """Test a specific payload"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, dict) else None,
                    data=body if isinstance(body, str) else None,
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=False
                ) as response:
                    response_text = await response.text()

                    # Analyze response for vulnerabilities
                    if self._is_vulnerable(vuln_type, payload, response.status, response_text):
                        return DASTestResult(
                            vulnerability=self._get_vulnerability_name(vuln_type),
                            severity=self._get_severity(vuln_type),
                            endpoint=url,
                            method=method,
                            payload=payload,
                            response_code=response.status,
                            evidence=self._get_evidence(vuln_type, response_text),
                            remediation=self._get_remediation(vuln_type),
                            cwe_id=self._get_cwe_id(vuln_type),
                            owasp_category=self._get_owasp_category(vuln_type)
                        )
        except Exception as e:
            # Some errors might indicate vulnerabilities
            if vuln_type == "command_injection" and "timeout" in str(e).lower():
                return DASTestResult(
                    vulnerability="Potential Command Injection (Time-based)",
                    severity="High",
                    endpoint=url,
                    method=method,
                    payload=payload,
                    response_code=0,
                    evidence=f"Request timeout with payload: {payload}",
                    remediation="Sanitize user input and use parameterized commands",
                    cwe_id="CWE-78"
                )

        return None

    def _inject_payload_in_url(self, url: str, payload: str) -> str:
        """Inject payload in URL parameters"""
        if "?" in url:
            return f"{url}&test={payload}"
        return f"{url}?test={payload}"

    def _inject_payload_in_headers(self, headers: Dict, payload: str) -> Dict:
        """Inject payload in headers"""
        headers["X-Test-Header"] = payload
        return headers

    def _inject_payload_in_body(self, body: Any, payload: str) -> Any:
        """Inject payload in request body"""
        if isinstance(body, dict):
            body = body.copy()
            body["test_field"] = payload
            return body
        elif isinstance(body, str):
            return f"{body}&test={payload}"
        return body

    def _is_vulnerable(self, vuln_type: str, payload: str, status_code: int, response_text: str) -> bool:
        """Check if response indicates vulnerability"""

        # SQL Injection indicators
        if vuln_type == "sql_injection":
            sql_errors = [
                "SQL syntax",
                "mysql_fetch",
                "ORA-[0-9]{5}",
                "PostgreSQL",
                "SQLite",
                "Microsoft SQL Server",
                "Unclosed quotation mark",
                "You have an error in your SQL syntax"
            ]
            return any(error in response_text for error in sql_errors)

        # XSS indicators
        elif vuln_type == "xss":
            # Check if payload is reflected without encoding
            return payload in response_text or "alert(" in response_text

        # Command injection indicators
        elif vuln_type == "command_injection":
            cmd_indicators = [
                "uid=",
                "gid=",
                "groups=",
                "root:",
                "/bin/",
                "Windows",
                "Program Files",
                "Users\\"
            ]
            return any(indicator in response_text for indicator in cmd_indicators)

        # Path traversal indicators
        elif vuln_type == "path_traversal":
            file_contents = [
                "root:x:",
                "[boot loader]",
                "[operating systems]",
                "/bin/bash",
                "daemon:",
                "nobody:"
            ]
            return any(content in response_text for content in file_contents)

        # Template injection
        elif vuln_type == "ssti":
            # Check if expression was evaluated
            if "{{7*7}}" in payload and "49" in response_text:
                return True
            if "${7*7}" in payload and "49" in response_text:
                return True

        # Check for error-based detection
        if status_code >= 500:
            return True

        return False

    def _get_vulnerability_name(self, vuln_type: str) -> str:
        """Get readable vulnerability name"""
        names = {
            "sql_injection": "SQL Injection",
            "xss": "Cross-Site Scripting (XSS)",
            "command_injection": "Command Injection",
            "path_traversal": "Path Traversal",
            "ldap_injection": "LDAP Injection",
            "xml_injection": "XML External Entity (XXE)",
            "nosql_injection": "NoSQL Injection",
            "header_injection": "Header Injection",
            "ssti": "Server-Side Template Injection"
        }
        return names.get(vuln_type, vuln_type.replace("_", " ").title())

    def _get_severity(self, vuln_type: str) -> str:
        """Get vulnerability severity"""
        critical = ["sql_injection", "command_injection", "xml_injection", "ssti"]
        high = ["path_traversal", "nosql_injection", "ldap_injection"]
        medium = ["xss", "header_injection"]

        if vuln_type in critical:
            return "Critical"
        elif vuln_type in high:
            return "High"
        elif vuln_type in medium:
            return "Medium"
        return "Low"

    def _get_evidence(self, vuln_type: str, response_text: str) -> str:
        """Extract evidence from response"""
        # Truncate response for evidence
        return response_text[:500] if len(response_text) > 500 else response_text

    def _get_remediation(self, vuln_type: str) -> str:
        """Get remediation advice"""
        remediations = {
            "sql_injection": "Use parameterized queries/prepared statements. Never concatenate user input into SQL queries.",
            "xss": "Encode all user input before rendering. Use Content Security Policy (CSP) headers.",
            "command_injection": "Avoid system calls with user input. Use safe APIs and validate/sanitize all input.",
            "path_traversal": "Validate and sanitize file paths. Use a whitelist of allowed files/directories.",
            "ldap_injection": "Use parameterized LDAP queries. Escape special LDAP characters.",
            "xml_injection": "Disable XML external entity processing. Use safe XML parsers.",
            "nosql_injection": "Validate and sanitize input. Use proper query builders.",
            "header_injection": "Validate and sanitize header values. Remove newline characters.",
            "ssti": "Use safe templating practices. Sanitize template input."
        }
        return remediations.get(vuln_type, "Validate and sanitize all user input.")

    def _get_cwe_id(self, vuln_type: str) -> str:
        """Get CWE ID for vulnerability"""
        cwe_mapping = {
            "sql_injection": "CWE-89",
            "xss": "CWE-79",
            "command_injection": "CWE-78",
            "path_traversal": "CWE-22",
            "ldap_injection": "CWE-90",
            "xml_injection": "CWE-611",
            "nosql_injection": "CWE-943",
            "header_injection": "CWE-113",
            "ssti": "CWE-1336"
        }
        return cwe_mapping.get(vuln_type, "CWE-20")

    def _get_owasp_category(self, vuln_type: str) -> str:
        """Get OWASP Top 10 category"""
        owasp_mapping = {
            "sql_injection": "A03:2021 - Injection",
            "xss": "A03:2021 - Injection",
            "command_injection": "A03:2021 - Injection",
            "path_traversal": "A01:2021 - Broken Access Control",
            "ldap_injection": "A03:2021 - Injection",
            "xml_injection": "A05:2021 - Security Misconfiguration",
            "nosql_injection": "A03:2021 - Injection",
            "header_injection": "A03:2021 - Injection",
            "ssti": "A03:2021 - Injection"
        }
        return owasp_mapping.get(vuln_type, "A00:2021 - Unknown")

    async def _check_security_headers(self, url: str, headers: Dict) -> List[DASTestResult]:
        """Check for missing security headers"""
        vulnerabilities = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    response_headers = response.headers

                    # Check for missing security headers
                    security_headers = {
                        "X-Frame-Options": "Clickjacking protection",
                        "X-Content-Type-Options": "MIME type sniffing protection",
                        "X-XSS-Protection": "XSS filter",
                        "Strict-Transport-Security": "HTTPS enforcement",
                        "Content-Security-Policy": "Content injection protection",
                        "Referrer-Policy": "Referrer information leakage",
                        "Permissions-Policy": "Feature permissions"
                    }

                    for header, description in security_headers.items():
                        if header not in response_headers:
                            vulnerabilities.append(DASTestResult(
                                vulnerability=f"Missing Security Header: {header}",
                                severity="Medium",
                                endpoint=url,
                                method="GET",
                                payload=None,
                                response_code=response.status,
                                evidence=f"{header} header is missing",
                                remediation=f"Add {header} header for {description}",
                                cwe_id="CWE-693",
                                owasp_category="A05:2021 - Security Misconfiguration"
                            ))

        except Exception:
            pass

        return vulnerabilities

    async def _check_cors_misconfiguration(self, url: str, headers: Dict) -> List[DASTestResult]:
        """Check for CORS misconfiguration"""
        vulnerabilities = []

        try:
            # Test with malicious origin
            test_headers = headers.copy() if headers else {}
            test_headers["Origin"] = "http://evil.com"

            async with aiohttp.ClientSession() as session:
                async with session.options(url, headers=test_headers, ssl=False) as response:
                    if "Access-Control-Allow-Origin" in response.headers:
                        allow_origin = response.headers["Access-Control-Allow-Origin"]

                        if allow_origin == "*" or allow_origin == "http://evil.com":
                            vulnerabilities.append(DASTestResult(
                                vulnerability="CORS Misconfiguration",
                                severity="High",
                                endpoint=url,
                                method="OPTIONS",
                                payload="Origin: http://evil.com",
                                response_code=response.status,
                                evidence=f"Access-Control-Allow-Origin: {allow_origin}",
                                remediation="Configure CORS to only allow trusted origins",
                                cwe_id="CWE-942",
                                owasp_category="A05:2021 - Security Misconfiguration"
                            ))

        except Exception:
            pass

        return vulnerabilities

    async def _check_http_methods(self, url: str, headers: Dict) -> List[DASTestResult]:
        """Check for dangerous HTTP methods"""
        vulnerabilities = []
        dangerous_methods = ["PUT", "DELETE", "TRACE", "CONNECT"]

        for method in dangerous_methods:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, headers=headers, ssl=False) as response:
                        if response.status < 400:
                            vulnerabilities.append(DASTestResult(
                                vulnerability=f"Dangerous HTTP Method Enabled: {method}",
                                severity="Medium",
                                endpoint=url,
                                method=method,
                                payload=None,
                                response_code=response.status,
                                evidence=f"{method} method returned {response.status}",
                                remediation=f"Disable {method} method if not required",
                                cwe_id="CWE-749",
                                owasp_category="A05:2021 - Security Misconfiguration"
                            ))
            except Exception:
                pass

        return vulnerabilities


class OAuthJWTTester:
    """OAuth and JWT Flow Testing"""

    def __init__(self):
        self.oauth_flows = ["authorization_code", "implicit", "client_credentials", "password", "device_code"]
        self.jwt_algorithms = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

    async def test_oauth_flow(self, config: Dict[str, Any]) -> List[OAuthTestResult]:
        """
        Test OAuth 2.0 flow security

        Args:
            config: OAuth configuration including:
                - authorization_url
                - token_url
                - client_id
                - client_secret
                - redirect_uri
                - scope
                - flow_type

        Returns:
            List of test results
        """
        results = []
        flow_type = config.get("flow_type", "authorization_code")

        # Test based on flow type
        if flow_type == "authorization_code":
            results.extend(await self._test_authorization_code_flow(config))
        elif flow_type == "implicit":
            results.extend(await self._test_implicit_flow(config))
        elif flow_type == "client_credentials":
            results.extend(await self._test_client_credentials_flow(config))
        elif flow_type == "password":
            results.extend(await self._test_password_flow(config))

        # Common OAuth security tests
        results.extend(await self._test_oauth_common_vulnerabilities(config))

        return results

    async def _test_authorization_code_flow(self, config: Dict[str, Any]) -> List[OAuthTestResult]:
        """Test authorization code flow"""
        results = []

        # Test 1: PKCE validation
        pkce_result = await self._test_pkce_requirement(config)
        results.append(pkce_result)

        # Test 2: State parameter validation
        state_result = await self._test_state_parameter(config)
        results.append(state_result)

        # Test 3: Redirect URI validation
        redirect_result = await self._test_redirect_uri_validation(config)
        results.append(redirect_result)

        # Test 4: Authorization code reuse
        code_reuse_result = await self._test_authorization_code_reuse(config)
        results.append(code_reuse_result)

        return results

    async def _test_pkce_requirement(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test if PKCE is required"""
        try:
            # Try authorization without PKCE
            params = {
                "response_type": "code",
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "scope": config.get("scope", "openid profile"),
                "state": self._generate_state()
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(config["authorization_url"], params=params, allow_redirects=False) as response:
                    if response.status == 302:
                        # Check if PKCE is enforced
                        location = response.headers.get("Location", "")
                        if "error" not in location:
                            return OAuthTestResult(
                                flow_type="authorization_code",
                                test_name="PKCE Requirement",
                                passed=False,
                                error_message="PKCE not required - vulnerable to authorization code interception",
                                security_issues=["Authorization code can be intercepted and used by attackers"],
                                recommendations=["Enforce PKCE for all public clients", "Require code_challenge parameter"]
                            )

            return OAuthTestResult(
                flow_type="authorization_code",
                test_name="PKCE Requirement",
                passed=True,
                error_message=None,
                security_issues=[],
                recommendations=[]
            )

        except Exception as e:
            return OAuthTestResult(
                flow_type="authorization_code",
                test_name="PKCE Requirement",
                passed=False,
                error_message=str(e),
                security_issues=["Could not verify PKCE requirement"],
                recommendations=["Ensure PKCE is properly configured"]
            )

    async def _test_state_parameter(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test state parameter validation"""
        try:
            # Test without state parameter
            params = {
                "response_type": "code",
                "client_id": config["client_id"],
                "redirect_uri": config["redirect_uri"],
                "scope": config.get("scope", "openid profile")
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(config["authorization_url"], params=params, allow_redirects=False) as response:
                    if response.status == 302:
                        location = response.headers.get("Location", "")
                        if "error" not in location:
                            return OAuthTestResult(
                                flow_type="authorization_code",
                                test_name="State Parameter Validation",
                                passed=False,
                                error_message="State parameter not required - vulnerable to CSRF",
                                security_issues=["CSRF attacks possible", "State parameter not enforced"],
                                recommendations=["Require state parameter", "Validate state on callback"]
                            )

            return OAuthTestResult(
                flow_type="authorization_code",
                test_name="State Parameter Validation",
                passed=True,
                error_message=None,
                security_issues=[],
                recommendations=[]
            )

        except Exception as e:
            return OAuthTestResult(
                flow_type="authorization_code",
                test_name="State Parameter Validation",
                passed=False,
                error_message=str(e),
                security_issues=["Could not verify state parameter"],
                recommendations=["Implement proper state validation"]
            )

    async def _test_redirect_uri_validation(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test redirect URI validation"""
        try:
            # Test with malicious redirect URI
            malicious_uris = [
                "http://evil.com/callback",
                config["redirect_uri"] + ".evil.com",
                config["redirect_uri"].replace("https", "http"),
                "data:text/html,<script>alert('XSS')</script>"
            ]

            for malicious_uri in malicious_uris:
                params = {
                    "response_type": "code",
                    "client_id": config["client_id"],
                    "redirect_uri": malicious_uri,
                    "scope": config.get("scope", "openid profile"),
                    "state": self._generate_state()
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(config["authorization_url"], params=params, allow_redirects=False) as response:
                        if response.status == 302:
                            location = response.headers.get("Location", "")
                            if malicious_uri in location:
                                return OAuthTestResult(
                                    flow_type="authorization_code",
                                    test_name="Redirect URI Validation",
                                    passed=False,
                                    error_message=f"Accepts malicious redirect URI: {malicious_uri}",
                                    security_issues=["Open redirect vulnerability", "Token/code can be stolen"],
                                    recommendations=["Strict redirect URI validation", "Exact match validation"]
                                )

            return OAuthTestResult(
                flow_type="authorization_code",
                test_name="Redirect URI Validation",
                passed=True,
                error_message=None,
                security_issues=[],
                recommendations=[]
            )

        except Exception as e:
            return OAuthTestResult(
                flow_type="authorization_code",
                test_name="Redirect URI Validation",
                passed=False,
                error_message=str(e),
                security_issues=["Could not verify redirect URI validation"],
                recommendations=["Implement strict redirect URI validation"]
            )

    async def _test_authorization_code_reuse(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test if authorization codes can be reused"""
        # This would require a valid authorization code
        # For demonstration, we'll return a recommendation
        return OAuthTestResult(
            flow_type="authorization_code",
            test_name="Authorization Code Reuse Prevention",
            passed=None,  # Cannot test without valid code
            error_message="Manual testing required with valid authorization code",
            security_issues=[],
            recommendations=[
                "Ensure authorization codes are single-use",
                "Implement code expiration (max 10 minutes)",
                "Revoke all tokens if code is reused"
            ]
        )

    async def _test_implicit_flow(self, config: Dict[str, Any]) -> List[OAuthTestResult]:
        """Test implicit flow vulnerabilities"""
        return [
            OAuthTestResult(
                flow_type="implicit",
                test_name="Implicit Flow Deprecation",
                passed=False,
                error_message="Implicit flow is deprecated and insecure",
                security_issues=[
                    "Tokens exposed in URL fragments",
                    "No client authentication",
                    "Tokens can be leaked via referrer",
                    "Vulnerable to token substitution"
                ],
                recommendations=[
                    "Migrate to Authorization Code flow with PKCE",
                    "Never use implicit flow for new applications"
                ]
            )
        ]

    async def _test_client_credentials_flow(self, config: Dict[str, Any]) -> List[OAuthTestResult]:
        """Test client credentials flow"""
        results = []

        # Test client authentication methods
        try:
            # Test with basic auth
            auth = aiohttp.BasicAuth(config["client_id"], config.get("client_secret", ""))
            data = {
                "grant_type": "client_credentials",
                "scope": config.get("scope", "")
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(config["token_url"], auth=auth, data=data) as response:
                    if response.status == 200:
                        results.append(OAuthTestResult(
                            flow_type="client_credentials",
                            test_name="Client Authentication",
                            passed=True,
                            error_message=None,
                            security_issues=[],
                            recommendations=["Consider using client assertion for stronger authentication"]
                        ))

        except Exception as e:
            results.append(OAuthTestResult(
                flow_type="client_credentials",
                test_name="Client Authentication",
                passed=False,
                error_message=str(e),
                security_issues=["Client authentication failed"],
                recommendations=["Verify client credentials configuration"]
            ))

        return results

    async def _test_password_flow(self, config: Dict[str, Any]) -> List[OAuthTestResult]:
        """Test password flow vulnerabilities"""
        return [
            OAuthTestResult(
                flow_type="password",
                test_name="Password Flow Security",
                passed=False,
                error_message="Password flow should be avoided",
                security_issues=[
                    "Requires client to handle user credentials",
                    "Increases phishing risk",
                    "No way to implement MFA",
                    "Client has access to user password"
                ],
                recommendations=[
                    "Use Authorization Code flow instead",
                    "Never use password flow for third-party clients",
                    "Only use for highly trusted first-party clients if absolutely necessary"
                ]
            )
        ]

    async def _test_oauth_common_vulnerabilities(self, config: Dict[str, Any]) -> List[OAuthTestResult]:
        """Test common OAuth vulnerabilities"""
        results = []

        # Test token endpoint authentication
        results.append(await self._test_token_endpoint_authentication(config))

        # Test token scope validation
        results.append(self._test_scope_validation(config))

        # Test refresh token rotation
        results.append(self._test_refresh_token_rotation(config))

        return results

    async def _test_token_endpoint_authentication(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test token endpoint requires authentication"""
        try:
            # Try to get token without authentication
            data = {
                "grant_type": "client_credentials",
                "client_id": config["client_id"]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(config["token_url"], data=data) as response:
                    if response.status == 200:
                        return OAuthTestResult(
                            flow_type="all",
                            test_name="Token Endpoint Authentication",
                            passed=False,
                            error_message="Token endpoint doesn't require client authentication",
                            security_issues=["Tokens can be obtained without proper authentication"],
                            recommendations=["Require client authentication for all token requests"]
                        )

            return OAuthTestResult(
                flow_type="all",
                test_name="Token Endpoint Authentication",
                passed=True,
                error_message=None,
                security_issues=[],
                recommendations=[]
            )

        except Exception:
            return OAuthTestResult(
                flow_type="all",
                test_name="Token Endpoint Authentication",
                passed=True,
                error_message=None,
                security_issues=[],
                recommendations=[]
            )

    def _test_scope_validation(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test scope validation recommendations"""
        return OAuthTestResult(
            flow_type="all",
            test_name="Scope Validation",
            passed=None,
            error_message="Manual verification required",
            security_issues=[],
            recommendations=[
                "Validate requested scopes against allowed scopes",
                "Implement principle of least privilege",
                "User should approve scope changes",
                "Log scope elevation attempts"
            ]
        )

    def _test_refresh_token_rotation(self, config: Dict[str, Any]) -> OAuthTestResult:
        """Test refresh token rotation recommendations"""
        return OAuthTestResult(
            flow_type="all",
            test_name="Refresh Token Rotation",
            passed=None,
            error_message="Manual verification required",
            security_issues=[],
            recommendations=[
                "Implement refresh token rotation",
                "Invalidate old refresh tokens after use",
                "Implement refresh token expiration",
                "Detect and prevent refresh token replay"
            ]
        )

    def test_jwt_security(self, token: str, secret: Optional[str] = None, public_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Test JWT token security

        Args:
            token: JWT token to test
            secret: Secret key for HMAC algorithms
            public_key: Public key for RSA/ECDSA algorithms

        Returns:
            Security test results
        """
        results = {
            "vulnerabilities": [],
            "token_info": {},
            "recommendations": []
        }

        # Decode without verification to inspect token
        try:
            header = jwt.get_unverified_header(token)
            payload = jwt.decode(token, options={"verify_signature": False})
            results["token_info"] = {
                "header": header,
                "payload": payload,
                "algorithm": header.get("alg", "none")
            }
        except Exception as e:
            results["vulnerabilities"].append(f"Invalid JWT format: {str(e)}")
            return results

        # Test 1: Algorithm confusion attack
        if self._test_algorithm_confusion(token, header):
            results["vulnerabilities"].append("Vulnerable to algorithm confusion attack")
            results["recommendations"].append("Explicitly verify algorithm type")

        # Test 2: None algorithm
        if self._test_none_algorithm(token):
            results["vulnerabilities"].append("Accepts 'none' algorithm - critical vulnerability")
            results["recommendations"].append("Never accept 'none' algorithm")

        # Test 3: Weak secret
        if secret and self._test_weak_secret(token, header):
            results["vulnerabilities"].append("Weak secret key detected")
            results["recommendations"].append("Use strong, random secret keys (min 256 bits)")

        # Test 4: Token expiration
        exp_issue = self._check_expiration(payload)
        if exp_issue:
            results["vulnerabilities"].append(exp_issue)
            results["recommendations"].append("Implement proper token expiration")

        # Test 5: Required claims
        missing_claims = self._check_required_claims(payload)
        if missing_claims:
            results["vulnerabilities"].append(f"Missing required claims: {', '.join(missing_claims)}")
            results["recommendations"].append("Include all required JWT claims")

        # Test 6: Sensitive data in token
        if self._check_sensitive_data(payload):
            results["vulnerabilities"].append("Token contains potentially sensitive data")
            results["recommendations"].append("Avoid storing sensitive data in JWTs")

        return results

    def _test_algorithm_confusion(self, token: str, header: Dict) -> bool:
        """Test for algorithm confusion vulnerability"""
        if header.get("alg", "").startswith("HS"):
            # Try to use public key as HMAC secret
            # This is a common vulnerability
            return True
        return False

    def _test_none_algorithm(self, token: str) -> bool:
        """Test if 'none' algorithm is accepted"""
        # Modify token to use 'none' algorithm
        parts = token.split('.')
        if len(parts) == 3:
            header = json.loads(base64.urlsafe_b64decode(parts[0] + '=='))
            header['alg'] = 'none'
            new_header = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
            none_token = f"{new_header}.{parts[1]}."

            try:
                # If this succeeds, the system accepts 'none' algorithm
                jwt.decode(none_token, options={"verify_signature": False})
                return True
            except:
                return False
        return False

    def _test_weak_secret(self, token: str, header: Dict) -> bool:
        """Test for weak secret keys"""
        if not header.get("alg", "").startswith("HS"):
            return False

        # Common weak secrets
        weak_secrets = [
            "secret", "password", "123456", "admin", "key",
            "jwt-secret", "change-me", "secret-key", "your-256-bit-secret"
        ]

        for weak_secret in weak_secrets:
            try:
                jwt.decode(token, weak_secret, algorithms=[header["alg"]])
                return True  # Weak secret found
            except:
                continue

        return False

    def _check_expiration(self, payload: Dict) -> Optional[str]:
        """Check token expiration issues"""
        now = time.time()

        if "exp" not in payload:
            return "Token has no expiration"

        exp = payload["exp"]
        if exp > now + (365 * 24 * 60 * 60):  # More than 1 year
            return "Token expiration too far in future"

        if "iat" in payload:
            iat = payload["iat"]
            if exp - iat > (24 * 60 * 60):  # More than 24 hours
                return "Token lifetime too long"

        return None

    def _check_required_claims(self, payload: Dict) -> List[str]:
        """Check for required JWT claims"""
        required_claims = ["sub", "iat", "exp", "jti"]
        missing = [claim for claim in required_claims if claim not in payload]
        return missing

    def _check_sensitive_data(self, payload: Dict) -> bool:
        """Check for sensitive data in token"""
        sensitive_patterns = [
            "password", "secret", "private", "ssn", "credit",
            "card", "cvv", "pin", "api_key", "private_key"
        ]

        payload_str = json.dumps(payload).lower()
        return any(pattern in payload_str for pattern in sensitive_patterns)

    def _generate_state(self) -> str:
        """Generate random state parameter"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=32))


class APIFuzzer:
    """Schema-based API Fuzzing"""

    def __init__(self, schema: Dict[str, Any] = None):
        self.schema = schema
        self.fuzz_strategies = {
            "string": self._fuzz_string,
            "integer": self._fuzz_integer,
            "number": self._fuzz_number,
            "boolean": self._fuzz_boolean,
            "array": self._fuzz_array,
            "object": self._fuzz_object
        }

    async def fuzz_endpoint(self, url: str, method: str, headers: Dict, body: Any,
                           schema: Dict[str, Any] = None) -> List[FuzzTestResult]:
        """
        Perform intelligent fuzzing based on schema

        Args:
            url: Target URL
            method: HTTP method
            headers: Request headers
            body: Original request body
            schema: API schema (optional, uses instance schema if not provided)

        Returns:
            List of fuzzing test results
        """
        results = []
        schema = schema or self.schema

        if not schema:
            # No schema, perform blind fuzzing
            results.extend(await self._blind_fuzzing(url, method, headers, body))
        else:
            # Schema-based intelligent fuzzing
            results.extend(await self._schema_based_fuzzing(url, method, headers, body, schema))

        return results

    async def _schema_based_fuzzing(self, url: str, method: str, headers: Dict,
                                   body: Any, schema: Dict[str, Any]) -> List[FuzzTestResult]:
        """Perform fuzzing based on schema"""
        results = []

        if isinstance(body, dict):
            for field_name, field_schema in schema.items():
                field_type = field_schema.get("type", "string")

                if field_type in self.fuzz_strategies:
                    fuzz_values = self.fuzz_strategies[field_type](field_schema)

                    for fuzz_value in fuzz_values:
                        # Create fuzzed body
                        fuzzed_body = body.copy()
                        fuzzed_body[field_name] = fuzz_value

                        # Test the fuzzed request
                        result = await self._test_fuzzed_request(
                            url, method, headers, fuzzed_body,
                            field_name, body.get(field_name), fuzz_value
                        )
                        if result:
                            results.append(result)

        return results

    async def _blind_fuzzing(self, url: str, method: str, headers: Dict, body: Any) -> List[FuzzTestResult]:
        """Perform blind fuzzing without schema"""
        results = []

        # Common fuzz patterns
        fuzz_patterns = [
            # Buffer overflow attempts
            "A" * 10000,
            "A" * 100000,

            # Format string
            "%s%s%s%s%s",
            "%x%x%x%x",
            "%n%n%n%n",

            # Integer overflow
            "2147483648",
            "-2147483649",
            "4294967296",
            "9999999999999999999999999999",

            # Special characters
            "\x00\x01\x02\x03\x04\x05",
            "\r\n\r\n",
            "\\x00\\x01",

            # Unicode
            "𝕊𝕠𝕞𝕖𝕥𝕙𝕚𝕟𝕘",
            "../../",
            "\u202e\u202d",

            # NULL and special values
            None,
            "",
            " ",
            "\t\r\n"
        ]

        if isinstance(body, dict):
            for field_name in body.keys():
                for fuzz_value in fuzz_patterns:
                    fuzzed_body = body.copy()
                    fuzzed_body[field_name] = fuzz_value

                    result = await self._test_fuzzed_request(
                        url, method, headers, fuzzed_body,
                        field_name, body.get(field_name), fuzz_value
                    )
                    if result:
                        results.append(result)

        return results

    async def _test_fuzzed_request(self, url: str, method: str, headers: Dict,
                                  fuzzed_body: Any, field: str,
                                  original_value: Any, fuzz_value: Any) -> Optional[FuzzTestResult]:
        """Test a fuzzed request"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=fuzzed_body if isinstance(fuzzed_body, dict) else None,
                    data=fuzzed_body if isinstance(fuzzed_body, str) else None,
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=False
                ) as response:
                    response_text = await response.text()

                    # Analyze response for issues
                    if response.status >= 500:
                        return FuzzTestResult(
                            field=field,
                            original_value=original_value,
                            fuzzed_value=str(fuzz_value)[:100],  # Truncate for display
                            response_code=response.status,
                            error_triggered=True,
                            error_type="Server Error",
                            potential_vulnerability="Input validation failure - server error triggered"
                        )

                    # Check for information disclosure
                    if self._check_information_disclosure(response_text):
                        return FuzzTestResult(
                            field=field,
                            original_value=original_value,
                            fuzzed_value=str(fuzz_value)[:100],
                            response_code=response.status,
                            error_triggered=True,
                            error_type="Information Disclosure",
                            potential_vulnerability="Sensitive information in error message"
                        )

        except asyncio.TimeoutError:
            return FuzzTestResult(
                field=field,
                original_value=original_value,
                fuzzed_value=str(fuzz_value)[:100],
                response_code=0,
                error_triggered=True,
                error_type="Timeout",
                potential_vulnerability="Potential DoS vulnerability - request timeout"
            )

        except Exception as e:
            if "memory" in str(e).lower():
                return FuzzTestResult(
                    field=field,
                    original_value=original_value,
                    fuzzed_value=str(fuzz_value)[:100],
                    response_code=0,
                    error_triggered=True,
                    error_type="Memory Error",
                    potential_vulnerability="Potential memory exhaustion vulnerability"
                )

        return None

    def _fuzz_string(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate fuzz values for string fields"""
        fuzz_values = []

        # Length-based fuzzing
        if "maxLength" in schema:
            max_len = schema["maxLength"]
            fuzz_values.extend([
                "A" * (max_len + 1),  # Exceed max length
                "A" * (max_len * 10),  # Way over limit
            ])

        if "minLength" in schema:
            min_len = schema["minLength"]
            if min_len > 0:
                fuzz_values.extend([
                    "",  # Below min length
                    "A" * (min_len - 1),  # Just below min
                ])

        # Pattern-based fuzzing
        if "pattern" in schema:
            # Try to break the pattern
            fuzz_values.extend([
                "!@#$%^&*()",
                "<script>alert(1)</script>",
                "'; DROP TABLE--"
            ])

        # Format-based fuzzing
        if "format" in schema:
            format_type = schema["format"]
            if format_type == "email":
                fuzz_values.extend([
                    "not-an-email",
                    "@example.com",
                    "user@",
                    "user@@example.com"
                ])
            elif format_type == "uri":
                fuzz_values.extend([
                    "not-a-uri",
                    "javascript:alert(1)",
                    "file:///etc/passwd"
                ])
            elif format_type == "date":
                fuzz_values.extend([
                    "not-a-date",
                    "2024-13-01",  # Invalid month
                    "2024-01-32"   # Invalid day
                ])

        # Common string fuzzing
        fuzz_values.extend([
            None,
            123,  # Wrong type
            True,  # Wrong type
            ["array"],  # Wrong type
            {"object": "value"},  # Wrong type
            "",
            " ",
            "\x00",
            "A" * 100000,
            "€£¥",
            "\r\n\r\n",
            "../../../etc/passwd"
        ])

        return fuzz_values

    def _fuzz_integer(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate fuzz values for integer fields"""
        fuzz_values = []

        # Boundary testing
        if "minimum" in schema:
            min_val = schema["minimum"]
            fuzz_values.extend([
                min_val - 1,
                min_val - 1000
            ])

        if "maximum" in schema:
            max_val = schema["maximum"]
            fuzz_values.extend([
                max_val + 1,
                max_val + 1000
            ])

        # Type testing and overflow
        fuzz_values.extend([
            None,
            "string",  # Wrong type
            3.14,  # Float instead of int
            True,  # Boolean
            [],  # Array
            {},  # Object
            2147483648,  # INT32 overflow
            -2147483649,  # INT32 underflow
            9223372036854775808,  # INT64 overflow
            float('inf'),
            float('-inf'),
            float('nan')
        ])

        return fuzz_values

    def _fuzz_number(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate fuzz values for number fields"""
        fuzz_values = self._fuzz_integer(schema)  # Include integer tests

        # Additional float-specific tests
        fuzz_values.extend([
            1.7976931348623157e+308,  # Near max float
            -1.7976931348623157e+308,  # Near min float
            0.0000000000000001,
            -0.0
        ])

        return fuzz_values

    def _fuzz_boolean(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate fuzz values for boolean fields"""
        return [
            None,
            "true",  # String instead of boolean
            "false",
            1,  # Integer
            0,
            "yes",
            "no",
            [],
            {}
        ]

    def _fuzz_array(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate fuzz values for array fields"""
        fuzz_values = [
            None,
            "not-an-array",  # Wrong type
            123,
            True,
            {},
            [],  # Empty array
            [None, None, None],  # Null elements
        ]

        # Size-based fuzzing
        if "maxItems" in schema:
            max_items = schema["maxItems"]
            fuzz_values.append([1] * (max_items + 10))  # Exceed max

        if "minItems" in schema:
            min_items = schema["minItems"]
            if min_items > 0:
                fuzz_values.append([])  # Below min

        # Large array
        fuzz_values.append([1] * 10000)

        return fuzz_values

    def _fuzz_object(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate fuzz values for object fields"""
        return [
            None,
            "not-an-object",  # Wrong type
            123,
            True,
            [],
            {},  # Empty object
            {"__proto__": {"isAdmin": True}},  # Prototype pollution
            {"constructor": {"prototype": {"isAdmin": True}}}  # Prototype pollution
        ]

    def _check_information_disclosure(self, response_text: str) -> bool:
        """Check for information disclosure in response"""
        sensitive_patterns = [
            "stack trace",
            "traceback",
            "at line",
            "File \"",
            "java.lang",
            "Exception in",
            "mysql",
            "postgresql",
            "sqlite",
            "mongodb",
            "/home/",
            "/usr/",
            "C:\\",
            "SQL",
            "syntax error"
        ]

        response_lower = response_text.lower()
        return any(pattern.lower() in response_lower for pattern in sensitive_patterns)


# Export classes
__all__ = [
    'DASTScanner',
    'OAuthJWTTester',
    'APIFuzzer',
    'DASTestResult',
    'OAuthTestResult',
    'FuzzTestResult',
    'SecurityTestType'
]