# -*- coding: utf-8 -*-
"""
Security Scanner and Compliance Module
Analyzes generated tests for security coverage and compliance
"""

import re
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Compliance standards"""
    OWASP_TOP_10 = "owasp_top_10"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CWE_TOP_25 = "cwe_top_25"


class SecurityCategory(Enum):
    """Security test categories"""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    DESERIALIZATION = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    SSRF = "server_side_request_forgery"
    RATE_LIMITING = "rate_limiting"
    CSRF = "cross_site_request_forgery"
    FILE_UPLOAD = "file_upload"
    ENCRYPTION = "encryption"


@dataclass
class SecurityIssue:
    """Identified security issue"""
    severity: str  # critical, high, medium, low
    category: SecurityCategory
    description: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Security scan results"""
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    issues: List[SecurityIssue]
    security_score: float  # 0-100
    covered_categories: Set[SecurityCategory]
    missing_categories: Set[SecurityCategory]
    recommendations: List[str]


@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    standard: ComplianceStandard
    is_compliant: bool
    compliance_score: float  # 0-100
    met_requirements: List[str]
    missing_requirements: List[str]
    recommendations: List[str]
    scan_date: datetime


class SecurityScanner:
    """Scans test code and APIs for security issues"""

    def __init__(self):
        self.security_patterns = self._load_security_patterns()
        self.compliance_requirements = self._load_compliance_requirements()

    def _load_security_patterns(self) -> Dict[SecurityCategory, List[Dict[str, Any]]]:
        """Load security detection patterns"""
        return {
            SecurityCategory.INJECTION: [
                {
                    "pattern": r"['\"].*(\bOR\b|\bAND\b).*['\"].*=.*['\"]",
                    "description": "SQL injection pattern detected",
                    "severity": "critical"
                },
                {
                    "pattern": r"['\"];.*DROP\s+TABLE",
                    "description": "SQL DROP TABLE injection",
                    "severity": "critical"
                },
                {
                    "pattern": r"\$\{.*\}",
                    "description": "Template injection risk",
                    "severity": "high"
                },
                {
                    "pattern": r"exec\(|eval\(|system\(",
                    "description": "Command injection risk",
                    "severity": "critical"
                }
            ],
            SecurityCategory.XSS: [
                {
                    "pattern": r"<script[^>]*>.*</script>",
                    "description": "XSS script tag pattern",
                    "severity": "high"
                },
                {
                    "pattern": r"javascript:",
                    "description": "JavaScript protocol handler",
                    "severity": "high"
                },
                {
                    "pattern": r"on\w+\s*=",
                    "description": "Event handler injection",
                    "severity": "medium"
                }
            ],
            SecurityCategory.SENSITIVE_DATA: [
                {
                    "pattern": r"(password|pwd|passwd|pass)\s*=\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded password detected",
                    "severity": "critical"
                },
                {
                    "pattern": r"(api[_-]?key|apikey)\s*=\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded API key detected",
                    "severity": "high"
                },
                {
                    "pattern": r"(secret|token)\s*=\s*['\"][^'\"]+['\"]",
                    "description": "Hardcoded secret/token detected",
                    "severity": "high"
                },
                {
                    "pattern": r"\b\d{13,19}\b",
                    "description": "Possible credit card number",
                    "severity": "high"
                },
                {
                    "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                    "description": "Possible SSN pattern",
                    "severity": "critical"
                }
            ],
            SecurityCategory.BROKEN_AUTH: [
                {
                    "pattern": r"auth.*=.*false|skip.*auth|bypass.*auth",
                    "description": "Authentication bypass detected",
                    "severity": "critical"
                },
                {
                    "pattern": r"Bearer\s+[A-Za-z0-9+/=]+",
                    "description": "Hardcoded bearer token",
                    "severity": "high"
                }
            ],
            SecurityCategory.ENCRYPTION: [
                {
                    "pattern": r"http://(?!localhost|127\.0\.0\.1)",
                    "description": "Unencrypted HTTP connection",
                    "severity": "medium"
                },
                {
                    "pattern": r"ssl.*=.*false|verify.*=.*false",
                    "description": "SSL verification disabled",
                    "severity": "high"
                }
            ]
        }

    def _load_compliance_requirements(self) -> Dict[ComplianceStandard, List[str]]:
        """Load compliance requirements"""
        return {
            ComplianceStandard.OWASP_TOP_10: [
                "injection_testing",
                "broken_authentication_testing",
                "sensitive_data_exposure_testing",
                "xxe_testing",
                "broken_access_control_testing",
                "security_misconfiguration_testing",
                "xss_testing",
                "insecure_deserialization_testing",
                "using_components_with_known_vulnerabilities",
                "insufficient_logging_monitoring"
            ],
            ComplianceStandard.PCI_DSS: [
                "encryption_in_transit",
                "encryption_at_rest",
                "access_control",
                "regular_security_testing",
                "secure_coding_practices",
                "vulnerability_management"
            ],
            ComplianceStandard.GDPR: [
                "data_minimization",
                "consent_management",
                "right_to_erasure",
                "data_portability",
                "privacy_by_design",
                "data_protection_impact_assessment"
            ],
            ComplianceStandard.HIPAA: [
                "access_controls",
                "audit_controls",
                "integrity_controls",
                "transmission_security",
                "encryption_requirements"
            ]
        }

    async def scan_test_code(self, test_code: str) -> SecurityScanResult:
        """
        Scan test code for security issues

        Args:
            test_code: Generated test code

        Returns:
            SecurityScanResult with identified issues
        """
        issues = []
        covered_categories = set()
        lines = test_code.split('\n')

        # Scan for security patterns
        for category, patterns in self.security_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                matches = []

                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append((i + 1, line.strip()))

                if matches:
                    for line_num, code_snippet in matches:
                        issue = SecurityIssue(
                            severity=pattern_info["severity"],
                            category=category,
                            description=pattern_info["description"],
                            line_number=line_num,
                            code_snippet=code_snippet[:100],
                            recommendation=self._get_recommendation(category)
                        )
                        issues.append(issue)

        # Check for security test coverage
        covered_categories = self._detect_security_test_coverage(test_code)
        all_categories = set(SecurityCategory)
        missing_categories = all_categories - covered_categories

        # Calculate severity counts
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        medium_issues = len([i for i in issues if i.severity == "medium"])
        low_issues = len([i for i in issues if i.severity == "low"])

        # Calculate security score
        security_score = self._calculate_security_score(
            issues,
            covered_categories,
            len(all_categories)
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(issues, missing_categories)

        return SecurityScanResult(
            total_issues=len(issues),
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            issues=issues,
            security_score=security_score,
            covered_categories=covered_categories,
            missing_categories=missing_categories,
            recommendations=recommendations
        )

    def _detect_security_test_coverage(self, test_code: str) -> Set[SecurityCategory]:
        """Detect which security categories are covered by tests"""
        covered = set()

        # SQL Injection tests
        if any(pattern in test_code for pattern in ["SQL", "injection", "DROP TABLE", "' OR '"]):
            covered.add(SecurityCategory.INJECTION)

        # XSS tests
        if any(pattern in test_code for pattern in ["<script>", "javascript:", "onerror", "XSS"]):
            covered.add(SecurityCategory.XSS)

        # Authentication tests
        if any(pattern in test_code for pattern in ["401", "403", "unauthorized", "authentication"]):
            covered.add(SecurityCategory.BROKEN_AUTH)

        # Rate limiting tests
        if any(pattern in test_code for pattern in ["429", "rate limit", "throttle"]):
            covered.add(SecurityCategory.RATE_LIMITING)

        # CSRF tests
        if any(pattern in test_code for pattern in ["csrf", "CSRF", "X-CSRF-Token"]):
            covered.add(SecurityCategory.CSRF)

        # Encryption tests
        if any(pattern in test_code for pattern in ["https", "TLS", "SSL", "encryption"]):
            covered.add(SecurityCategory.ENCRYPTION)

        return covered

    def _calculate_security_score(
        self,
        issues: List[SecurityIssue],
        covered_categories: Set[SecurityCategory],
        total_categories: int
    ) -> float:
        """Calculate overall security score"""
        base_score = 100.0

        # Deduct points for issues
        for issue in issues:
            if issue.severity == "critical":
                base_score -= 15
            elif issue.severity == "high":
                base_score -= 10
            elif issue.severity == "medium":
                base_score -= 5
            elif issue.severity == "low":
                base_score -= 2

        # Coverage bonus
        coverage_ratio = len(covered_categories) / total_categories if total_categories > 0 else 0
        coverage_bonus = coverage_ratio * 20

        final_score = max(0, min(100, base_score + coverage_bonus))
        return round(final_score, 2)

    def _get_recommendation(self, category: SecurityCategory) -> str:
        """Get recommendation for security issue"""
        recommendations = {
            SecurityCategory.INJECTION: "Use parameterized queries and input validation",
            SecurityCategory.XSS: "Sanitize all user input and use Content Security Policy",
            SecurityCategory.SENSITIVE_DATA: "Use environment variables for sensitive data",
            SecurityCategory.BROKEN_AUTH: "Implement proper authentication and session management",
            SecurityCategory.ENCRYPTION: "Always use HTTPS and enable SSL verification",
            SecurityCategory.RATE_LIMITING: "Implement rate limiting to prevent abuse",
            SecurityCategory.CSRF: "Use CSRF tokens for state-changing operations"
        }
        return recommendations.get(category, "Review and fix security issue")

    def _generate_recommendations(
        self,
        issues: List[SecurityIssue],
        missing_categories: Set[SecurityCategory]
    ) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        # Critical issues recommendations
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append(f"URGENT: Fix {len(critical_issues)} critical security issues immediately")

        # Missing coverage recommendations
        if missing_categories:
            missing_list = [cat.value.replace('_', ' ').title() for cat in list(missing_categories)[:3]]
            recommendations.append(f"Add tests for: {', '.join(missing_list)}")

        # Specific recommendations based on issues
        if any(i.category == SecurityCategory.SENSITIVE_DATA for i in issues):
            recommendations.append("Remove all hardcoded credentials and use environment variables")

        if any(i.category == SecurityCategory.INJECTION for i in issues):
            recommendations.append("Implement input validation and parameterized queries")

        if not any(i.category == SecurityCategory.RATE_LIMITING for i in issues):
            recommendations.append("Consider adding rate limiting tests")

        return recommendations

    async def check_compliance(
        self,
        test_code: str,
        standard: ComplianceStandard
    ) -> ComplianceReport:
        """
        Check compliance with specific standard

        Args:
            test_code: Generated test code
            standard: Compliance standard to check

        Returns:
            ComplianceReport with assessment
        """
        requirements = self.compliance_requirements.get(standard, [])
        met_requirements = []
        missing_requirements = []

        # Check each requirement
        for requirement in requirements:
            if self._check_requirement(test_code, requirement):
                met_requirements.append(requirement)
            else:
                missing_requirements.append(requirement)

        # Calculate compliance score
        compliance_score = (len(met_requirements) / len(requirements) * 100) if requirements else 0
        is_compliant = compliance_score >= 80  # 80% threshold for compliance

        # Generate recommendations
        recommendations = []
        if not is_compliant:
            recommendations.append(f"Implement missing requirements to achieve {standard.value} compliance")
            for req in missing_requirements[:3]:  # Top 3 missing
                recommendations.append(f"Add tests for: {req.replace('_', ' ')}")

        return ComplianceReport(
            standard=standard,
            is_compliant=is_compliant,
            compliance_score=round(compliance_score, 2),
            met_requirements=met_requirements,
            missing_requirements=missing_requirements,
            recommendations=recommendations,
            scan_date=datetime.now()
        )

    def _check_requirement(self, test_code: str, requirement: str) -> bool:
        """Check if a specific compliance requirement is met"""
        requirement_patterns = {
            "injection_testing": ["injection", "SQL", "' OR '", "DROP TABLE"],
            "broken_authentication_testing": ["401", "403", "authentication", "unauthorized"],
            "sensitive_data_exposure_testing": ["encryption", "https", "sensitive"],
            "xss_testing": ["<script>", "XSS", "javascript:", "sanitize"],
            "broken_access_control_testing": ["403", "access control", "permission", "role"],
            "security_misconfiguration_testing": ["security headers", "configuration"],
            "encryption_in_transit": ["https", "TLS", "SSL"],
            "encryption_at_rest": ["encrypt", "AES", "RSA"],
            "access_control": ["authentication", "authorization", "role-based"],
            "audit_controls": ["logging", "audit", "monitor"]
        }

        patterns = requirement_patterns.get(requirement, [requirement])
        return any(pattern.lower() in test_code.lower() for pattern in patterns)


class QualityAnalyzer:
    """Analyzes test quality and provides metrics"""

    def analyze_test_quality(self, test_code: str, framework: str) -> Dict[str, Any]:
        """
        Analyze test code quality

        Args:
            test_code: Generated test code
            framework: Test framework

        Returns:
            Quality metrics and analysis
        """
        metrics = {
            "total_lines": len(test_code.split('\n')),
            "test_count": self._count_tests(test_code, framework),
            "assertion_count": self._count_assertions(test_code, framework),
            "coverage_types": self._identify_coverage_types(test_code),
            "code_duplication": self._check_duplication(test_code),
            "maintainability_index": self._calculate_maintainability(test_code),
            "complexity_score": self._calculate_complexity(test_code),
            "best_practices": self._check_best_practices(test_code, framework),
            "quality_score": 0.0  # Will be calculated
        }

        # Calculate overall quality score
        metrics["quality_score"] = self._calculate_quality_score(metrics)

        return metrics

    def _count_tests(self, code: str, framework: str) -> int:
        """Count number of tests"""
        if framework in ["jest", "mocha"]:
            return len(re.findall(r'\bit\(|test\(', code))
        elif framework == "pytest":
            return len(re.findall(r'def test_', code))
        return 0

    def _count_assertions(self, code: str, framework: str) -> int:
        """Count number of assertions"""
        if framework in ["jest", "mocha"]:
            return len(re.findall(r'expect\(', code))
        elif framework == "pytest":
            return len(re.findall(r'assert ', code))
        return 0

    def _identify_coverage_types(self, code: str) -> List[str]:
        """Identify types of test coverage"""
        coverage_types = []

        if re.search(r'200|201|204', code):
            coverage_types.append("positive_tests")
        if re.search(r'400|404|422', code):
            coverage_types.append("negative_tests")
        if re.search(r'401|403', code):
            coverage_types.append("auth_tests")
        if re.search(r'500|502|503', code):
            coverage_types.append("error_tests")
        if re.search(r'injection|XSS|CSRF', code, re.IGNORECASE):
            coverage_types.append("security_tests")
        if re.search(r'performance|timeout|latency', code, re.IGNORECASE):
            coverage_types.append("performance_tests")

        return coverage_types

    def _check_duplication(self, code: str) -> float:
        """Check for code duplication"""
        lines = code.split('\n')
        unique_lines = set(lines)
        duplication_ratio = 1.0 - (len(unique_lines) / len(lines)) if lines else 0
        return round(duplication_ratio * 100, 2)

    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability index"""
        # Simplified maintainability calculation
        lines = len(code.split('\n'))
        complexity = code.count('if ') + code.count('for ') + code.count('while ')
        comments = len(re.findall(r'//|#|/\*', code))

        # Base score
        score = 100

        # Deduct for high complexity
        score -= min(complexity * 2, 30)

        # Deduct for very long files
        if lines > 500:
            score -= 20
        elif lines > 300:
            score -= 10

        # Bonus for comments
        comment_ratio = comments / lines if lines > 0 else 0
        if comment_ratio > 0.1:
            score += 10

        return max(0, min(100, score))

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        # Simplified complexity calculation
        complexity = 1  # Base complexity

        # Add complexity for control structures
        complexity += code.count('if ')
        complexity += code.count('else ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('case ')
        complexity += code.count('catch ')

        return complexity

    def _check_best_practices(self, code: str, framework: str) -> Dict[str, bool]:
        """Check for best practices"""
        practices = {
            "has_setup_teardown": bool(re.search(r'before|after|setup|teardown', code, re.IGNORECASE)),
            "has_error_handling": bool(re.search(r'try|catch|except', code)),
            "has_timeouts": bool(re.search(r'timeout', code, re.IGNORECASE)),
            "uses_constants": bool(re.search(r'const |let |var |BASE_URL|API_KEY', code)),
            "has_descriptive_names": self._check_descriptive_names(code),
            "has_comments": bool(re.search(r'//|#|/\*', code)),
            "no_hardcoded_values": not bool(re.search(r'localhost:\d+|127\.0\.0\.1', code)),
            "uses_async_await": bool(re.search(r'async|await', code)) if framework in ["jest", "mocha"] else True
        }
        return practices

    def _check_descriptive_names(self, code: str) -> bool:
        """Check if test names are descriptive"""
        test_names = re.findall(r'it\([\'"]([^\'"]+)', code)
        test_names.extend(re.findall(r'test\([\'"]([^\'"]+)', code))
        test_names.extend(re.findall(r'def (test_\w+)', code))

        if not test_names:
            return False

        # Check if names are descriptive (more than 3 words on average)
        avg_words = sum(len(name.split()) for name in test_names) / len(test_names)
        return avg_words >= 3

    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        score = 50.0  # Base score

        # Test coverage (20 points)
        test_count = metrics["test_count"]
        if test_count >= 20:
            score += 20
        elif test_count >= 10:
            score += 15
        elif test_count >= 5:
            score += 10
        elif test_count > 0:
            score += 5

        # Assertion density (15 points)
        assertion_ratio = metrics["assertion_count"] / max(test_count, 1)
        if assertion_ratio >= 3:
            score += 15
        elif assertion_ratio >= 2:
            score += 10
        elif assertion_ratio >= 1:
            score += 5

        # Coverage types (15 points)
        coverage_types = len(metrics["coverage_types"])
        score += min(coverage_types * 3, 15)

        # Best practices (20 points)
        best_practices = metrics["best_practices"]
        practices_met = sum(1 for v in best_practices.values() if v)
        score += (practices_met / len(best_practices)) * 20

        # Deductions
        score -= min(metrics["code_duplication"], 10)  # Max 10 point deduction
        score -= min(metrics["complexity_score"] / 10, 10)  # Max 10 point deduction

        # Maintainability bonus (up to 10 points)
        score += (metrics["maintainability_index"] / 100) * 10

        return max(0, min(100, round(score, 2)))


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'SecurityScanner',
    'QualityAnalyzer',
    'SecurityScanResult',
    'ComplianceReport',
    'SecurityIssue',
    'ComplianceStandard',
    'SecurityCategory'
]