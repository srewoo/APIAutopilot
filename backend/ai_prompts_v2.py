# -*- coding: utf-8 -*-
"""
Enhanced AI Prompts System V2
Generates focused, effective prompts with minimal token usage
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class PromptType(Enum):
    """Types of prompts"""
    ASSERTIONS = "assertions"
    BUSINESS_LOGIC = "business_logic"
    SECURITY_PAYLOADS = "security_payloads"
    ERROR_SCENARIOS = "error_scenarios"
    EDGE_CASES = "edge_cases"
    PERFORMANCE = "performance"
    DATA_GENERATION = "data_generation"


@dataclass
class PromptTemplate:
    """Reusable prompt template"""
    type: PromptType
    template: str
    max_tokens: int
    temperature: float
    system_message: str


class SmartPromptSystem:
    """
    Intelligent prompt generation system with templates and context awareness
    """

    def __init__(self):
        self.templates = self._load_templates()
        self.context_cache = {}

    def _load_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Load optimized prompt templates"""
        return {
            PromptType.ASSERTIONS: PromptTemplate(
                type=PromptType.ASSERTIONS,
                template="""Generate test assertions for this response:
{response_data}

Output ONLY assertions (one per line):
expect(response.data.{field}).toBe({value});
expect(typeof response.data.{field}).toBe('{type}');

NO explanations. NO comments.""",
                max_tokens=500,
                temperature=0.1,
                system_message="You are a test assertion generator. Output ONLY valid test assertions."
            ),

            PromptType.BUSINESS_LOGIC: PromptTemplate(
                type=PromptType.BUSINESS_LOGIC,
                template="""API: {endpoint}
Method: {method}

Generate 3 business test scenarios.

Output JSON only:
[{{"scenario": "...", "input": {{}}, "expected": "..."}}]""",
                max_tokens=400,
                temperature=0.3,
                system_message="Generate realistic business test scenarios. Output valid JSON only."
            ),

            PromptType.SECURITY_PAYLOADS: PromptTemplate(
                type=PromptType.SECURITY_PAYLOADS,
                template="""Generate {count} {category} payloads.

Output one per line:
payload_here

NO explanations.""",
                max_tokens=300,
                temperature=0.2,
                system_message="Generate security test payloads. One per line, no explanations."
            ),

            PromptType.ERROR_SCENARIOS: PromptTemplate(
                type=PromptType.ERROR_SCENARIOS,
                template="""API: {endpoint}

List 5 error scenarios with status codes:
scenario|status_code

Example:
Missing auth token|401""",
                max_tokens=200,
                temperature=0.2,
                system_message="Generate error test scenarios with HTTP status codes."
            ),

            PromptType.EDGE_CASES: PromptTemplate(
                type=PromptType.EDGE_CASES,
                template="""Field: {field_name}
Type: {field_type}

Generate 5 edge case values:""",
                max_tokens=150,
                temperature=0.3,
                system_message="Generate edge case test values. One per line."
            ),

            PromptType.PERFORMANCE: PromptTemplate(
                type=PromptType.PERFORMANCE,
                template="""Endpoint: {endpoint}

Generate performance test parameters:
- Load (requests/sec):
- Duration (seconds):
- Timeout threshold (ms):
- Expected p95 (ms):""",
                max_tokens=100,
                temperature=0.1,
                system_message="Generate performance test parameters."
            ),

            PromptType.DATA_GENERATION: PromptTemplate(
                type=PromptType.DATA_GENERATION,
                template="""Generate {count} test data samples matching:
{schema}

Output JSON array only.""",
                max_tokens=500,
                temperature=0.4,
                system_message="Generate test data matching the schema. Output valid JSON only."
            )
        }

    def generate_prompt(
        self,
        prompt_type: PromptType,
        context: Dict[str, Any],
        optimize_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Generate optimized prompt based on type and context

        Args:
            prompt_type: Type of prompt to generate
            context: Context data for the prompt
            optimize_tokens: Whether to optimize for token usage

        Returns:
            Dictionary with prompt, system message, and settings
        """
        template = self.templates.get(prompt_type)
        if not template:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Format the template with context
        prompt = template.template.format(**context)

        # Optimize if requested
        if optimize_tokens:
            prompt = self._optimize_prompt(prompt)

        return {
            "prompt": prompt,
            "system_message": template.system_message,
            "max_tokens": template.max_tokens,
            "temperature": template.temperature
        }

    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt to reduce token usage"""
        # Remove extra whitespace
        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        optimized = '\n'.join(lines)

        # Remove redundant words
        replacements = {
            "Please generate": "Generate",
            "You should": "",
            "Make sure to": "",
            "Remember to": "",
            "It's important to": "",
            "You need to": ""
        }

        for old, new in replacements.items():
            optimized = optimized.replace(old, new)

        return optimized

    def generate_assertion_prompt(self, response_data: Dict[str, Any]) -> str:
        """Generate focused prompt for assertions only"""
        # Simplify response data to reduce tokens
        simplified = self._simplify_data_structure(response_data)

        context = {
            "response_data": json.dumps(simplified, indent=2)[:500]  # Limit size
        }

        result = self.generate_prompt(PromptType.ASSERTIONS, context)
        return result["prompt"]

    def generate_business_logic_prompt(self, endpoint: str, method: str) -> str:
        """Generate prompt for business logic tests"""
        context = {
            "endpoint": endpoint,
            "method": method
        }

        result = self.generate_prompt(PromptType.BUSINESS_LOGIC, context)
        return result["prompt"]

    def generate_security_payload_prompt(self, category: str, count: int = 5) -> str:
        """Generate prompt for security payloads"""
        context = {
            "category": category,
            "count": count
        }

        result = self.generate_prompt(PromptType.SECURITY_PAYLOADS, context)
        return result["prompt"]

    def generate_error_scenario_prompt(self, endpoint: str) -> str:
        """Generate prompt for error scenarios"""
        context = {
            "endpoint": endpoint
        }

        result = self.generate_prompt(PromptType.ERROR_SCENARIOS, context)
        return result["prompt"]

    def generate_edge_case_prompt(self, field_name: str, field_type: str) -> str:
        """Generate prompt for edge cases"""
        context = {
            "field_name": field_name,
            "field_type": field_type
        }

        result = self.generate_prompt(PromptType.EDGE_CASES, context)
        return result["prompt"]

    def generate_data_generation_prompt(self, schema: Dict[str, Any], count: int = 5) -> str:
        """Generate prompt for test data generation"""
        context = {
            "schema": json.dumps(schema, indent=2)[:300],  # Limit size
            "count": count
        }

        result = self.generate_prompt(PromptType.DATA_GENERATION, context)
        return result["prompt"]

    def _simplify_data_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
        """Simplify data structure to reduce tokens"""
        if current_depth >= max_depth:
            return "..."

        if isinstance(data, dict):
            simplified = {}
            for key, value in list(data.items())[:5]:  # Limit to first 5 keys
                simplified[key] = self._simplify_data_structure(value, max_depth, current_depth + 1)
            if len(data) > 5:
                simplified["..."] = f"({len(data) - 5} more fields)"
            return simplified

        elif isinstance(data, list):
            if not data:
                return []
            # Only show first item as example
            return [self._simplify_data_structure(data[0], max_depth, current_depth + 1), "..."] if len(data) > 1 else [data[0]]

        else:
            return data

    def create_chain_prompt(self, prompts: List[Dict[str, Any]]) -> str:
        """Chain multiple prompts efficiently"""
        chained = []
        for i, prompt_data in enumerate(prompts, 1):
            chained.append(f"Task {i}: {prompt_data['prompt']}")

        return "\n\n".join(chained)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def validate_prompt_size(self, prompt: str, max_tokens: int = 4000) -> bool:
        """Check if prompt is within token limits"""
        estimated = self.estimate_tokens(prompt)
        return estimated <= max_tokens


class ContextAwarePromptGenerator:
    """
    Generates prompts with context awareness and learning
    """

    def __init__(self):
        self.prompt_system = SmartPromptSystem()
        self.context_history = []
        self.successful_patterns = {}

    def generate_with_context(
        self,
        prompt_type: PromptType,
        api_context: Dict[str, Any],
        previous_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate prompt with API context awareness

        Args:
            prompt_type: Type of prompt
            api_context: API-specific context
            previous_results: Previous generation results for learning

        Returns:
            Optimized prompt string
        """
        # Check for successful patterns
        pattern_key = f"{api_context.get('api_type')}_{prompt_type.value}"
        if pattern_key in self.successful_patterns:
            base_prompt = self.successful_patterns[pattern_key]
        else:
            base_prompt = self.prompt_system.generate_prompt(prompt_type, api_context)

        # Enhance with context
        enhanced_prompt = self._enhance_with_context(base_prompt, api_context)

        # Learn from previous results
        if previous_results:
            self._learn_from_results(pattern_key, previous_results)

        return enhanced_prompt

    def _enhance_with_context(self, base_prompt: Dict[str, Any], api_context: Dict[str, Any]) -> str:
        """Enhance prompt with API-specific context"""
        prompt = base_prompt["prompt"]

        # Add API-specific hints
        if api_context.get("api_type") == "graphql":
            prompt += "\n\nNote: This is a GraphQL API. Use appropriate query/mutation syntax."
        elif api_context.get("auth_type") == "bearer":
            prompt += "\n\nNote: Use Bearer token authentication."
        elif api_context.get("auth_type") == "api_key":
            prompt += "\n\nNote: Use API key authentication."

        # Add industry-specific context
        if api_context.get("industry"):
            industry_hints = {
                "fintech": "Include PCI compliance and financial data security tests.",
                "healthcare": "Include HIPAA compliance and PHI protection tests.",
                "ecommerce": "Include payment processing and inventory tests.",
                "social": "Include privacy and content moderation tests."
            }
            hint = industry_hints.get(api_context["industry"])
            if hint:
                prompt += f"\n\n{hint}"

        return prompt

    def _learn_from_results(self, pattern_key: str, results: List[Dict[str, Any]]):
        """Learn from successful generation results"""
        # Simple learning: store successful patterns
        for result in results:
            if result.get("success") and result.get("quality_score", 0) > 80:
                self.successful_patterns[pattern_key] = result.get("prompt_used")

    def generate_minimal_prompt(self, task: str, output_format: str) -> str:
        """Generate minimal prompt for simple tasks"""
        return f"{task}\n\nOutput format:\n{output_format}\n\nOutput only:"

    def generate_few_shot_prompt(
        self,
        task: str,
        examples: List[Dict[str, Any]],
        new_input: Dict[str, Any]
    ) -> str:
        """Generate few-shot learning prompt"""
        prompt_parts = [task, "\nExamples:"]

        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {json.dumps(example['input'], separators=(',', ':'))}")
            prompt_parts.append(f"Output: {json.dumps(example['output'], separators=(',', ':'))}")

        prompt_parts.append(f"\nNow generate for:")
        prompt_parts.append(f"Input: {json.dumps(new_input, separators=(',', ':'))}")
        prompt_parts.append("Output:")

        return "\n".join(prompt_parts)


class PromptOptimizer:
    """
    Optimizes prompts for better results and lower token usage
    """

    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()

    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load prompt optimization rules"""
        return [
            {"pattern": r'\s+', "replacement": ' '},  # Collapse whitespace
            {"pattern": r'^\s+|\s+$', "replacement": ''},  # Trim
            {"pattern": r'\n{3,}', "replacement": '\n\n'},  # Limit newlines
            {"pattern": r'please\s+', "replacement": '', "flags": re.IGNORECASE},
            {"pattern": r'could you\s+', "replacement": '', "flags": re.IGNORECASE},
            {"pattern": r'make sure to\s+', "replacement": '', "flags": re.IGNORECASE},
            {"pattern": r'remember to\s+', "replacement": '', "flags": re.IGNORECASE},
        ]

    def optimize(self, prompt: str) -> str:
        """Optimize prompt for efficiency"""
        import re

        optimized = prompt

        # Apply optimization rules
        for rule in self.optimization_rules:
            pattern = rule["pattern"]
            replacement = rule["replacement"]
            flags = rule.get("flags", 0)
            optimized = re.sub(pattern, replacement, optimized, flags=flags)

        # Remove redundant instructions
        optimized = self._remove_redundancy(optimized)

        # Compress JSON if present
        optimized = self._compress_json(optimized)

        return optimized

    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant instructions"""
        lines = text.split('\n')
        seen = set()
        unique_lines = []

        for line in lines:
            # Normalize for comparison
            normalized = line.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)

        return '\n'.join(unique_lines)

    def _compress_json(self, text: str) -> str:
        """Compress JSON in prompts"""
        import re

        def compress_json_match(match):
            try:
                data = json.loads(match.group(0))
                return json.dumps(data, separators=(',', ':'))
            except:
                return match.group(0)

        # Find and compress JSON blocks
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        return re.sub(pattern, compress_json_match, text)

    def split_large_prompt(self, prompt: str, max_tokens: int = 3000) -> List[str]:
        """Split large prompts into smaller chunks"""
        estimated_tokens = len(prompt) // 4

        if estimated_tokens <= max_tokens:
            return [prompt]

        # Split into logical chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for line in prompt.split('\n'):
            line_size = len(line) // 4
            if current_size + line_size > max_tokens:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks


# ============================================================================
# EXPORT
# ============================================================================

class SmartPromptGenerator:
    """Smart prompt generator for AI enhancement"""

    def generate_enhancement_prompt(self, test_cases, api_analysis, framework):
        """Generate AI enhancement prompt for test cases"""
        prompt = f"""
Enhance these {framework.value} test cases with better assertions and edge cases.

Current test cases:
{self._format_test_cases(test_cases[:3])}  # Only send first 3 for context

API Analysis:
- Method: {api_analysis.method}
- Endpoint: {api_analysis.endpoint}
- Has Auth: {api_analysis.has_auth}

REQUIREMENTS:
1. Add more specific assertions based on the API response
2. Include edge case testing
3. Add performance assertions where appropriate
4. Keep responses concise and code-focused
5. Temperature is set to {getattr(self, 'temperature', 0.1)} for consistency

Return ONLY improved test code snippets. No explanations.
"""
        return prompt

    def _format_test_cases(self, test_cases):
        """Format test cases for prompt"""
        formatted = []
        for tc in test_cases:
            formatted.append(f"- {tc.name}: {tc.description}")
        return "\n".join(formatted)


__all__ = [
    'SmartPromptSystem',
    'ContextAwarePromptGenerator',
    'PromptOptimizer',
    'PromptType',
    'PromptTemplate',
    'SmartPromptGenerator'
]