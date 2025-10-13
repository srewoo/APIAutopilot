# -*- coding: utf-8 -*-
"""
API Discovery and Specification Detection Module
Automatically discovers API endpoints and detects specification formats
"""

import json
import yaml
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse, urljoin
import aiohttp
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SpecificationType(Enum):
    """Supported API specification types"""
    OPENAPI_3 = "openapi_3"
    SWAGGER_2 = "swagger_2"
    GRAPHQL = "graphql"
    POSTMAN = "postman"
    INSOMNIA = "insomnia"
    HAR = "har"
    RAML = "raml"
    API_BLUEPRINT = "api_blueprint"
    WADL = "wadl"
    WSDL = "wsdl"
    UNKNOWN = "unknown"


@dataclass
class DiscoveredEndpoint:
    """Represents a discovered API endpoint"""
    path: str
    method: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = None
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = None
    auth_required: bool = False
    tags: List[str] = None


@dataclass
class APISpecification:
    """Parsed API specification"""
    type: SpecificationType
    version: str
    title: str
    description: str
    base_url: str
    endpoints: List[DiscoveredEndpoint]
    auth_schemes: Dict[str, Any]
    schemas: Dict[str, Any]
    metadata: Dict[str, Any]


class APISpecDetector:
    """Detects and parses various API specification formats"""

    def detect_spec_type(self, content: str) -> Tuple[SpecificationType, Dict[str, Any]]:
        """
        Detect specification type from content
        Returns: (spec_type, parsed_content)
        """
        # Try JSON first
        try:
            data = json.loads(content)

            # OpenAPI 3.x
            if "openapi" in data and data["openapi"].startswith("3."):
                return SpecificationType.OPENAPI_3, data

            # Swagger 2.0
            if "swagger" in data and data["swagger"] == "2.0":
                return SpecificationType.SWAGGER_2, data

            # Postman Collection
            if "info" in data and "item" in data and data.get("info", {}).get("schema"):
                return SpecificationType.POSTMAN, data

            # Insomnia Export
            if "__export_format" in data and data["__export_format"] == 4:
                return SpecificationType.INSOMNIA, data

            # HAR format
            if "log" in data and "entries" in data["log"]:
                return SpecificationType.HAR, data

            # GraphQL Introspection
            if "data" in data and "__schema" in data.get("data", {}):
                return SpecificationType.GRAPHQL, data

        except json.JSONDecodeError:
            pass

        # Try YAML
        try:
            data = yaml.safe_load(content)

            # OpenAPI/Swagger in YAML
            if isinstance(data, dict):
                if "openapi" in data:
                    return SpecificationType.OPENAPI_3, data
                elif "swagger" in data:
                    return SpecificationType.SWAGGER_2, data
                elif "raml" in str(content)[:100]:
                    return SpecificationType.RAML, data

        except yaml.YAMLError:
            pass

        # Check for other formats by content patterns
        if "FORMAT: 1A" in content[:50]:
            return SpecificationType.API_BLUEPRINT, {"raw": content}

        if "<application xmlns" in content and "wadl" in content:
            return SpecificationType.WADL, {"raw": content}

        if "<definitions xmlns" in content and "wsdl" in content:
            return SpecificationType.WSDL, {"raw": content}

        return SpecificationType.UNKNOWN, {"raw": content}

    def parse_specification(self, content: str) -> Optional[APISpecification]:
        """Parse API specification from content"""
        spec_type, data = self.detect_spec_type(content)

        if spec_type == SpecificationType.OPENAPI_3:
            return self._parse_openapi_3(data)
        elif spec_type == SpecificationType.SWAGGER_2:
            return self._parse_swagger_2(data)
        elif spec_type == SpecificationType.POSTMAN:
            return self._parse_postman(data)
        elif spec_type == SpecificationType.HAR:
            return self._parse_har(data)
        elif spec_type == SpecificationType.GRAPHQL:
            return self._parse_graphql(data)
        else:
            logger.warning(f"Unsupported specification type: {spec_type}")
            return None

    def _parse_openapi_3(self, spec: Dict[str, Any]) -> APISpecification:
        """Parse OpenAPI 3.x specification"""
        endpoints = []
        servers = spec.get("servers", [])
        base_url = servers[0]["url"] if servers else ""

        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                    endpoint = DiscoveredEndpoint(
                        path=path,
                        method=method.upper(),
                        description=operation.get("summary", ""),
                        parameters=operation.get("parameters", []),
                        request_body=operation.get("requestBody"),
                        responses=operation.get("responses", {}),
                        auth_required="security" in operation or "security" in spec,
                        tags=operation.get("tags", [])
                    )
                    endpoints.append(endpoint)

        return APISpecification(
            type=SpecificationType.OPENAPI_3,
            version=spec.get("openapi", "3.0.0"),
            title=spec.get("info", {}).get("title", ""),
            description=spec.get("info", {}).get("description", ""),
            base_url=base_url,
            endpoints=endpoints,
            auth_schemes=spec.get("components", {}).get("securitySchemes", {}),
            schemas=spec.get("components", {}).get("schemas", {}),
            metadata={"original": spec}
        )

    def _parse_swagger_2(self, spec: Dict[str, Any]) -> APISpecification:
        """Parse Swagger 2.0 specification"""
        endpoints = []
        schemes = spec.get("schemes", ["https"])
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        base_url = f"{schemes[0]}://{host}{base_path}"

        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method in ["get", "post", "put", "patch", "delete", "head", "options"]:
                    endpoint = DiscoveredEndpoint(
                        path=path,
                        method=method.upper(),
                        description=operation.get("summary", ""),
                        parameters=operation.get("parameters", []),
                        responses=operation.get("responses", {}),
                        auth_required="security" in operation or "security" in spec,
                        tags=operation.get("tags", [])
                    )
                    endpoints.append(endpoint)

        return APISpecification(
            type=SpecificationType.SWAGGER_2,
            version="2.0",
            title=spec.get("info", {}).get("title", ""),
            description=spec.get("info", {}).get("description", ""),
            base_url=base_url,
            endpoints=endpoints,
            auth_schemes=spec.get("securityDefinitions", {}),
            schemas=spec.get("definitions", {}),
            metadata={"original": spec}
        )

    def _parse_postman(self, collection: Dict[str, Any]) -> APISpecification:
        """Parse Postman collection"""
        endpoints = []

        def extract_requests(items, parent_path=""):
            for item in items:
                if "request" in item:
                    request = item["request"]
                    url = request.get("url", {})

                    # Handle different URL formats
                    if isinstance(url, str):
                        path = urlparse(url).path
                    elif isinstance(url, dict):
                        path = "/" + "/".join(url.get("path", []))
                    else:
                        path = "/"

                    endpoint = DiscoveredEndpoint(
                        path=path,
                        method=request.get("method", "GET"),
                        description=item.get("name", ""),
                        request_body=request.get("body"),
                        auth_required=request.get("auth") is not None
                    )
                    endpoints.append(endpoint)

                # Recursive for folders
                if "item" in item:
                    extract_requests(item["item"], parent_path)

        extract_requests(collection.get("item", []))

        return APISpecification(
            type=SpecificationType.POSTMAN,
            version=collection.get("info", {}).get("schema", ""),
            title=collection.get("info", {}).get("name", ""),
            description=collection.get("info", {}).get("description", ""),
            base_url="",
            endpoints=endpoints,
            auth_schemes={},
            schemas={},
            metadata={"original": collection}
        )

    def _parse_har(self, har: Dict[str, Any]) -> APISpecification:
        """Parse HAR file"""
        endpoints = []
        entries = har.get("log", {}).get("entries", [])

        for entry in entries:
            request = entry.get("request", {})
            url = urlparse(request.get("url", ""))

            endpoint = DiscoveredEndpoint(
                path=url.path,
                method=request.get("method", "GET"),
                request_body=request.get("postData"),
                auth_required=any("authorization" in h.get("name", "").lower()
                                 for h in request.get("headers", []))
            )
            endpoints.append(endpoint)

        return APISpecification(
            type=SpecificationType.HAR,
            version="1.2",
            title="HAR Import",
            description="Imported from HAR file",
            base_url="",
            endpoints=endpoints,
            auth_schemes={},
            schemas={},
            metadata={"original": har}
        )

    def _parse_graphql(self, introspection: Dict[str, Any]) -> APISpecification:
        """Parse GraphQL introspection"""
        schema = introspection.get("data", {}).get("__schema", {})
        endpoints = []

        # Extract queries
        query_type = schema.get("queryType", {})
        if query_type:
            for field in self._get_graphql_fields(schema, query_type.get("name")):
                endpoints.append(DiscoveredEndpoint(
                    path="/graphql",
                    method="POST",
                    description=field.get("description", ""),
                    tags=["query", field.get("name", "")]
                ))

        # Extract mutations
        mutation_type = schema.get("mutationType", {})
        if mutation_type:
            for field in self._get_graphql_fields(schema, mutation_type.get("name")):
                endpoints.append(DiscoveredEndpoint(
                    path="/graphql",
                    method="POST",
                    description=field.get("description", ""),
                    tags=["mutation", field.get("name", "")]
                ))

        return APISpecification(
            type=SpecificationType.GRAPHQL,
            version="GraphQL",
            title="GraphQL API",
            description="GraphQL schema",
            base_url="/graphql",
            endpoints=endpoints,
            auth_schemes={},
            schemas=schema.get("types", []),
            metadata={"original": introspection}
        )

    def _get_graphql_fields(self, schema: Dict, type_name: str) -> List[Dict]:
        """Get fields for a GraphQL type"""
        for type_def in schema.get("types", []):
            if type_def.get("name") == type_name:
                return type_def.get("fields", [])
        return []


class APIDiscovery:
    """Automatically discover API endpoints and documentation"""

    def __init__(self):
        self.common_doc_paths = [
            "/swagger.json",
            "/swagger.yaml",
            "/openapi.json",
            "/openapi.yaml",
            "/api-docs",
            "/api-docs.json",
            "/api/swagger.json",
            "/api/swagger.yaml",
            "/api/openapi.json",
            "/api/openapi.yaml",
            "/v1/swagger.json",
            "/v2/swagger.json",
            "/v3/swagger.json",
            "/v1/openapi.json",
            "/v2/openapi.json",
            "/v3/openapi.json",
            "/.well-known/openapi.json",
            "/.well-known/swagger.json",
            "/api-documentation",
            "/docs/api",
            "/graphql",
            "/graphql/schema",
            "/_doc",
            "/api.json",
            "/api.yaml",
            "/api.raml",
            "/application.wadl"
        ]
        self.spec_detector = APISpecDetector()

    async def discover_api_spec(self, base_url: str, timeout: int = 10) -> Optional[APISpecification]:
        """
        Discover API specification from a base URL

        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds

        Returns:
            Discovered API specification or None
        """
        discovered_specs = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for path in self.common_doc_paths:
                url = urljoin(base_url, path)
                task = self._fetch_spec(session, url, timeout)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if result and not isinstance(result, Exception):
                    spec = self.spec_detector.parse_specification(result)
                    if spec:
                        discovered_specs.append(spec)

        # Return the first valid specification found
        if discovered_specs:
            logger.info(f"Discovered {len(discovered_specs)} API specification(s)")
            return discovered_specs[0]

        # Try GraphQL introspection as fallback
        graphql_spec = await self._introspect_graphql(base_url)
        if graphql_spec:
            return graphql_spec

        logger.warning(f"No API specification found at {base_url}")
        return None

    async def _fetch_spec(self, session: aiohttp.ClientSession, url: str, timeout: int) -> Optional[str]:
        """Fetch potential API specification from URL"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    content_type = response.headers.get("content-type", "")

                    # Check if it's likely an API spec
                    if any(x in content_type.lower() for x in ["json", "yaml", "xml", "text"]):
                        content = await response.text()

                        # Quick validation - check if content looks like a spec
                        if any(keyword in content.lower()[:1000] for keyword in
                              ["openapi", "swagger", "paths", "graphql", "postman", "insomnia"]):
                            logger.info(f"Found potential API spec at {url}")
                            return content
        except Exception as e:
            # Silently fail for individual paths
            pass

        return None

    async def _introspect_graphql(self, base_url: str) -> Optional[APISpecification]:
        """Attempt GraphQL introspection"""
        introspection_query = """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
              ...FullType
            }
          }
        }

        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
        }

        fragment InputValue on __InputValue {
          name
          description
          type { ...TypeRef }
          defaultValue
        }

        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
          }
        }
        """

        graphql_endpoints = ["/graphql", "/api/graphql", "/graphql/v1"]

        async with aiohttp.ClientSession() as session:
            for endpoint in graphql_endpoints:
                url = urljoin(base_url, endpoint)
                try:
                    async with session.post(
                        url,
                        json={"query": introspection_query},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "data" in result and "__schema" in result["data"]:
                                logger.info(f"Successfully introspected GraphQL at {url}")
                                return self.spec_detector._parse_graphql(result)
                except Exception:
                    continue

        return None

    async def discover_from_html(self, base_url: str) -> List[str]:
        """
        Discover API endpoints by crawling HTML pages

        Args:
            base_url: Base URL to start crawling

        Returns:
            List of discovered API endpoint URLs
        """
        discovered_endpoints = set()

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(base_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()

                        # Find API-related links in HTML
                        api_patterns = [
                            r'href=["\']([^"\']*api[^"\']*)["\']',
                            r'href=["\']([^"\']*swagger[^"\']*)["\']',
                            r'href=["\']([^"\']*docs[^"\']*)["\']',
                            r'action=["\']([^"\']*)["\']',
                            r'data-url=["\']([^"\']*)["\']',
                            r'\.ajax\(["\']([^"\']*)["\']',
                            r'fetch\(["\']([^"\']*)["\']',
                            r'axios\.[a-z]+\(["\']([^"\']*)["\']'
                        ]

                        for pattern in api_patterns:
                            matches = re.findall(pattern, html, re.IGNORECASE)
                            for match in matches:
                                if match.startswith("/"):
                                    endpoint_url = urljoin(base_url, match)
                                elif match.startswith("http"):
                                    endpoint_url = match
                                else:
                                    endpoint_url = urljoin(base_url, "/" + match)

                                discovered_endpoints.add(endpoint_url)

                        logger.info(f"Discovered {len(discovered_endpoints)} potential endpoints from HTML")
            except Exception as e:
                logger.error(f"Error discovering from HTML: {e}")

        return list(discovered_endpoints)

    def import_from_curl(self, curl_command: str) -> Optional[DiscoveredEndpoint]:
        """
        Import endpoint from cURL command

        Args:
            curl_command: cURL command string

        Returns:
            Discovered endpoint or None
        """
        # Parse cURL command
        url_match = re.search(r'curl\s+(?:-X\s+\w+\s+)?[\'"]?([^\s\'"]+)', curl_command)
        if not url_match:
            return None

        url = url_match.group(1)
        parsed_url = urlparse(url)

        # Extract method
        method_match = re.search(r'-X\s+(\w+)', curl_command)
        method = method_match.group(1) if method_match else "GET"

        # Extract headers
        headers = {}
        header_matches = re.finditer(r'-H\s+[\'"]([^:]+):\s*([^\'"]+)[\'"]', curl_command)
        for match in header_matches:
            headers[match.group(1)] = match.group(2)

        # Extract body
        body = None
        body_match = re.search(r'--data(?:-raw|-binary)?\s+[\'"](.+?)[\'"]', curl_command, re.DOTALL)
        if body_match:
            body = body_match.group(1)

        return DiscoveredEndpoint(
            path=parsed_url.path or "/",
            method=method,
            request_body={"raw": body} if body else None,
            auth_required="Authorization" in headers
        )


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'APISpecDetector',
    'APIDiscovery',
    'APISpecification',
    'DiscoveredEndpoint',
    'SpecificationType'
]