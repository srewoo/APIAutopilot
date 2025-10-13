import { useState, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, CheckCircle2, Code, Copy, Download, Loader2, Sparkles, Shield, Database, Zap, HelpCircle, ExternalLink } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const USE_V2 = process.env.REACT_APP_USE_V2 !== 'false'; // Default to V2
const API = USE_V2 ? `${BACKEND_URL}/api/v2` : `${BACKEND_URL}/api`;

function App() {
  const [inputType, setInputType] = useState("curl");
  const [curlData, setCurlData] = useState("");
  const [harData, setHarData] = useState("");
  const [textData, setTextData] = useState("");
  const [exampleResponse, setExampleResponse] = useState("");
  const [autoCapture, setAutoCapture] = useState(true); // Default to true
  const [aiProvider, setAiProvider] = useState(() => localStorage.getItem("aiProvider") || "openai");
  const [aiModel, setAiModel] = useState(() => localStorage.getItem("aiModel") || "gpt-4o");
  const [aiApiKey, setAiApiKey] = useState(() => localStorage.getItem("aiApiKey") || "");
  const [testFramework, setTestFramework] = useState("jest");
  const [testProfile, setTestProfile] = useState("full_regression"); // Add test profile
  const [moduleSystem, setModuleSystem] = useState("commonjs"); // Default to CommonJS
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [temperature, setTemperature] = useState(0.1); // V2 uses lower temperature for deterministic output
  const [verifySsl, setVerifySsl] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);

  // Define available models for each provider
  const aiModels = {
    openai: [
      { value: "gpt-4o", label: "GPT-4o" },
      { value: "gpt-4o-mini", label: "GPT-4o Mini" },
      { value: "o3-mini", label: "O3 Mini" },
    ],
    anthropic: [
      { value: "claude-3-7-sonnet-20250219", label: "Claude 3.7 Sonnet" },
      { value: "claude-sonnet-4-5-20250514", label: "Claude 4.5 Sonnet" },
    ],
    gemini: [
      { value: "gemini-2.5-pro-latest", label: "Gemini 2.5 Pro" },
      { value: "gemini-2.5-flash-latest", label: "Gemini 2.5 Flash" },
    ],
  };

  // Save AI provider, model, and API key to localStorage whenever they change
  useEffect(() => {
    if (aiProvider) {
      localStorage.setItem("aiProvider", aiProvider);
      // Reset model to first available when provider changes
      const firstModel = aiModels[aiProvider]?.[0]?.value;
      if (firstModel && !aiModels[aiProvider].find(m => m.value === aiModel)) {
        setAiModel(firstModel);
      }
    }
  }, [aiProvider, aiModels, aiModel]);

  useEffect(() => {
    if (aiModel) {
      localStorage.setItem("aiModel", aiModel);
    }
  }, [aiModel]);

  useEffect(() => {
    if (aiApiKey) {
      localStorage.setItem("aiApiKey", aiApiKey);
    }
  }, [aiApiKey]);

  // Get current input data based on selected tab
  const getCurrentInputData = () => {
    switch (inputType) {
      case "curl": return curlData;
      case "har": return harData;
      case "text": return textData;
      default: return "";
    }
  };

  // Auto-detect if input is GraphQL or cURL
  const detectInputType = (data) => {
    const trimmedData = data.trim();
    // Check if it's GraphQL (contains query/mutation keywords or JSON with query field)
    if (trimmedData.match(/^\s*(query|mutation|subscription|fragment)\s*[{\(]/i)) {
      return 'graphql';
    }
    try {
      const parsed = JSON.parse(trimmedData);
      if (parsed.query || parsed.mutation) {
        return 'graphql';
      }
    } catch (e) {
      // Not JSON, continue checking
    }
    // If starts with curl, it's cURL
    if (trimmedData.startsWith('curl ')) {
      return 'curl';
    }
    // Default to curl for the curl/graphql tab
    return 'curl';
  };

  const handleGenerate = async () => {
    const currentInputData = getCurrentInputData();
    
    if (!currentInputData.trim()) {
      setError("Please provide input data");
      return;
    }
    if (!aiApiKey.trim()) {
      setError("Please provide AI API key");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      // Auto-detect GraphQL vs cURL for the merged tab
      let actualInputType = inputType;
      if (inputType === 'curl') {
        actualInputType = detectInputType(currentInputData);
      }

      const response = await axios.post(`${API}/generate-tests`, {
        input_type: actualInputType,
        input_data: currentInputData,
        ai_provider: aiProvider,
        ai_model: aiModel,
        ai_api_key: aiApiKey,
        test_framework: testFramework,
        test_profile: testProfile, // Add test profile
        module_system: moduleSystem,
        example_response: exampleResponse || null,
        auto_capture: autoCapture,
        temperature: temperature,
        verify_ssl: verifySsl,
        security_scan: true, // Enable security scan by default
        execute_tests: false, // Can be toggled in UI
      });

      if (response.data.success) {
        setResult(response.data);
      } else {
        setError(response.data.error || "Failed to generate tests");
      }
    } catch (err) {
      // Handle various error formats (string, object, array)
      let errorMessage = "An error occurred";
      
      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;
        
        if (typeof detail === "string") {
          errorMessage = detail;
        } else if (Array.isArray(detail)) {
          // Handle Pydantic validation errors (array of error objects)
          errorMessage = detail.map(e => `${e.loc?.join('.') || 'Error'}: ${e.msg}`).join(', ');
        } else if (typeof detail === "object") {
          // Handle single error object
          errorMessage = detail.msg || JSON.stringify(detail);
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (result?.test_script) {
      navigator.clipboard.writeText(result.test_script);
      setCopySuccess(true);
      // Hide success message after 3 seconds
      setTimeout(() => {
        setCopySuccess(false);
      }, 3000);
    }
  };

  const handleDownload = () => {
    if (result?.test_script) {
      const extensions = {
        jest: ".test.js",
        mocha: ".test.js",
        cypress: ".cy.js",
        pytest: ".py",
        requests: ".py",
        testng: ".java",
        junit: ".java",
        restassured: ".java",
        behave: ".feature",
      };
      const ext = extensions[result.framework] || ".txt";
      const blob = new Blob([result.test_script], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `api_test${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="App">
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-orange-50 to-slate-100">
        {/* Header */}
        <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl">
                  <Code className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-slate-900">API Autopilot</h1>
                  <p className="text-sm text-slate-600">Your AI Testing Co-Pilot</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  <Shield className="w-3 h-3 mr-1" />
                  Security Tests
                </Badge>
                <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                  <Database className="w-3 h-3 mr-1" />
                  REST + GraphQL
                </Badge>
                <a 
                  href="http://localhost:8000/docs" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-slate-600 hover:text-orange-600 hover:bg-orange-50 rounded-lg transition-colors"
                  title="API Documentation & Help"
                >
                  <HelpCircle className="w-4 h-4" />
                  <span className="hidden sm:inline">Docs</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="container mx-auto px-6 py-8">
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Column - Input */}
            <div className="space-y-6">
              <Card data-testid="input-card" className="border-slate-200 shadow-lg">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-orange-600" />
                    Test Configuration
                  </CardTitle>
                  <CardDescription>Configure comprehensive test generation</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* AI Provider and Model Selection - Side by Side */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="ai-provider" className="text-sm font-semibold text-slate-700">
                        AI Provider
                      </Label>
                      <Select value={aiProvider} onValueChange={setAiProvider}>
                        <SelectTrigger data-testid="ai-provider-select" id="ai-provider">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="openai">OpenAI</SelectItem>
                          <SelectItem value="anthropic">Anthropic</SelectItem>
                          <SelectItem value="gemini">Google</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="ai-model" className="text-sm font-semibold text-slate-700">
                        Model
                      </Label>
                      <Select value={aiModel} onValueChange={setAiModel}>
                        <SelectTrigger data-testid="ai-model-select" id="ai-model">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {aiModels[aiProvider]?.map((model) => (
                            <SelectItem key={model.value} value={model.value}>
                              {model.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* API Key */}
                  <div className="space-y-2">
                    <Label htmlFor="api-key" className="text-sm font-semibold text-slate-700">
                      API Key
                    </Label>
                    <Input
                      data-testid="api-key-input"
                      id="api-key"
                      type="password"
                      placeholder="Enter your AI API key"
                      value={aiApiKey}
                      onChange={(e) => setAiApiKey(e.target.value)}
                      className="border-slate-300"
                    />
                  </div>

                  {/* Test Framework */}
                  <div className="space-y-2">
                    <Label htmlFor="framework" className="text-sm font-semibold text-slate-700">
                      Test Framework
                    </Label>
                    <Select value={testFramework} onValueChange={setTestFramework}>
                      <SelectTrigger data-testid="framework-select" id="framework">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="jest">Jest (JavaScript)</SelectItem>
                        <SelectItem value="mocha">Mocha (JavaScript)</SelectItem>
                        <SelectItem value="cypress">Cypress (JavaScript)</SelectItem>
                        <SelectItem value="pytest">Pytest (Python)</SelectItem>
                        <SelectItem value="requests">Requests + unittest (Python)</SelectItem>
                        <SelectItem value="testng">TestNG (Java)</SelectItem>
                        <SelectItem value="junit">JUnit (Java)</SelectItem>
                        <SelectItem value="restassured">RestAssured (Java)</SelectItem>
                        <SelectItem value="behave">Behave BDD (Python)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Test Profile */}
                  <div className="space-y-2">
                    <Label htmlFor="test-profile" className="text-sm font-semibold text-slate-700">
                      Test Profile
                    </Label>
                    <Select value={testProfile} onValueChange={setTestProfile}>
                      <SelectTrigger data-testid="test-profile-select" id="test-profile">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="quick_smoke">Quick Smoke (5-10 tests)</SelectItem>
                        <SelectItem value="full_regression">Full Regression (20-30 tests)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-slate-500">
                      {testProfile === "quick_smoke" && "Fast validation with essential positive and auth tests"}
                      {testProfile === "full_regression" && "Comprehensive coverage with positive, negative, security, and edge cases"}
                    </p>
                  </div>

                  {/* Module System (only for JavaScript frameworks) */}
                  {['jest', 'mocha', 'cypress'].includes(testFramework) && (
                    <div className="space-y-2">
                      <Label htmlFor="module-system" className="text-sm font-semibold text-slate-700">
                        Module System
                      </Label>
                      <Select value={moduleSystem} onValueChange={setModuleSystem}>
                        <SelectTrigger data-testid="module-system-select" id="module-system">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="commonjs">CommonJS (require/module.exports)</SelectItem>
                          <SelectItem value="esm">ES Modules (import/export)</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-slate-500">
                        {moduleSystem === "commonjs" 
                          ? "Uses: const axios = require('axios')" 
                          : "Uses: import axios from 'axios'"}
                      </p>
                    </div>
                  )}

                  {/* Input Type Tabs */}
                  <div className="space-y-2">
                    <Label className="text-sm font-semibold text-slate-700">Input Method</Label>
                    <Tabs value={inputType} onValueChange={setInputType} className="w-full">
                      <TabsList className="grid w-full grid-cols-4">
                        <TabsTrigger data-testid="curl-tab" value="curl">cURL / GraphQL</TabsTrigger>
                        <TabsTrigger data-testid="har-tab" value="har">HAR</TabsTrigger>
                        <TabsTrigger data-testid="text-tab" value="text">Contract</TabsTrigger>
                        <TabsTrigger data-testid="response-tab" value="response">Response</TabsTrigger>
                      </TabsList>
                      <TabsContent value="curl" className="mt-4">
                        <Textarea
                          data-testid="curl-input"
                          placeholder="Paste your cURL command or GraphQL query here...

cURL Example:
curl -X POST https://api.example.com/users -H 'Content-Type: application/json' -d '{name:John}'

GraphQL Example:
query { user(id: 1) { name email } }

Or full GraphQL request with variables:
{ 'query': 'query($id: Int!) { user(id: $id) { name } }', 'variables': {'id': 1} }"
                          value={curlData}
                          onChange={(e) => setCurlData(e.target.value)}
                          className="min-h-[200px] font-mono text-sm border-slate-300"
                        />
                      </TabsContent>
                      <TabsContent value="har" className="mt-4">
                        <Textarea
                          data-testid="har-input"
                          placeholder="Paste your HAR file content here (JSON format)..."
                          value={harData}
                          onChange={(e) => setHarData(e.target.value)}
                          className="min-h-[200px] font-mono text-sm border-slate-300"
                        />
                      </TabsContent>
                      <TabsContent value="text" className="mt-4">
                        <Textarea
                          data-testid="text-input"
                          placeholder="Paste your API contract, OpenAPI/Swagger spec, or API documentation...

Example OpenAPI:
{
  'openapi': '3.0.0',
  'paths': {
    '/users': {
      'post': {
        'summary': 'Create user',
        'requestBody': { 'content': { 'application/json': { 'schema': { 'type': 'object' } } } }
      }
    }
  }
}

Or plain text description:
POST /api/users - Creates a new user
Request: name (string), email (string)
Response: user object with id"
                          value={textData}
                          onChange={(e) => setTextData(e.target.value)}
                          className="min-h-[200px] border-slate-300"
                        />
                      </TabsContent>
                      <TabsContent value="response" className="mt-4">
                        <div className="space-y-3">
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                            <div className="flex items-start gap-2 text-sm text-blue-800">
                              <Sparkles className="w-4 h-4 mt-0.5 flex-shrink-0" />
                              <div>
                                <p className="font-semibold">Optional: Provide Example Response</p>
                                <p className="text-xs mt-1">Paste an actual API response (JSON) to generate more accurate assertions with specific field validations.</p>
                              </div>
                            </div>
                          </div>
                          <Textarea
                            data-testid="response-input"
                            placeholder="Paste example API response (JSON format)...

Example:
{
  &quot;id&quot;: &quot;123e4567-e89b-12d3-a456-426614174000&quot;,
  &quot;name&quot;: &quot;Jane Smith&quot;,
  &quot;email&quot;: &quot;jane.smith@company.com&quot;,
  &quot;role&quot;: &quot;admin&quot;,
  &quot;createdAt&quot;: &quot;2024-01-15T10:30:00Z&quot;
}

This will generate tests with precise assertions for each field."
                            value={exampleResponse}
                            onChange={(e) => setExampleResponse(e.target.value)}
                            className="min-h-[200px] font-mono text-sm border-slate-300"
                          />
                        </div>
                      </TabsContent>
                    </Tabs>
                  </div>

                  {/* Auto-Capture Checkbox */}
                  <div className="space-y-3 pt-2 pb-2 border-t border-slate-200">
                    <div className="flex items-start space-x-3">
                      <Checkbox 
                        id="auto-capture" 
                        checked={autoCapture}
                        onCheckedChange={setAutoCapture}
                        className="mt-1"
                      />
                      <div className="flex-1">
                        <Label 
                          htmlFor="auto-capture" 
                          className="text-sm font-medium text-slate-700 cursor-pointer flex items-center gap-2"
                        >
                          <Zap className="w-4 h-4 text-orange-600" />
                          Execute API call to capture real response
                        </Label>
                        <p className="text-xs text-slate-500 mt-1">
                          Automatically make the API request and use the actual response for generating precise assertions (works with cURL input)
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Advanced Settings */}
                  <div className="space-y-3 pt-2 pb-2 border-t border-slate-200">
                    <details className="group">
                      <summary className="cursor-pointer text-sm font-semibold text-slate-700 flex items-center gap-2">
                        <span>⚙️ Advanced Settings</span>
                        <span className="text-xs text-slate-500">(optional)</span>
                      </summary>
                      <div className="mt-3 space-y-4 pl-6">
                        {/* Temperature Slider */}
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Label htmlFor="temperature" className="text-sm font-medium text-slate-700">
                              AI Temperature: {temperature.toFixed(1)}
                            </Label>
                          </div>
                          <Input
                            id="temperature"
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            className="w-full"
                          />
                          <p className="text-xs text-slate-500">
                            Lower = more focused/deterministic, Higher = more creative/varied (default: 0.1)
                          </p>
                        </div>

                        {/* SSL Verification Checkbox */}
                        <div className="flex items-start space-x-3">
                          <Checkbox 
                            id="verify-ssl" 
                            checked={verifySsl}
                            onCheckedChange={setVerifySsl}
                            className="mt-1"
                          />
                          <div className="flex-1">
                            <Label 
                              htmlFor="verify-ssl" 
                              className="text-sm font-medium text-slate-700 cursor-pointer"
                            >
                              Verify SSL certificates (auto-capture)
                            </Label>
                            <p className="text-xs text-slate-500 mt-1">
                              Disable for self-signed certificates or local development (not recommended for production)
                            </p>
                          </div>
                        </div>
                      </div>
                    </details>
                  </div>

                  {/* Generate Button */}
                  <Button
                    data-testid="generate-button"
                    onClick={handleGenerate}
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white font-semibold py-6 rounded-xl shadow-lg transition-all"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Generating Comprehensive Tests...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5 mr-2" />
                        Generate Test Scripts
                      </>
                    )}
                  </Button>

                  {/* Error Display */}
                  {error && (
                    <Alert data-testid="error-alert" variant="destructive" className="border-red-300 bg-red-50">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        {typeof error === 'string' ? error : JSON.stringify(error)}
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Right Column - Output */}
            <div className="space-y-6">
              <Card data-testid="output-card" className="border-slate-200 shadow-lg">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Code className="w-5 h-5 text-orange-600" />
                        Generated Tests
                      </CardTitle>
                      <CardDescription>Comprehensive test suite with security checks</CardDescription>
                    </div>
                    {result && (
                      <div className="flex items-center gap-2">
                        {copySuccess && (
                          <span className="text-sm text-green-600 font-medium flex items-center gap-1 animate-in fade-in slide-in-from-right-2">
                            <CheckCircle2 className="w-4 h-4" />
                            Copied!
                          </span>
                        )}
                        <Button
                          data-testid="copy-button"
                          onClick={handleCopy}
                          size="sm"
                          variant="outline"
                          className="border-slate-300"
                        >
                          <Copy className="w-4 h-4" />
                        </Button>
                        <Button
                          data-testid="download-button"
                          onClick={handleDownload}
                          size="sm"
                          variant="outline"
                          className="border-slate-300"
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                      </div>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  {result ? (
                    <div className="space-y-4">
                      <Alert data-testid="success-alert" className="border-green-300 bg-green-50">
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                        <AlertDescription className="text-green-800">
                          Comprehensive test suite generated for <strong>{result.framework}</strong>
                        </AlertDescription>
                      </Alert>
                      {result.warnings && result.warnings.length > 0 && (
                        <Alert className="border-orange-300 bg-orange-50">
                          <AlertCircle className="h-4 w-4 text-orange-600" />
                          <AlertDescription className="text-orange-800">
                            <div className="font-semibold mb-2">⚠️ Code Quality Warnings:</div>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                              {result.warnings.map((warning, idx) => (
                                <li key={idx}>{warning}</li>
                              ))}
                            </ul>
                            <div className="mt-2 text-xs">The code may still work, but consider reviewing these issues.</div>
                          </AlertDescription>
                        </Alert>
                      )}
                      <div className="relative">
                        <pre
                          data-testid="generated-code"
                          className="bg-slate-900 text-slate-100 p-6 rounded-xl overflow-x-auto text-sm font-mono max-h-[600px] overflow-y-auto"
                        >
                          <code>{result.test_script}</code>
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-16 text-slate-400">
                      <Code className="w-16 h-16 mx-auto mb-4 opacity-30" />
                      <p className="text-lg font-medium">No tests generated yet</p>
                      <p className="text-sm mt-2">Configure your settings and click Generate</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Enhanced Info Card */}
              <Card className="border-orange-200 bg-orange-50/50">
                <CardHeader>
                  <CardTitle className="text-base text-orange-900 flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    Comprehensive Test Coverage
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="space-y-2">
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Positive tests with schema validation</span>
                      </div>
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Negative tests (4xx errors)</span>
                      </div>
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Security tests (injections, XSS)</span>
                      </div>
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Authentication tests (401/403)</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Authorization validation</span>
                      </div>
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Edge cases & boundaries</span>
                      </div>
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>Rate limiting tests</span>
                      </div>
                      <div className="flex items-start gap-2 text-orange-800">
                        <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>REST & GraphQL support</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 pt-3 border-t border-orange-200">
                    <div className="flex items-center gap-2 text-xs text-orange-700">
                      <Zap className="w-3 h-3" />
                      <span className="font-semibold">Zero TODOs Promise:</span>
                      <span>Production-ready code with no placeholders</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-orange-700 mt-1">
                      <Shield className="w-3 h-3" />
                      <span className="font-semibold">Smart Generation:</span>
                      <span>Auto-validation & chunked processing for large tests</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
