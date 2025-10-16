import { useState, useEffect, useRef } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertCircle, CheckCircle2, Code, Copy, Download, Loader2, Sparkles, Database, Zap, HelpCircle, ExternalLink, Play, Edit3, Save, Eye, X, Terminal, Clock, Activity, TrendingUp, Undo, Redo, RefreshCw, Square, XCircle, History, Trash2, LayoutDashboard, Pencil, Check } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import Editor from '@monaco-editor/react';
import { AuthProvider, useAuth, UserMenu, AuthModal } from './components/Auth';
import { Dashboard } from './components/Dashboard';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
const USE_V2 = process.env.REACT_APP_USE_V2 !== 'false'; // Default to V2
const API = USE_V2 ? `${BACKEND_URL}/api/v2` : `${BACKEND_URL}/api`;

// Main App component - this is the inner component that uses auth context
function AppContent() {
  const { user, token } = useAuth();
  const [showDashboard, setShowDashboard] = useState(false);

  const [inputType, setInputType] = useState("curl");
  const [curlData, setCurlData] = useState("");
  const [harData, setHarData] = useState("");
  const [textData, setTextData] = useState("");
  const [exampleResponse, setExampleResponse] = useState("");
  const [autoCapture, setAutoCapture] = useState(true); // Default to true
  const [aiProvider, setAiProvider] = useState(() => localStorage.getItem("aiProvider") || "openai");
  const [aiModel, setAiModel] = useState(() => localStorage.getItem("aiModel") || "gpt-4o");
  const [aiApiKey, setAiApiKey] = useState(() => {
    // Get API key from backend if user is authenticated
    if (user?.api_keys) {
      const providerKey = user.api_keys[aiProvider];
      if (providerKey) return providerKey;
    }
    return localStorage.getItem("aiApiKey") || "";
  });
  const [testFramework, setTestFramework] = useState("jest");
  const [testProfile, setTestProfile] = useState("full_regression"); // Add test profile
  const [moduleSystem, setModuleSystem] = useState("commonjs"); // Default to CommonJS
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [temperature, setTemperature] = useState(0.1); // V2 uses lower temperature for deterministic output
  const [verifySsl, setVerifySsl] = useState(true);
  const [copySuccess, setCopySuccess] = useState(false);
  const [showEditor, setShowEditor] = useState(false);
  const [editedScript, setEditedScript] = useState("");
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResults, setExecutionResults] = useState(null);
  const [savedScripts, setSavedScripts] = useState([]);
  const [executionLogs, setExecutionLogs] = useState([]);
  const [executionProgress, setExecutionProgress] = useState(0);
  const [executionStatus, setExecutionStatus] = useState(null);
  const [editingScriptId, setEditingScriptId] = useState(null);
  const [editingScriptName, setEditingScriptName] = useState("");
  const [showSavedScripts, setShowSavedScripts] = useState(false);
  const [expandedTests, setExpandedTests] = useState({});
  const ws = useRef(null);
  const editorRef = useRef(null);
  const logsEndRef = useRef(null);

  // Load saved scripts from backend if user is authenticated
  useEffect(() => {
    if (token && user) {
      loadUserScripts();
    } else {
      // Fall back to localStorage for non-authenticated users
      const saved = localStorage.getItem('saved_test_scripts');
      setSavedScripts(saved ? JSON.parse(saved) : []);
    }
  }, [token, user]);

  const loadUserScripts = async () => {
    try {
      const response = await axios.get(`${API}/user/scripts`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (response.data.scripts) {
        // Ensure each script has the required fields
        const scripts = response.data.scripts.map(s => ({
          id: s.id || s._id,
          name: s.name || 'Unnamed Script',
          script: s.script || s.content || '', // Check both 'script' and 'content' fields
          framework: s.framework || 'jest',
          timestamp: s.timestamp || s.created_at || new Date().toISOString()
        }));
        setSavedScripts(scripts);
        console.log("Loaded scripts from backend:", scripts.length);
      }
    } catch (err) {
      console.error("Failed to load scripts:", err);
      // Fall back to localStorage
      const saved = localStorage.getItem('saved_test_scripts');
      if (saved) {
        const scripts = JSON.parse(saved);
        setSavedScripts(scripts);
        console.log("Loaded scripts from localStorage:", scripts.length);
      } else {
        setSavedScripts([]);
      }
    }
  };

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

  const handleEdit = () => {
    setEditedScript(result?.test_script || "");
    setShowEditor(true);
  };

  const handleSaveScript = async () => {
    // Prompt user for a script name
    const scriptName = prompt("Enter a name for this test script:", `Test Script ${new Date().toLocaleDateString()}`);

    if (!scriptName) {
      // User cancelled
      return;
    }

    const scriptToSave = {
      id: Date.now(),
      name: scriptName,
      script: editedScript || result?.test_script,
      framework: result?.framework || testFramework,
      timestamp: new Date().toISOString()
    };

    // Save to backend if user is authenticated
    if (token) {
      try {
        const response = await axios.post(`${API}/user/scripts`, scriptToSave, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (response.data.success && response.data.script) {
          // Add the saved script directly to the list with backend response
          const savedScript = response.data.script;
          setSavedScripts(prev => [...prev, savedScript]);
        } else {
          // Reload scripts from backend as fallback
          await loadUserScripts();
        }
      } catch (err) {
        console.error("Failed to save script to backend:", err);
        // Fall back to localStorage
        const updatedScripts = [...savedScripts, scriptToSave];
        setSavedScripts(updatedScripts);
        localStorage.setItem('saved_test_scripts', JSON.stringify(updatedScripts));
      }
    } else {
      // Save to localStorage for non-authenticated users
      const updatedScripts = [...savedScripts, scriptToSave];
      setSavedScripts(updatedScripts);
      localStorage.setItem('saved_test_scripts', JSON.stringify(updatedScripts));
    }

    // Update result with edited script
    if (editedScript && result) {
      setResult({ ...result, test_script: editedScript });
    }
    setShowEditor(false);

    // Show saved scripts section immediately
    setShowSavedScripts(true);
  };

  const handleRunTests = async () => {
    const scriptToRun = editedScript || result?.test_script;
    if (!scriptToRun) return;

    setIsExecuting(true);
    setExecutionResults(null);
    setExecutionLogs([]);
    setExecutionProgress(0);
    setExecutionStatus('starting');
    setError(null); // Clear any previous errors

    // Add initial log
    setExecutionLogs([{
      type: 'info',
      message: 'Starting test execution...',
      timestamp: new Date().toISOString()
    }]);

    // Try WebSocket first with a connection timeout
    let wsConnected = false;
    let wsTimeout = null;

    try {
      // Try WebSocket for real-time updates
      const wsUrl = `ws://localhost:8000/api/v2/ws/execute-tests`;
      ws.current = new WebSocket(wsUrl);

      // Set a timeout for WebSocket connection
      wsTimeout = setTimeout(() => {
        if (!wsConnected) {
          console.log("WebSocket connection timeout, falling back to REST");
          if (ws.current) {
            ws.current.close();
          }
          executeTestsREST(scriptToRun);
        }
      }, 3000); // 3 second timeout for WS connection

      ws.current.onopen = () => {
        wsConnected = true;
        clearTimeout(wsTimeout);
        console.log("WebSocket connected for test execution");
        setExecutionLogs(prev => [...prev, {
          type: 'info',
          message: 'Connected to execution service via WebSocket',
          timestamp: new Date().toISOString()
        }]);

        const request = {
          action: "execute",
          test_code: scriptToRun,
          framework: result?.framework || testFramework,
          timeout: 300,
          api_base_url: "http://localhost:3000",
          environment: null  // Use null instead of empty object
        };
        ws.current.send(JSON.stringify(request));
        setExecutionStatus('running');
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case 'progress':
            setExecutionLogs(prev => [...prev, { type: 'info', message: data.message, timestamp: new Date().toISOString() }]);
            setExecutionProgress(prev => Math.min(prev + 10, 90));
            break;
          case 'console':
            setExecutionLogs(prev => [...prev, { type: 'console', message: data.line, timestamp: new Date().toISOString() }]);
            break;
          case 'result':
            // Ensure test_results is properly set
            const resultData = {
              ...data.data,
              test_results: data.data.test_results || []
            };
            setExecutionResults(resultData);
            setExecutionProgress(100);

            // Determine status based on test failures
            const hasFailedTests = resultData.test_results && resultData.test_results.some(test => test.status === 'failed');
            setExecutionStatus(hasFailedTests ? 'failed' : 'success');
            setIsExecuting(false);
            break;
          case 'error':
            setError(data.message);
            setExecutionStatus('error');
            setIsExecuting(false);
            break;
        }
      };

      ws.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        clearTimeout(wsTimeout);
        if (!wsConnected) {
          setExecutionLogs(prev => [...prev, {
            type: 'info',
            message: 'WebSocket not available, using REST API...',
            timestamp: new Date().toISOString()
          }]);
          // Fallback to REST API
          executeTestsREST(scriptToRun);
        }
      };

      ws.current.onclose = () => {
        console.log("WebSocket disconnected");
        clearTimeout(wsTimeout);
        if (!wsConnected) {
          // If never connected, try REST
          executeTestsREST(scriptToRun);
        }
      };

    } catch (err) {
      console.error("Error setting up WebSocket:", err);
      clearTimeout(wsTimeout);
      // Fallback to REST API
      executeTestsREST(scriptToRun);
    }
  };

  const executeTestsREST = async (scriptToRun) => {
    try {
      setExecutionLogs(prev => [...prev, {
        type: 'info',
        message: 'Connecting to execution service via REST API...',
        timestamp: new Date().toISOString()
      }]);

      // Show that we're executing
      setExecutionStatus('running');
      setExecutionProgress(10);

      const response = await axios.post(`${API}/execute-tests`, {
        test_code: scriptToRun,
        framework: result?.framework || testFramework,
        timeout: 300,
        api_base_url: "http://localhost:3000", // Add base URL for API calls in tests
        environment: null  // Use null for environment
      }, {
        timeout: 305000, // 5 seconds more than execution timeout
        validateStatus: function (status) {
          return status < 500; // Accept any status code less than 500
        }
      });

      if (response.data.success) {
        // Ensure test_results is properly set
        const results = {
          ...response.data,
          test_results: response.data.test_results || []
        };
        setExecutionResults(results);

        // Determine status based on test results
        const hasFailures = results.test_results && results.test_results.some(test => test.status === 'failed');
        setExecutionStatus(hasFailures ? 'failed' : 'completed');
        setExecutionProgress(100);
        setExecutionLogs(prev => [...prev, {
          type: 'info',
          message: 'Test execution completed successfully',
          timestamp: new Date().toISOString()
        }]);
      } else {
        const errorMsg = response.data.error || "Execution failed";
        setError(errorMsg);
        setExecutionStatus('error');
        setExecutionLogs(prev => [...prev, {
          type: 'error',
          message: errorMsg,
          timestamp: new Date().toISOString()
        }]);
      }
    } catch (err) {
      let errorMessage = "Test execution failed";

      if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
        errorMessage = "Test execution timed out after 300 seconds. Your tests may be taking too long to complete.";
      } else if (err.response?.status === 500) {
        errorMessage = "Server error during test execution. Please check the backend server is running.";
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
      setExecutionStatus('error');
      setExecutionLogs(prev => [...prev, {
        type: 'error',
        message: errorMessage,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsExecuting(false);
    }
  };

  const loadSavedScript = (script) => {
    // Ensure script.script exists and is not empty
    if (!script || !script.script) {
      console.error("Invalid script data:", script);
      setError("Unable to load script - script content is missing");
      return;
    }

    // Set the result to display the script
    setResult({
      test_script: script.script,
      framework: script.framework || 'jest',
      success: true
    });

    // Also set the edited script if editing
    setEditedScript(script.script);

    // Clear any previous errors
    setError("");

    console.log("Loaded script:", script.name, "Framework:", script.framework);
  };

  const renameSavedScript = async (scriptId, newName) => {
    if (!newName.trim()) {
      // Cancel rename if name is empty
      setEditingScriptId(null);
      setEditingScriptName("");
      return;
    }

    // Update in backend if user is authenticated
    if (token) {
      try {
        const scriptToUpdate = savedScripts.find(s => s.id === scriptId);
        if (scriptToUpdate) {
          const response = await axios.put(`${API}/user/scripts/${scriptId}`, {
            ...scriptToUpdate,
            name: newName
          }, {
            headers: { Authorization: `Bearer ${token}` }
          });
          if (response.data.success) {
            // Reload scripts from backend
            await loadUserScripts();
          }
        }
      } catch (err) {
        console.error("Failed to rename script in backend:", err);
        // Fall back to local update
        const updatedScripts = savedScripts.map(script =>
          script.id === scriptId ? { ...script, name: newName } : script
        );
        setSavedScripts(updatedScripts);
        localStorage.setItem('saved_test_scripts', JSON.stringify(updatedScripts));
      }
    } else {
      // Update in localStorage for non-authenticated users
      const updatedScripts = savedScripts.map(script =>
        script.id === scriptId ? { ...script, name: newName } : script
      );
      setSavedScripts(updatedScripts);
      localStorage.setItem('saved_test_scripts', JSON.stringify(updatedScripts));
    }

    // Clear editing state
    setEditingScriptId(null);
    setEditingScriptName("");
  };

  const deleteSavedScript = async (scriptId, e) => {
    e.stopPropagation(); // Prevent triggering the load action when clicking delete

    // Delete from backend if user is authenticated
    if (token) {
      try {
        const response = await axios.delete(`${API}/user/scripts/${scriptId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (response.data.success) {
          // Reload scripts from backend
          await loadUserScripts();
        }
      } catch (err) {
        console.error("Failed to delete script from backend:", err);
        // Fall back to local update
        const updatedScripts = savedScripts.filter(script => script.id !== scriptId);
        setSavedScripts(updatedScripts);
        localStorage.setItem('saved_test_scripts', JSON.stringify(updatedScripts));
      }
    } else {
      // Delete from localStorage for non-authenticated users
      const updatedScripts = savedScripts.filter(script => script.id !== scriptId);
      setSavedScripts(updatedScripts);
      localStorage.setItem('saved_test_scripts', JSON.stringify(updatedScripts));
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

  // Show dashboard if requested
  if (showDashboard) {
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
                  <Button
                    onClick={() => setShowDashboard(false)}
                    variant="outline"
                    size="sm"
                    className="border-slate-300"
                  >
                    <Code className="w-4 h-4 mr-1" />
                    Test Generator
                  </Button>
                  <UserMenu />
                </div>
              </div>
            </div>
          </header>
          <Dashboard />
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      {/* Show mandatory auth modal if user is not authenticated */}
      {!token && (
        <AuthModal
          isOpen={true}
          mandatory={true}
          onClose={() => {}} // Modal cannot be closed
          onSuccess={() => {
            // Auth successful, modal will auto-hide when token is set
          }}
        />
      )}

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
                <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                  <Database className="w-3 h-3 mr-1" />
                  REST + GraphQL
                </Badge>
                {user && (
                  <Button
                    onClick={() => setShowDashboard(true)}
                    variant="outline"
                    size="sm"
                    className="border-slate-300"
                  >
                    <LayoutDashboard className="w-4 h-4 mr-1" />
                    Dashboard
                  </Button>
                )}
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
                {user && <UserMenu />}
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
                          data-testid="edit-button"
                          onClick={handleEdit}
                          size="sm"
                          variant="default"
                          className="bg-blue-600 hover:bg-blue-700"
                        >
                          <Edit3 className="w-4 h-4 mr-1" />
                          Edit
                        </Button>
                        <Button
                          data-testid="run-button"
                          onClick={handleRunTests}
                          size="sm"
                          variant="default"
                          className="bg-green-600 hover:bg-green-700"
                          disabled={isExecuting}
                        >
                          {isExecuting ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                              Running...
                            </>
                          ) : (
                            <>
                              <Play className="w-4 h-4 mr-1" />
                              Run
                            </>
                          )}
                        </Button>
                        <Button
                          data-testid="save-button"
                          onClick={handleSaveScript}
                          size="sm"
                          variant="outline"
                          className="border-slate-300"
                        >
                          <Save className="w-4 h-4" />
                        </Button>
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

              {/* Inline Script Editor */}
              {showEditor && (
                <Card className="border-slate-200 shadow-lg animate-in slide-in-from-bottom-2">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Edit3 className="w-5 h-5 text-blue-600" />
                          Script Editor
                        </CardTitle>
                        <CardDescription>Edit and customize your {result?.framework || testFramework} test script</CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          onClick={() => setEditedScript(result?.test_script || "")}
                          size="sm"
                          variant="outline"
                          className="border-slate-300"
                        >
                          <RefreshCw className="w-4 h-4 mr-1" />
                          Reset
                        </Button>
                        <Button
                          onClick={handleSaveScript}
                          size="sm"
                          variant="default"
                          className="bg-green-600 hover:bg-green-700"
                        >
                          <Save className="w-4 h-4 mr-1" />
                          Save & Apply
                        </Button>
                        <Button
                          onClick={() => setShowEditor(false)}
                          size="sm"
                          variant="ghost"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="border rounded-lg overflow-hidden">
                      <Editor
                        height="400px"
                        language={
                          ["jest", "mocha", "cypress"].includes(result?.framework || testFramework) ? "javascript" :
                          ["pytest", "requests"].includes(result?.framework || testFramework) ? "python" :
                          ["junit", "testng", "restassured"].includes(result?.framework || testFramework) ? "java" :
                          "javascript"
                        }
                        value={editedScript}
                        onChange={setEditedScript}
                        onMount={(editor) => { editorRef.current = editor; }}
                        theme="vs-dark"
                        options={{
                          minimap: { enabled: false },
                          fontSize: 14,
                          lineNumbers: 'on',
                          scrollBeyondLastLine: false,
                          automaticLayout: true,
                          formatOnPaste: true,
                          formatOnType: true,
                          wordWrap: 'on',
                          folding: true,
                          bracketPairColorization: { enabled: true },
                          suggest: {
                            showKeywords: true,
                            showSnippets: true,
                          },
                        }}
                      />
                    </div>
                    <div className="mt-4 flex items-center justify-between text-sm text-slate-600">
                      <div>
                        Lines: <strong>{editedScript.split('\n').length}</strong> |
                        Characters: <strong>{editedScript.length}</strong>
                      </div>
                      <Button
                        onClick={handleRunTests}
                        size="sm"
                        variant="default"
                        className="bg-blue-600 hover:bg-blue-700"
                        disabled={isExecuting}
                      >
                        {isExecuting ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                            Running...
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 mr-1" />
                            Run Edited Script
                          </>
                        )}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Test Execution Panel */}
              {(isExecuting || executionResults || executionLogs.length > 0) && (
                <Card className="border-slate-200 shadow-lg">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Terminal className="w-5 h-5 text-green-600" />
                          Test Execution
                        </CardTitle>
                        <CardDescription>
                          {isExecuting ? "Executing tests..." : "Test execution completed"}
                        </CardDescription>
                      </div>
                      <div className="flex items-center gap-2">
                        {executionStatus === 'running' && (
                          <>
                            <Badge className="bg-blue-500">
                              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                              Running
                            </Badge>
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={() => {
                                if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                                  ws.current.close();
                                }
                                setIsExecuting(false);
                                setExecutionStatus('cancelled');
                                setExecutionLogs(prev => [...prev, {
                                  type: 'error',
                                  message: 'Test execution cancelled by user',
                                  timestamp: new Date().toISOString()
                                }]);
                              }}
                            >
                              <Square className="w-4 h-4 mr-1" />
                              Cancel
                            </Button>
                          </>
                        )}
                        {executionStatus === 'success' && (
                          <Badge className="bg-green-500">
                            <CheckCircle2 className="w-3 h-3 mr-1" />
                            Passed
                          </Badge>
                        )}
                        {executionStatus === 'failed' && (
                          <Badge className="bg-red-500">
                            <XCircle className="w-3 h-3 mr-1" />
                            Failed
                          </Badge>
                        )}
                        {executionStatus === 'error' && (
                          <Badge className="bg-red-500">
                            <AlertCircle className="w-3 h-3 mr-1" />
                            Error
                          </Badge>
                        )}
                        {executionStatus === 'cancelled' && (
                          <Badge className="bg-gray-500">
                            <Square className="w-3 h-3 mr-1" />
                            Cancelled
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {/* Error Display */}
                    {executionStatus === 'error' && (
                      <Alert variant="destructive" className="mb-4 border-red-300 bg-red-50">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-red-800">
                          <div className="font-semibold mb-1">Test Execution Failed</div>
                          {error || "Test execution timed out after 300 seconds. Please check your test code and try again."}
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Progress Bar */}
                    {isExecuting && (
                      <div className="space-y-2 mb-4">
                        <Progress value={executionProgress} className="h-2" />
                        <p className="text-xs text-gray-500">Executing tests... {executionProgress}%</p>
                      </div>
                    )}

                    {/* Results Summary */}
                    {executionResults && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <Card>
                          <CardContent className="p-4">
                            <div className="text-2xl font-bold">{executionResults.total_tests || 0}</div>
                            <p className="text-xs text-gray-500">Total Tests</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4">
                            <div className="text-2xl font-bold text-green-600">
                              {executionResults.passed_tests || 0}
                            </div>
                            <p className="text-xs text-gray-500">Passed</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4">
                            <div className="text-2xl font-bold text-red-600">
                              {executionResults.failed_tests || 0}
                            </div>
                            <p className="text-xs text-gray-500">Failed</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4">
                            <div className="text-2xl font-bold">
                              {executionResults.duration ? `${executionResults.duration.toFixed(2)}s` : '0s'}
                            </div>
                            <p className="text-xs text-gray-500">Duration</p>
                          </CardContent>
                        </Card>
                      </div>
                    )}

                    {/* Console Logs */}
                    <Card>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm">Console Output</CardTitle>
                          {executionLogs.length > 0 && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => {
                                const logContent = executionLogs.map(log =>
                                  `[${new Date(log.timestamp).toLocaleTimeString()}] ${log.message}`
                                ).join('\n');
                                navigator.clipboard.writeText(logContent);
                              }}
                            >
                              <Copy className="w-4 h-4 mr-1" />
                              Copy Logs
                            </Button>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="bg-gray-900 text-gray-100 rounded-lg p-4 font-mono text-xs h-64 overflow-y-auto">
                          {executionLogs.length === 0 ? (
                            <div className="text-gray-500">Waiting for output...</div>
                          ) : (
                            executionLogs.map((log, idx) => (
                              <div
                                key={idx}
                                className={`mb-2 ${
                                  log.type === 'error' ? 'text-red-400 font-semibold' :
                                  log.type === 'info' ? 'text-blue-400' :
                                  'text-gray-300'
                                }`}
                              >
                                <span className="text-gray-500 mr-2">
                                  [{new Date(log.timestamp).toLocaleTimeString()}]
                                </span>
                                {log.type === 'error' && <span className="mr-2">❌</span>}
                                {log.type === 'info' && <span className="mr-2">ℹ️</span>}
                                {log.message}
                              </div>
                            ))
                          )}
                          <div ref={logsEndRef} />
                        </div>
                      </CardContent>
                    </Card>

                    {/* Individual Test Results */}
                    {executionResults && executionResults.test_results && executionResults.test_results.length > 0 && (
                      <Card className="mt-4">
                        <CardHeader>
                          <CardTitle className="text-sm">Test Results</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            {executionResults.test_results.map((test, idx) => {
                              const isExpanded = expandedTests[idx] !== undefined ? expandedTests[idx] : test.status === 'failed';
                              // Debug logging for test error details
                              if (test.status === 'failed' && test.error) {
                                console.log(`Test ${idx} failed with error:`, test.error);
                              }
                              return (
                                <div key={idx} className="border rounded-lg overflow-hidden">
                                  <div
                                    className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                                    onClick={() => setExpandedTests(prev => ({ ...prev, [idx]: !isExpanded }))}
                                  >
                                    <div className="flex items-center gap-2">
                                      {test.status === 'passed' ? (
                                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                                      ) : (
                                        <XCircle className="w-4 h-4 text-red-500" />
                                      )}
                                      <span className="text-sm font-medium">{test.name}</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Clock className="w-3 h-3 text-gray-500" />
                                      <span className="text-xs text-gray-500">
                                        {test.duration ? `${test.duration.toFixed(2)}s` : '0s'}
                                      </span>
                                      {/* Chevron indicator for expansion */}
                                      <svg
                                        className={`w-4 h-4 text-gray-500 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                      </svg>
                                    </div>
                                  </div>
                                  {/* Show details when expanded */}
                                  {isExpanded && (
                                    <>
                                      {/* Show failure details if test failed */}
                                      {test.status === 'failed' && (
                                        <div className="p-3 bg-red-50 border-t border-red-100">
                                          <div className="space-y-2">
                                            <div className="text-sm font-medium text-red-800">Failure Details:</div>
                                            {test.error ? (
                                              <>
                                                {test.error.message && (
                                                  <div className="text-sm text-red-700">
                                                    <span className="font-medium">Error: </span>
                                                    {test.error.message}
                                                  </div>
                                                )}
                                                {(test.error.expected !== undefined || test.error.actual !== undefined) && (
                                                  <div className="space-y-1">
                                                    {test.error.expected !== undefined && (
                                                      <div className="text-sm text-red-700">
                                                        <span className="font-medium">Expected: </span>
                                                        <code className="bg-red-100 px-1 py-0.5 rounded">
                                                          {typeof test.error.expected === 'object'
                                                            ? JSON.stringify(test.error.expected, null, 2)
                                                            : String(test.error.expected)}
                                                        </code>
                                                      </div>
                                                    )}
                                                    {test.error.actual !== undefined && (
                                                      <div className="text-sm text-red-700">
                                                        <span className="font-medium">Actual: </span>
                                                        <code className="bg-red-100 px-1 py-0.5 rounded">
                                                          {typeof test.error.actual === 'object'
                                                            ? JSON.stringify(test.error.actual, null, 2)
                                                            : String(test.error.actual)}
                                                        </code>
                                                      </div>
                                                    )}
                                                  </div>
                                                )}
                                                {test.error.stack && (
                                                  <details className="mt-2">
                                                    <summary className="text-xs text-red-600 cursor-pointer hover:text-red-700">
                                                      Show stack trace
                                                    </summary>
                                                    <pre className="mt-2 text-xs text-red-600 bg-red-100 p-2 rounded overflow-x-auto">
                                                      {test.error.stack}
                                                    </pre>
                                                  </details>
                                                )}
                                              </>
                                            ) : test.error_message ? (
                                              // Fallback to error_message if error object is not present
                                              <div className="text-sm text-red-700">
                                                {test.error_message}
                                              </div>
                                            ) : (
                                              // Final fallback
                                              <div className="text-sm text-red-700">
                                                Test failed (no error details available)
                                              </div>
                                            )}
                                          </div>
                                        </div>
                                      )}
                                      {/* Show assertion details for passed tests */}
                                      {test.status === 'passed' && test.assertions && test.assertions.length > 0 && (
                                        <div className="p-2 bg-green-50 border-t border-green-100 space-y-1">
                                          <div className="text-sm font-medium text-green-800 mb-1">Assertions:</div>
                                          {test.assertions.map((assertion, aIdx) => (
                                            <div key={aIdx} className="text-xs text-green-700">
                                              ✓ {assertion}
                                            </div>
                                          ))}
                                        </div>
                                      )}
                                      {/* Show success message for passed tests without specific assertions */}
                                      {test.status === 'passed' && (!test.assertions || test.assertions.length === 0) && (
                                        <div className="p-2 bg-green-50 border-t border-green-100">
                                          <div className="text-sm text-green-700">✓ Test passed successfully</div>
                                        </div>
                                      )}
                                    </>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Saved Scripts - Always show when there are saved scripts or after saving */}
              {(savedScripts.length > 0 || showSavedScripts) && (
                <Card className="border-slate-200 shadow-lg">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <History className="w-5 h-5 text-purple-600" />
                          Saved Scripts
                        </CardTitle>
                        <CardDescription>Previously saved test scripts</CardDescription>
                      </div>
                      <Badge variant="outline">
                        {savedScripts.length} saved
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {savedScripts.length === 0 ? (
                      <div className="text-center py-8 text-gray-400">
                        <History className="w-12 h-12 mx-auto mb-2 opacity-30" />
                        <p className="text-sm">No saved scripts yet</p>
                        <p className="text-xs mt-1">Save a script to see it here</p>
                      </div>
                    ) : (
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {savedScripts.slice(-5).reverse().map((script) => (
                          <div
                            key={script.id}
                            className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50"
                          >
                            <div
                              className="flex-1 cursor-pointer"
                              onClick={() => !editingScriptId && loadSavedScript(script)}
                            >
                              {editingScriptId === script.id ? (
                                <div className="flex items-center gap-2">
                                  <Input
                                    type="text"
                                    value={editingScriptName}
                                    onChange={(e) => setEditingScriptName(e.target.value)}
                                    onKeyPress={(e) => {
                                      if (e.key === 'Enter') {
                                        renameSavedScript(script.id, editingScriptName);
                                      } else if (e.key === 'Escape') {
                                        setEditingScriptId(null);
                                        setEditingScriptName("");
                                      }
                                    }}
                                    onBlur={() => renameSavedScript(script.id, editingScriptName)}
                                    className="h-7 text-sm"
                                    autoFocus
                                    onClick={(e) => e.stopPropagation()}
                                  />
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="h-7 w-7 p-0"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      renameSavedScript(script.id, editingScriptName);
                                    }}
                                  >
                                    <Check className="w-4 h-4 text-green-600" />
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="h-7 w-7 p-0"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setEditingScriptId(null);
                                      setEditingScriptName("");
                                    }}
                                  >
                                    <X className="w-4 h-4 text-red-600" />
                                  </Button>
                                </div>
                              ) : (
                                <>
                                  <div className="font-medium text-sm">{script.name}</div>
                                  <div className="text-xs text-gray-500">
                                    {script.timestamp ? new Date(script.timestamp).toLocaleString() : 'Recently saved'} • {script.framework || 'jest'}
                                  </div>
                                </>
                              )}
                            </div>
                            {editingScriptId !== script.id && (
                              <div className="flex items-center gap-1">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="h-8 w-8 p-0"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setEditingScriptId(script.id);
                                    setEditingScriptName(script.name);
                                  }}
                                >
                                  <Pencil className="w-4 h-4 text-gray-600" />
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    loadSavedScript(script);
                                  }}
                                >
                                  <Eye className="w-4 h-4 mr-1" />
                                  Load
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200 hover:border-red-300"
                                  onClick={(e) => deleteSavedScript(script.id, e)}
                                >
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

// Main App component with AuthProvider wrapper
function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
