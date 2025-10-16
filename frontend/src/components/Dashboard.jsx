import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from './Auth';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  Calendar,
  Clock,
  Code,
  Database,
  FileCode,
  Key,
  Activity,
  TrendingUp,
  User,
  Shield,
  Settings,
  Save,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Play,
  Download,
  Trash2,
  Eye,
  EyeOff,
  RefreshCw,
  History,
  Zap,
  Award,
  Target,
  Users,
  Lock
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

// ============================================================================
// USER DASHBOARD
// ============================================================================

export const Dashboard = () => {
  const { user, updateProfile } = useAuth();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [executionHistory, setExecutionHistory] = useState([]);
  const [preferences, setPreferences] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      // Fetch user statistics
      const [statsRes, historyRes, prefsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/v2/user/stats`),
        axios.get(`${API_BASE}/api/v2/user/execution-history?limit=10`),
        axios.get(`${API_BASE}/api/v2/user/preferences`)
      ]);

      setStats(statsRes.data);
      setExecutionHistory(historyRes.data.history || []);
      setPreferences(prefsRes.data.preferences);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-orange-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Welcome back, {user?.name}!</h1>
          <p className="text-gray-600">Here's your testing activity overview</p>
        </div>
        <Button
          onClick={fetchDashboardData}
          variant="outline"
          className="gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </Button>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tests Generated</CardTitle>
            <Code className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.totalTests || 0}</div>
            <p className="text-xs text-muted-foreground">
              +{stats?.testsThisMonth || 0} this month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Tests Executed</CardTitle>
            <Play className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.totalExecutions || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.successRate || 0}% success rate
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Saved Scripts</CardTitle>
            <Database className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.savedScripts || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats?.scriptsShared || 0} shared with team
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Keys</CardTitle>
            <Key className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.apiKeysStored || 0}</div>
            <p className="text-xs text-muted-foreground">
              Securely encrypted
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="overview">
            <BarChart3 className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="w-4 h-4 mr-2" />
            Execution History
          </TabsTrigger>
          <TabsTrigger value="profile">
            <User className="w-4 h-4 mr-2" />
            Profile
          </TabsTrigger>
          <TabsTrigger value="security">
            <Shield className="w-4 h-4 mr-2" />
            Security
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <UsageOverview stats={stats} />
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          <ExecutionHistory history={executionHistory} />
        </TabsContent>

        <TabsContent value="profile" className="space-y-4">
          <ProfileSettings user={user} preferences={preferences} />
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <SecuritySettings />
        </TabsContent>
      </Tabs>
    </div>
  );
};

// ============================================================================
// USAGE OVERVIEW COMPONENT
// ============================================================================

const UsageOverview = ({ stats }) => {
  return (
    <div className="space-y-4">
      {/* Activity Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Testing Activity</CardTitle>
          <CardDescription>Your test generation and execution over time</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-gray-400">
            {/* Placeholder for chart - you can integrate recharts or similar */}
            <Activity className="w-8 h-8 mr-2" />
            <span>Activity chart will be displayed here</span>
          </div>
        </CardContent>
      </Card>

      {/* Framework Usage */}
      <Card>
        <CardHeader>
          <CardTitle>Framework Usage</CardTitle>
          <CardDescription>Your most used testing frameworks</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {stats?.frameworkUsage?.map((framework) => (
              <div key={framework.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{framework.name}</span>
                  <span className="text-sm text-gray-500">{framework.count} tests</span>
                </div>
                <Progress value={framework.percentage} className="h-2" />
              </div>
            )) || (
              <p className="text-gray-500">No framework data available yet</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Recent Achievements */}
      <Card>
        <CardHeader>
          <CardTitle>Achievements</CardTitle>
          <CardDescription>Your testing milestones</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex flex-col items-center p-4 border rounded-lg">
              <Award className="w-8 h-8 text-yellow-500 mb-2" />
              <span className="text-xs font-medium">First Test</span>
            </div>
            <div className="flex flex-col items-center p-4 border rounded-lg">
              <Zap className="w-8 h-8 text-blue-500 mb-2" />
              <span className="text-xs font-medium">Speed Tester</span>
            </div>
            <div className="flex flex-col items-center p-4 border rounded-lg">
              <Target className="w-8 h-8 text-green-500 mb-2" />
              <span className="text-xs font-medium">100 Tests</span>
            </div>
            <div className="flex flex-col items-center p-4 border rounded-lg">
              <Shield className="w-8 h-8 text-purple-500 mb-2" />
              <span className="text-xs font-medium">Security Pro</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// ============================================================================
// EXECUTION HISTORY COMPONENT
// ============================================================================

const ExecutionHistory = ({ history }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Test Executions</CardTitle>
        <CardDescription>Your last 10 test execution results</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {history.length > 0 ? (
            history.map((execution) => (
              <div
                key={execution.id}
                className="flex items-center justify-between p-4 border rounded-lg"
              >
                <div className="flex items-center gap-4">
                  {execution.status === 'passed' ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                  ) : execution.status === 'failed' ? (
                    <XCircle className="w-5 h-5 text-red-500" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-yellow-500" />
                  )}
                  <div>
                    <p className="font-medium">{execution.scriptName}</p>
                    <p className="text-sm text-gray-500">
                      {execution.framework} • {new Date(execution.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-medium">
                    {execution.passedTests}/{execution.totalTests} passed
                  </p>
                  <p className="text-sm text-gray-500">{execution.duration}s</p>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-gray-500">
              <History className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No execution history yet</p>
              <p className="text-sm">Run some tests to see them here</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// PROFILE SETTINGS COMPONENT
// ============================================================================

const ProfileSettings = ({ user, preferences }) => {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    defaultFramework: preferences?.defaultFramework || 'jest',
    defaultAiProvider: preferences?.defaultAiProvider || 'openai',
    defaultAiModel: preferences?.defaultAiModel || 'gpt-4o'
  });

  const handleSave = async () => {
    setSaving(true);
    try {
      await axios.put(`${API_BASE}/api/v2/user/profile`, formData);
      setEditing(false);
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Profile Information</CardTitle>
              <CardDescription>Manage your account details</CardDescription>
            </div>
            <Button
              onClick={() => editing ? handleSave() : setEditing(true)}
              disabled={saving}
            >
              {saving ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : editing ? (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save
                </>
              ) : (
                <>
                  <Settings className="w-4 h-4 mr-2" />
                  Edit
                </>
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                disabled={!editing}
              />
            </div>
            <div className="space-y-2">
              <Label>Email</Label>
              <Input
                value={formData.email}
                disabled
                className="bg-gray-50"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label>Member Since</Label>
            <Input
              value={new Date(user?.created_at).toLocaleDateString()}
              disabled
              className="bg-gray-50"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Default Preferences</CardTitle>
          <CardDescription>Set your default testing preferences</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label>Default Framework</Label>
              <select
                className="w-full px-3 py-2 border rounded-lg"
                value={formData.defaultFramework}
                onChange={(e) => setFormData({...formData, defaultFramework: e.target.value})}
                disabled={!editing}
              >
                <option value="jest">Jest</option>
                <option value="mocha">Mocha</option>
                <option value="cypress">Cypress</option>
                <option value="pytest">Pytest</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label>Default AI Provider</Label>
              <select
                className="w-full px-3 py-2 border rounded-lg"
                value={formData.defaultAiProvider}
                onChange={(e) => setFormData({...formData, defaultAiProvider: e.target.value})}
                disabled={!editing}
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// ============================================================================
// SECURITY SETTINGS COMPONENT
// ============================================================================

const SecuritySettings = () => {
  const [showApiKeys, setShowApiKeys] = useState(false);
  const [apiKeys, setApiKeys] = useState({});
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    fetchSecurityData();
  }, []);

  const fetchSecurityData = async () => {
    setLoading(true);
    try {
      const [keysRes, sessionsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/v2/user/api-keys`),
        axios.get(`${API_BASE}/api/v2/user/sessions`)
      ]);

      setApiKeys(keysRes.data.apiKeys || {});
      setSessions(sessionsRes.data.sessions || []);
    } catch (error) {
      console.error('Failed to fetch security data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordReset = async () => {
    try {
      await axios.post(`${API_BASE}/api/v2/auth/reset-password-request`);
      alert('Password reset email sent!');
    } catch (error) {
      alert('Failed to send reset email');
    }
  };

  const handleRevokeSession = async (sessionId) => {
    try {
      await axios.delete(`${API_BASE}/api/v2/user/sessions/${sessionId}`);
      setSessions(sessions.filter(s => s.id !== sessionId));
    } catch (error) {
      alert('Failed to revoke session');
    }
  };

  return (
    <div className="space-y-4">
      {/* Password & Authentication */}
      <Card>
        <CardHeader>
          <CardTitle>Password & Authentication</CardTitle>
          <CardDescription>Manage your authentication settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Password</p>
              <p className="text-sm text-gray-500">Last changed: Never</p>
            </div>
            <Button onClick={handlePasswordReset} variant="outline">
              Reset Password
            </Button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Two-Factor Authentication</p>
              <p className="text-sm text-gray-500">Add an extra layer of security</p>
            </div>
            <Badge variant="outline">Coming Soon</Badge>
          </div>
        </CardContent>
      </Card>

      {/* API Keys */}
      <Card>
        <CardHeader>
          <CardTitle>API Keys</CardTitle>
          <CardDescription>Your encrypted API keys</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(apiKeys).map(([provider, keyData]) => (
              <div key={provider} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <Key className="w-4 h-4 text-gray-400" />
                  <div>
                    <p className="font-medium capitalize">{provider}</p>
                    <p className="text-sm text-gray-500">
                      {showApiKeys ? keyData.key : '••••••••••••••••'}
                    </p>
                  </div>
                </div>
                <Button
                  onClick={() => setShowApiKeys(!showApiKeys)}
                  variant="ghost"
                  size="sm"
                >
                  {showApiKeys ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Active Sessions */}
      <Card>
        <CardHeader>
          <CardTitle>Active Sessions</CardTitle>
          <CardDescription>Manage your active login sessions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {sessions.map((session) => (
              <div key={session.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div>
                  <p className="font-medium">{session.device}</p>
                  <p className="text-sm text-gray-500">
                    {session.location} • {new Date(session.lastActive).toLocaleString()}
                  </p>
                </div>
                {session.current ? (
                  <Badge className="bg-green-100 text-green-700">Current</Badge>
                ) : (
                  <Button
                    onClick={() => handleRevokeSession(session.id)}
                    variant="outline"
                    size="sm"
                    className="text-red-600"
                  >
                    Revoke
                  </Button>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};