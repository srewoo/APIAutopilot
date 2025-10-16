import React, { useState, useEffect, createContext, useContext } from 'react';
import axios from 'axios';
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
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  User,
  Mail,
  Lock,
  LogIn,
  UserPlus,
  LogOut,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Eye,
  EyeOff,
  Shield,
  KeyRound
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

// ============================================================================
// AUTH CONTEXT
// ============================================================================

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

// ============================================================================
// AXIOS INTERCEPTORS SETUP
// ============================================================================

const setupAxiosInterceptors = (token, logout) => {
  // Request interceptor to add token
  axios.interceptors.request.use(
    (config) => {
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => Promise.reject(error)
  );

  // Response interceptor to handle 401 errors
  axios.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        // Token expired or invalid
        logout();
      }
      return Promise.reject(error);
    }
  );
};

// ============================================================================
// AUTH PROVIDER
// ============================================================================

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('auth_token'));
  const [loading, setLoading] = useState(true);
  const [refreshTimer, setRefreshTimer] = useState(null);

  // Initialize auth state on mount
  useEffect(() => {
    const initAuth = async () => {
      const savedToken = localStorage.getItem('auth_token');
      const savedUser = localStorage.getItem('auth_user');

      if (savedToken && savedUser) {
        setToken(savedToken);
        setUser(JSON.parse(savedUser));
        setupAxiosInterceptors(savedToken, logout);

        // Verify token is still valid
        try {
          const response = await axios.get(`${API_BASE}/api/v2/auth/me`, {
            headers: { Authorization: `Bearer ${savedToken}` }
          });
          setUser(response.data.user);
        } catch (error) {
          // Token invalid, clear auth
          logout();
        }
      }
      setLoading(false);
    };

    initAuth();
  }, []);

  // Setup token refresh (refresh 5 minutes before expiry)
  useEffect(() => {
    if (token) {
      // Clear existing timer
      if (refreshTimer) clearTimeout(refreshTimer);

      // Set new refresh timer (23 hours and 55 minutes)
      const timer = setTimeout(() => {
        refreshToken();
      }, 23 * 60 * 60 * 1000 + 55 * 60 * 1000);

      setRefreshTimer(timer);
    }

    return () => {
      if (refreshTimer) clearTimeout(refreshTimer);
    };
  }, [token]);

  const login = async (email, password) => {
    try {
      const response = await axios.post(`${API_BASE}/api/v2/auth/login`, {
        email,
        password
      });

      const { access_token, user } = response.data;

      // Save to state
      setToken(access_token);
      setUser(user);

      // Save to localStorage
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('auth_user', JSON.stringify(user));

      // Setup axios interceptors
      setupAxiosInterceptors(access_token, logout);

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed'
      };
    }
  };

  const register = async (email, password, name) => {
    try {
      const response = await axios.post(`${API_BASE}/api/v2/auth/register`, {
        email,
        password,
        name
      });

      const { access_token, user } = response.data;

      // Save to state
      setToken(access_token);
      setUser(user);

      // Save to localStorage
      localStorage.setItem('auth_token', access_token);
      localStorage.setItem('auth_user', JSON.stringify(user));

      // Setup axios interceptors
      setupAxiosInterceptors(access_token, logout);

      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Registration failed'
      };
    }
  };

  const logout = () => {
    // Clear state
    setUser(null);
    setToken(null);

    // Clear localStorage
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');

    // Clear saved scripts from localStorage (migrate to backend)
    // Keep API keys in backend only

    // Remove axios interceptors
    axios.interceptors.request.clear();
    axios.interceptors.response.clear();

    // Clear refresh timer
    if (refreshTimer) clearTimeout(refreshTimer);
  };

  const refreshToken = async () => {
    try {
      const response = await axios.post(`${API_BASE}/api/v2/auth/refresh`, {
        token
      });

      const { access_token } = response.data;

      // Update token
      setToken(access_token);
      localStorage.setItem('auth_token', access_token);

      // Setup new interceptors
      setupAxiosInterceptors(access_token, logout);
    } catch (error) {
      // Refresh failed, logout
      logout();
    }
  };

  const updateProfile = async (updates) => {
    try {
      const response = await axios.put(`${API_BASE}/api/v2/user/profile`, updates);
      setUser(response.data.user);
      localStorage.setItem('auth_user', JSON.stringify(response.data.user));
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Update failed'
      };
    }
  };

  const value = {
    user,
    token,
    loading,
    login,
    register,
    logout,
    refreshToken,
    updateProfile,
    isAuthenticated: !!token
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// ============================================================================
// LOGIN/SIGNUP COMPONENT
// ============================================================================

export const AuthModal = ({ isOpen, onClose, onSuccess, mandatory = false }) => {
  const [tab, setTab] = useState('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // Form data
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    name: ''
  });

  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    try {
      if (tab === 'login') {
        const result = await login(formData.email, formData.password);
        if (result.success) {
          setSuccess('Login successful! Redirecting...');
          setTimeout(() => {
            onSuccess && onSuccess();
            if (!mandatory) {
              onClose && onClose();
            }
          }, 1000);
        } else {
          setError(result.error);
        }
      } else {
        // Validate registration
        if (formData.password !== formData.confirmPassword) {
          setError('Passwords do not match');
          setLoading(false);
          return;
        }

        if (formData.password.length < 6) {
          setError('Password must be at least 6 characters');
          setLoading(false);
          return;
        }

        const result = await register(
          formData.email,
          formData.password,
          formData.name
        );

        if (result.success) {
          setSuccess('Registration successful! Welcome aboard!');
          setTimeout(() => {
            onSuccess && onSuccess();
            if (!mandatory) {
              onClose && onClose();
            }
          }, 1000);
        } else {
          setError(result.error);
        }
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={mandatory ? undefined : (e) => {
        if (e.target === e.currentTarget) onClose && onClose();
      }}
    >
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="w-6 h-6 text-orange-600" />
              <CardTitle>Authentication</CardTitle>
            </div>
            {!mandatory && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="hover:bg-gray-100"
              >
                ✕
              </Button>
            )}
          </div>
          <CardDescription>
            {mandatory
              ? "Please sign in to access API Autopilot"
              : "Sign in to save your test scripts and API keys securely"
            }
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">
                <LogIn className="w-4 h-4 mr-2" />
                Login
              </TabsTrigger>
              <TabsTrigger value="register">
                <UserPlus className="w-4 h-4 mr-2" />
                Sign Up
              </TabsTrigger>
            </TabsList>

            <TabsContent value="login" className="space-y-4">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      className="pl-10"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="pl-10 pr-10"
                      value={formData.password}
                      onChange={(e) => setFormData({...formData, password: e.target.value})}
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-orange-600 hover:bg-orange-700"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Signing in...
                    </>
                  ) : (
                    <>
                      <LogIn className="w-4 h-4 mr-2" />
                      Sign In
                    </>
                  )}
                </Button>

                <div className="text-center">
                  <a href="#" className="text-sm text-orange-600 hover:underline">
                    Forgot password?
                  </a>
                </div>
              </form>
            </TabsContent>

            <TabsContent value="register" className="space-y-4">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                    <Input
                      id="name"
                      type="text"
                      placeholder="John Doe"
                      className="pl-10"
                      value={formData.name}
                      onChange={(e) => setFormData({...formData, name: e.target.value})}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="reg-email">Email</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                    <Input
                      id="reg-email"
                      type="email"
                      placeholder="you@example.com"
                      className="pl-10"
                      value={formData.email}
                      onChange={(e) => setFormData({...formData, email: e.target.value})}
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="reg-password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                    <Input
                      id="reg-password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="pl-10 pr-10"
                      value={formData.password}
                      onChange={(e) => setFormData({...formData, password: e.target.value})}
                      required
                      minLength={6}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirm-password">Confirm Password</Label>
                  <div className="relative">
                    <KeyRound className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                    <Input
                      id="confirm-password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="pl-10"
                      value={formData.confirmPassword}
                      onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
                      required
                      minLength={6}
                    />
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-orange-600 hover:bg-orange-700"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Creating account...
                    </>
                  ) : (
                    <>
                      <UserPlus className="w-4 h-4 mr-2" />
                      Create Account
                    </>
                  )}
                </Button>
              </form>
            </TabsContent>
          </Tabs>

          {/* Error/Success Messages */}
          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="mt-4 border-green-300 bg-green-50">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-800">{success}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

// ============================================================================
// PROTECTED ROUTE COMPONENT
// ============================================================================

export const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const [showAuth, setShowAuth] = useState(false);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-orange-600" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <>
        <div className="flex flex-col items-center justify-center h-screen">
          <Shield className="w-16 h-16 text-gray-400 mb-4" />
          <h2 className="text-2xl font-bold mb-2">Authentication Required</h2>
          <p className="text-gray-600 mb-6">Please sign in to access this feature</p>
          <Button
            onClick={() => setShowAuth(true)}
            className="bg-orange-600 hover:bg-orange-700"
          >
            <LogIn className="w-4 h-4 mr-2" />
            Sign In
          </Button>
        </div>
        <AuthModal
          isOpen={showAuth}
          onClose={() => setShowAuth(false)}
          onSuccess={() => setShowAuth(false)}
        />
      </>
    );
  }

  return children;
};

// ============================================================================
// USER MENU COMPONENT
// ============================================================================

export const UserMenu = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const [showAuth, setShowAuth] = useState(false);

  if (!isAuthenticated) {
    return (
      <>
        <Button
          onClick={() => setShowAuth(true)}
          className="bg-orange-600 hover:bg-orange-700"
          size="sm"
        >
          <LogIn className="w-4 h-4 mr-2" />
          Sign In
        </Button>
        <AuthModal
          isOpen={showAuth}
          onClose={() => setShowAuth(false)}
          onSuccess={() => setShowAuth(false)}
        />
      </>
    );
  }

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-orange-600 rounded-full flex items-center justify-center text-white font-bold">
          {user?.name?.charAt(0).toUpperCase()}
        </div>
        <div className="text-sm">
          <p className="font-medium">{user?.name}</p>
          <p className="text-xs text-gray-500">{user?.email}</p>
        </div>
      </div>
      <Button
        onClick={logout}
        variant="outline"
        size="sm"
        className="border-red-200 text-red-600 hover:bg-red-50"
      >
        <LogOut className="w-4 h-4" />
      </Button>
    </div>
  );
};