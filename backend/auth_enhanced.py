"""
Enhanced Authentication Module with Advanced Security Features
Includes: Refresh tokens, Password reset, Rate limiting, Session management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import jwt
import bcrypt
from motor.motor_asyncio import AsyncIOMotorDatabase
import secrets
import logging
import os
import time
from collections import defaultdict
import hashlib
import random
import string

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-" + secrets.token_hex(32))
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Rate limiting configuration
RATE_LIMIT_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 300  # 5 minutes
ACCOUNT_LOCKOUT_ATTEMPTS = 5
ACCOUNT_LOCKOUT_DURATION = 1800  # 30 minutes

# Security
security = HTTPBearer()

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self):
        self.attempts = defaultdict(list)
        self.locked_accounts = {}

    def is_rate_limited(self, key: str) -> bool:
        """Check if a key is rate limited"""
        now = time.time()

        # Clean old attempts
        self.attempts[key] = [
            attempt for attempt in self.attempts[key]
            if now - attempt < RATE_LIMIT_WINDOW
        ]

        # Check rate limit
        return len(self.attempts[key]) >= RATE_LIMIT_ATTEMPTS

    def add_attempt(self, key: str):
        """Add an attempt for a key"""
        self.attempts[key].append(time.time())

    def is_account_locked(self, email: str) -> bool:
        """Check if an account is locked"""
        if email in self.locked_accounts:
            if time.time() < self.locked_accounts[email]:
                return True
            else:
                del self.locked_accounts[email]
        return False

    def lock_account(self, email: str):
        """Lock an account for a duration"""
        self.locked_accounts[email] = time.time() + ACCOUNT_LOCKOUT_DURATION

    def get_lock_time_remaining(self, email: str) -> int:
        """Get remaining lock time in seconds"""
        if email in self.locked_accounts:
            remaining = self.locked_accounts[email] - time.time()
            return max(0, int(remaining))
        return 0

rate_limiter = RateLimiter()

# ============================================================================
# ENHANCED MODELS
# ============================================================================

class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""
    refresh_token: str

class PasswordResetRequest(BaseModel):
    """Password reset request model"""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model"""
    token: str
    new_password: str = Field(..., min_length=6)

class PasswordChange(BaseModel):
    """Password change model"""
    current_password: str
    new_password: str = Field(..., min_length=6)

class SessionInfo(BaseModel):
    """Session information model"""
    id: str
    user_id: str
    device: str
    ip_address: str
    location: str
    created_at: datetime
    last_active: datetime
    is_current: bool = False

class ExecutionHistory(BaseModel):
    """Execution history model"""
    id: str
    user_id: str
    script_name: str
    framework: str
    status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    duration: float
    timestamp: datetime
    api_provider: Optional[str] = None
    tokens_used: Optional[int] = None

class UserStatistics(BaseModel):
    """User statistics model"""
    total_tests: int
    tests_this_month: int
    total_executions: int
    success_rate: float
    saved_scripts: int
    scripts_shared: int
    api_keys_stored: int
    framework_usage: List[Dict[str, Any]]
    last_activity: datetime

class UserPreferences(BaseModel):
    """User preferences model"""
    default_framework: str = "jest"
    default_ai_provider: str = "openai"
    default_ai_model: str = "gpt-4o"
    theme: str = "light"
    email_notifications: bool = True
    two_factor_enabled: bool = False

# ============================================================================
# ENHANCED AUTHENTICATION SERVICE
# ============================================================================

class EnhancedAuthService:
    """Enhanced authentication service with advanced features"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.users_collection = db.users
        self.scripts_collection = db.scripts
        self.sessions_collection = db.sessions
        self.reset_tokens_collection = db.reset_tokens
        self.execution_history_collection = db.execution_history
        self.refresh_tokens_collection = db.refresh_tokens

    # ========== TOKEN MANAGEMENT ==========

    def create_tokens(self, user_id: str, email: str) -> Dict[str, str]:
        """Create both access and refresh tokens"""
        # Access token (short-lived)
        access_payload = {
            "sub": user_id,
            "email": email,
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)
        }
        access_token = jwt.encode(access_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        # Refresh token (long-lived)
        refresh_payload = {
            "sub": user_id,
            "email": email,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)
        }
        refresh_token = jwt.encode(refresh_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token
        }

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

            # Verify it's a refresh token
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )

            # Check if refresh token is blacklisted
            blacklisted = await self.refresh_tokens_collection.find_one({
                "token": refresh_token,
                "blacklisted": True
            })
            if blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            # Create new access token
            user_id = payload["sub"]
            email = payload["email"]

            access_payload = {
                "sub": user_id,
                "email": email,
                "type": "access",
                "exp": datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
                "iat": datetime.utcnow(),
                "jti": secrets.token_hex(16)
            }

            new_access_token = jwt.encode(access_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

            return {"access_token": new_access_token}

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

    # ========== PASSWORD RESET ==========

    async def request_password_reset(self, email: str) -> bool:
        """Request password reset - generates reset token"""
        user = await self.users_collection.find_one({"email": email})

        if not user:
            # Don't reveal if email exists
            return True

        # Generate reset token
        reset_token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        reset_hash = hashlib.sha256(reset_token.encode()).hexdigest()

        # Store reset token with expiry
        await self.reset_tokens_collection.insert_one({
            "user_id": str(user["_id"]),
            "email": email,
            "token_hash": reset_hash,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=1),
            "used": False
        })

        # In production, send email with reset link
        # For now, log the token (remove in production!)
        logger.info(f"Password reset token for {email}: {reset_token}")

        return True

    async def reset_password(self, token: str, new_password: str) -> bool:
        """Reset password using reset token"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Find valid reset token
        reset_doc = await self.reset_tokens_collection.find_one({
            "token_hash": token_hash,
            "used": False,
            "expires_at": {"$gt": datetime.utcnow()}
        })

        if not reset_doc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        # Update password
        from bson import ObjectId
        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        await self.users_collection.update_one(
            {"_id": ObjectId(reset_doc["user_id"])},
            {
                "$set": {
                    "password_hash": new_password_hash.decode('utf-8'),
                    "password_changed_at": datetime.utcnow()
                }
            }
        )

        # Mark token as used
        await self.reset_tokens_collection.update_one(
            {"_id": reset_doc["_id"]},
            {"$set": {"used": True}}
        )

        # Revoke all refresh tokens for this user (force re-login)
        await self.refresh_tokens_collection.update_many(
            {"user_id": reset_doc["user_id"]},
            {"$set": {"blacklisted": True}}
        )

        return True

    # ========== SESSION MANAGEMENT ==========

    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str
    ) -> str:
        """Create a new session"""
        session_id = secrets.token_hex(32)

        # Parse user agent for device info
        device = self._parse_user_agent(user_agent)

        # Get approximate location from IP (in production, use GeoIP service)
        location = "Unknown Location"  # Placeholder

        session_doc = {
            "session_id": session_id,
            "user_id": user_id,
            "ip_address": ip_address,
            "device": device,
            "location": location,
            "user_agent": user_agent,
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow(),
            "active": True
        }

        await self.sessions_collection.insert_one(session_doc)

        return session_id

    async def get_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user"""
        from bson import ObjectId

        cursor = self.sessions_collection.find({
            "user_id": user_id,
            "active": True
        }).sort("last_active", -1)

        sessions = []
        async for session in cursor:
            sessions.append(SessionInfo(
                id=session["session_id"],
                user_id=session["user_id"],
                device=session["device"],
                ip_address=session["ip_address"],
                location=session["location"],
                created_at=session["created_at"],
                last_active=session["last_active"],
                is_current=False  # Set by the endpoint based on current session
            ))

        return sessions

    async def revoke_session(self, user_id: str, session_id: str) -> bool:
        """Revoke a specific session"""
        result = await self.sessions_collection.update_one(
            {"user_id": user_id, "session_id": session_id},
            {"$set": {"active": False, "revoked_at": datetime.utcnow()}}
        )

        return result.modified_count > 0

    async def revoke_all_sessions(self, user_id: str, except_current: str = None) -> int:
        """Revoke all sessions for a user (except current if specified)"""
        query = {"user_id": user_id, "active": True}
        if except_current:
            query["session_id"] = {"$ne": except_current}

        result = await self.sessions_collection.update_many(
            query,
            {"$set": {"active": False, "revoked_at": datetime.utcnow()}}
        )

        return result.modified_count

    # ========== EXECUTION HISTORY ==========

    async def log_execution(
        self,
        user_id: str,
        script_name: str,
        framework: str,
        result: Dict[str, Any]
    ) -> str:
        """Log test execution"""
        execution_id = secrets.token_hex(16)

        execution_doc = {
            "execution_id": execution_id,
            "user_id": user_id,
            "script_name": script_name,
            "framework": framework,
            "status": result.get("status", "unknown"),
            "total_tests": result.get("total_tests", 0),
            "passed_tests": result.get("passed_tests", 0),
            "failed_tests": result.get("failed_tests", 0),
            "duration": result.get("duration", 0),
            "timestamp": datetime.utcnow(),
            "api_provider": result.get("api_provider"),
            "tokens_used": result.get("tokens_used"),
            "error": result.get("error")
        }

        await self.execution_history_collection.insert_one(execution_doc)

        # Update user statistics
        await self._update_user_statistics(user_id, execution_doc)

        return execution_id

    async def get_execution_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[ExecutionHistory]:
        """Get execution history for a user"""
        cursor = self.execution_history_collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).skip(offset).limit(limit)

        history = []
        async for execution in cursor:
            history.append(ExecutionHistory(
                id=execution["execution_id"],
                user_id=execution["user_id"],
                script_name=execution["script_name"],
                framework=execution["framework"],
                status=execution["status"],
                total_tests=execution["total_tests"],
                passed_tests=execution["passed_tests"],
                failed_tests=execution["failed_tests"],
                duration=execution["duration"],
                timestamp=execution["timestamp"],
                api_provider=execution.get("api_provider"),
                tokens_used=execution.get("tokens_used")
            ))

        return history

    # ========== USER STATISTICS ==========

    async def get_user_statistics(self, user_id: str) -> UserStatistics:
        """Get comprehensive user statistics"""
        from bson import ObjectId

        # Get user data
        user = await self.users_collection.find_one({"_id": ObjectId(user_id)})

        # Count scripts
        script_count = await self.scripts_collection.count_documents({"user_id": ObjectId(user_id)})

        # Count API keys
        api_keys_count = len(user.get("api_keys", {})) if user else 0

        # Get execution statistics
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "total_executions": {"$sum": 1},
                "total_tests": {"$sum": "$total_tests"},
                "passed_tests": {"$sum": "$passed_tests"},
                "frameworks": {"$push": "$framework"}
            }}
        ]

        exec_stats = await self.execution_history_collection.aggregate(pipeline).to_list(1)

        if exec_stats:
            exec_stats = exec_stats[0]
            total_executions = exec_stats["total_executions"]
            total_tests = exec_stats["total_tests"]
            passed_tests = exec_stats["passed_tests"]

            # Calculate success rate
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

            # Framework usage
            framework_counts = {}
            for framework in exec_stats["frameworks"]:
                framework_counts[framework] = framework_counts.get(framework, 0) + 1

            framework_usage = [
                {
                    "name": framework,
                    "count": count,
                    "percentage": count / total_executions * 100
                }
                for framework, count in framework_counts.items()
            ]
            framework_usage.sort(key=lambda x: x["count"], reverse=True)
        else:
            total_executions = 0
            total_tests = 0
            success_rate = 0
            framework_usage = []

        # Get tests this month
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
        tests_this_month = await self.execution_history_collection.count_documents({
            "user_id": user_id,
            "timestamp": {"$gte": start_of_month}
        })

        # Get last activity
        last_execution = await self.execution_history_collection.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )
        last_activity = last_execution["timestamp"] if last_execution else user.get("created_at", datetime.utcnow())

        return UserStatistics(
            total_tests=total_tests,
            tests_this_month=tests_this_month,
            total_executions=total_executions,
            success_rate=round(success_rate, 2),
            saved_scripts=script_count,
            scripts_shared=0,  # TODO: Implement sharing
            api_keys_stored=api_keys_count,
            framework_usage=framework_usage,
            last_activity=last_activity
        )

    async def _update_user_statistics(self, user_id: str, execution: Dict[str, Any]):
        """Update user statistics after execution"""
        # This could update cached statistics or trigger analytics
        pass

    # ========== USER PREFERENCES ==========

    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences"""
        from bson import ObjectId

        user = await self.users_collection.find_one({"_id": ObjectId(user_id)})

        if user and "preferences" in user:
            return UserPreferences(**user["preferences"])

        return UserPreferences()

    async def update_user_preferences(
        self,
        user_id: str,
        preferences: UserPreferences
    ) -> bool:
        """Update user preferences"""
        from bson import ObjectId

        result = await self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"preferences": preferences.dict()}}
        )

        return result.modified_count > 0

    # ========== ACCOUNT SECURITY ==========

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change user password"""
        from bson import ObjectId

        user = await self.users_collection.find_one({"_id": ObjectId(user_id)})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify current password
        if not bcrypt.checkpw(current_password.encode('utf-8'), user["password_hash"].encode('utf-8')):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )

        # Hash new password
        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        # Update password
        await self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "password_hash": new_password_hash.decode('utf-8'),
                    "password_changed_at": datetime.utcnow()
                }
            }
        )

        # Revoke all refresh tokens (force re-login)
        await self.refresh_tokens_collection.update_many(
            {"user_id": user_id},
            {"$set": {"blacklisted": True}}
        )

        return True

    async def enable_two_factor(self, user_id: str) -> str:
        """Enable two-factor authentication"""
        # Generate TOTP secret
        import pyotp
        secret = pyotp.random_base32()

        from bson import ObjectId
        await self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "two_factor_secret": secret,
                    "two_factor_enabled": False,  # Will be enabled after verification
                    "preferences.two_factor_enabled": True
                }
            }
        )

        return secret

    def _parse_user_agent(self, user_agent: str) -> str:
        """Parse user agent to get device info"""
        # Simple parsing - in production use a proper library
        if "Mobile" in user_agent:
            return "Mobile Device"
        elif "Tablet" in user_agent:
            return "Tablet"
        elif "Windows" in user_agent:
            return "Windows PC"
        elif "Mac" in user_agent:
            return "Mac"
        elif "Linux" in user_agent:
            return "Linux PC"
        else:
            return "Unknown Device"

# ============================================================================
# RATE LIMITING DECORATOR
# ============================================================================

async def check_rate_limit(request: Request, endpoint: str):
    """Check rate limit for an endpoint"""
    client_ip = request.client.host
    key = f"{endpoint}:{client_ip}"

    if rate_limiter.is_rate_limited(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later."
        )

    rate_limiter.add_attempt(key)