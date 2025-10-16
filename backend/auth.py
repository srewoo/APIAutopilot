"""
JWT Authentication Module for API Autopilot
Handles user registration, login, and token management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import jwt
import bcrypt
from motor.motor_asyncio import AsyncIOMotorDatabase
import secrets
import logging
import os

logger = logging.getLogger(__name__)

# Configuration - Load from environment variables
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-this-in-production-" + secrets.token_hex(32))
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", "24"))

# Security
security = HTTPBearer()

# ============================================================================
# MODELS
# ============================================================================

class UserRegistration(BaseModel):
    """User registration model"""
    email: EmailStr
    password: str = Field(..., min_length=6, description="Minimum 6 characters")
    name: str = Field(..., min_length=2, max_length=100)

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    """User response model (excludes sensitive data)"""
    id: str
    email: str
    name: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600
    user: UserResponse

class SavedScript(BaseModel):
    """Saved test script model"""
    name: str
    script: str
    framework: str
    tags: Optional[list] = []

class APIKey(BaseModel):
    """API Key model for secure storage"""
    provider: str  # openai, anthropic, gemini
    encrypted_key: str
    last_used: Optional[datetime] = None

# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )

def create_access_token(user_id: str, email: str) -> str:
    """Create JWT access token"""
    payload = {
        "sub": user_id,  # Subject (user ID)
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
        "jti": secrets.token_hex(16)  # JWT ID for token revocation if needed
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# ============================================================================
# AUTHENTICATION SERVICE
# ============================================================================

class AuthService:
    """Authentication service for user management"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.users_collection = db.users
        self.scripts_collection = db.scripts
        self.api_keys_collection = db.api_keys

    async def register_user(self, registration: UserRegistration) -> TokenResponse:
        """Register a new user"""
        # Check if user already exists
        existing_user = await self.users_collection.find_one({"email": registration.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Create new user document
        user_doc = {
            "email": registration.email,
            "password_hash": hash_password(registration.password),
            "name": registration.name,
            "role": "user",
            "created_at": datetime.utcnow(),
            "last_login": None,
            "is_active": True,
            "saved_scripts": [],
            "api_keys": {},
            "preferences": {
                "default_framework": "jest",
                "default_ai_provider": "openai",
                "default_ai_model": "gpt-4o"
            }
        }

        # Insert user into database
        result = await self.users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)

        # Generate token
        access_token = create_access_token(user_id, registration.email)

        # Prepare response
        user_response = UserResponse(
            id=user_id,
            email=registration.email,
            name=registration.name,
            role="user",
            created_at=user_doc["created_at"],
            last_login=None
        )

        return TokenResponse(
            access_token=access_token,
            user=user_response
        )

    async def login_user(self, login: UserLogin) -> TokenResponse:
        """Authenticate user and return token"""
        # Find user by email
        user = await self.users_collection.find_one({"email": login.email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Verify password
        if not verify_password(login.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Check if account is active
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )

        # Update last login
        await self.users_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        # Generate token
        user_id = str(user["_id"])
        access_token = create_access_token(user_id, user["email"])

        # Prepare response
        user_response = UserResponse(
            id=user_id,
            email=user["email"],
            name=user["name"],
            role=user.get("role", "user"),
            created_at=user["created_at"],
            last_login=datetime.utcnow()
        )

        return TokenResponse(
            access_token=access_token,
            user=user_response
        )

    async def get_current_user(self, token: str) -> Dict[str, Any]:
        """Get current user from token"""
        payload = decode_token(token)
        user_id = payload["sub"]

        # Convert string ID to ObjectId for MongoDB
        from bson import ObjectId
        user = await self.users_collection.find_one({"_id": ObjectId(user_id)})

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )

        return {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "role": user.get("role", "user")
        }

    async def save_script(self, user_id: str, script: SavedScript) -> Dict[str, Any]:
        """Save a test script for a user"""
        from bson import ObjectId

        script_doc = {
            "user_id": ObjectId(user_id),
            "name": script.name,
            "script": script.script,
            "framework": script.framework,
            "tags": script.tags,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        result = await self.scripts_collection.insert_one(script_doc)

        return {
            "id": str(result.inserted_id),
            "name": script.name,
            "script": script.script,
            "framework": script.framework,
            "tags": script.tags,
            "created_at": script_doc["created_at"].isoformat(),
            "updated_at": script_doc["updated_at"].isoformat(),
            "message": "Script saved successfully"
        }

    async def get_user_scripts(self, user_id: str) -> list:
        """Get all saved scripts for a user"""
        from bson import ObjectId

        cursor = self.scripts_collection.find(
            {"user_id": ObjectId(user_id)}
        ).sort("created_at", -1)

        scripts = []
        async for script in cursor:
            scripts.append({
                "id": str(script["_id"]),
                "name": script["name"],
                "script": script["script"],  # Include the actual script content
                "framework": script["framework"],
                "tags": script.get("tags", []),
                "created_at": script["created_at"].isoformat(),
                "updated_at": script.get("updated_at", script["created_at"]).isoformat()
            })

        return scripts

    async def get_script_content(self, user_id: str, script_id: str) -> Dict[str, Any]:
        """Get full script content"""
        from bson import ObjectId

        script = await self.scripts_collection.find_one({
            "_id": ObjectId(script_id),
            "user_id": ObjectId(user_id)
        })

        if not script:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Script not found"
            )

        return {
            "id": str(script["_id"]),
            "name": script["name"],
            "script": script["script"],
            "framework": script["framework"],
            "tags": script.get("tags", []),
            "created_at": script["created_at"].isoformat(),
            "updated_at": script.get("updated_at", script["created_at"]).isoformat()
        }

    async def delete_script(self, user_id: str, script_id: str) -> Dict[str, str]:
        """Delete a user's script"""
        from bson import ObjectId

        result = await self.scripts_collection.delete_one({
            "_id": ObjectId(script_id),
            "user_id": ObjectId(user_id)
        })

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Script not found"
            )

        return {"message": "Script deleted successfully"}

    async def save_api_key(self, user_id: str, provider: str, api_key: str) -> Dict[str, str]:
        """Save encrypted API key for a user"""
        from bson import ObjectId
        from cryptography.fernet import Fernet
        import os

        # Generate encryption key (should be stored securely)
        # In production, use a proper key management service
        encryption_key = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key().decode())
        fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)

        # Encrypt the API key
        encrypted_key = fernet.encrypt(api_key.encode()).decode()

        # Update user's API keys
        await self.users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    f"api_keys.{provider}": {
                        "encrypted_key": encrypted_key,
                        "updated_at": datetime.utcnow()
                    }
                }
            }
        )

        return {"message": f"API key for {provider} saved securely"}

    async def get_api_key(self, user_id: str, provider: str) -> Optional[str]:
        """Get decrypted API key for a user"""
        from bson import ObjectId
        from cryptography.fernet import Fernet
        import os

        user = await self.users_collection.find_one({"_id": ObjectId(user_id)})

        if not user or "api_keys" not in user or provider not in user["api_keys"]:
            return None

        encrypted_key = user["api_keys"][provider]["encrypted_key"]

        # Decrypt the API key
        encryption_key = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key().decode())
        fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)

        try:
            decrypted_key = fernet.decrypt(encrypted_key.encode()).decode()
            return decrypted_key
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            return None

# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """FastAPI dependency to get current user from JWT token"""
    token = credentials.credentials
    payload = decode_token(token)

    return {
        "id": payload["sub"],
        "email": payload["email"]
    }