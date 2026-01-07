from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests
from passlib.context import CryptContext
from backend.config import GOOGLE_CLIENT_ID
from backend.database import get_user_collection
from backend.schemas import UserSignup, UserLogin, ResetPasswordRequest
from datetime import datetime
import logging

# Ensure logger is set up
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])

class GoogleAuthRequest(BaseModel):
    token: str

@router.post("/auth/google")
async def google_login(request: GoogleAuthRequest):
    try:
        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        # ID token is valid. Get the user's Google Account ID from the decoded token.
        userid = idinfo['sub']
        email = idinfo.get('email')
        name = idinfo.get('name')
        picture = idinfo.get('picture')

        # Check if user exists in DB, if not create
        users_col = get_user_collection()
        if users_col is not None:
            existing_user = await users_col.find_one({"email": email})
            if not existing_user:
                new_user = {
                    "google_id": userid,
                    "email": email,
                    "name": name,
                    "picture": picture,
                    "created_at": datetime.utcnow(),
                    "role": "user"
                }
                await users_col.insert_one(new_user)
                logger.info(f"Created new user: {email}")
            else:
                # Update login time or info if needed
                await users_col.update_one(
                    {"email": email},
                    {"$set": {"last_login": datetime.utcnow(), "picture": picture, "name": name}}
                )
                logger.info(f"User logged in: {email}")
        
        return {
            "status": "success",
            "user": {
                "email": email,
                "name": name,
                "picture": picture
            }
        }

    except ValueError:
        # Invalid token
        raise HTTPException(status_code=401, detail="Invalid Google Token")
    except Exception as e:
        logger.error(f"Auth loop error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


# --- Local Auth ---

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

@router.post("/auth/signup")
async def signup(user: UserSignup):
    users_col = get_user_collection()
    if users_col is None:
         raise HTTPException(status_code=500, detail="Database not available")
    
    existing = await users_col.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Account already exists")
    
    hashed_pw = get_password_hash(user.password)
    new_user = {
        "email": user.email,
        "password_hash": hashed_pw,
        "name": user.full_name,
        "created_at": datetime.utcnow(),
        "role": "user"
    }
    await users_col.insert_one(new_user)
    return {"status": "success", "user": {"email": user.email, "name": user.full_name}}

@router.post("/auth/login")
async def login(user: UserLogin):
    users_col = get_user_collection()
    if users_col is None:
         raise HTTPException(status_code=500, detail="Database not available")

    # DB user
    db_user = await users_col.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    # Check if user uses Google Auth only (no password set)
    if "password_hash" not in db_user:
        raise HTTPException(status_code=400, detail="Please sign in with Google")

    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    # Update last login
    await users_col.update_one(
        {"email": user.email},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    return {
        "status": "success", 
        "user": {
            "email": db_user["email"], 
            "name": db_user.get("name", "User"),
            "picture": db_user.get("picture", "")
        }
    }

@router.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    users_col = get_user_collection()
    if users_col is None:
         raise HTTPException(status_code=500, detail="Database not available")
    
    db_user = await users_col.find_one({"email": req.email})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_hash = get_password_hash(req.new_password)
    await users_col.update_one(
        {"email": req.email},
        {"$set": {"password_hash": new_hash}}
    )
    
    return {"status": "success", "message": "Password updated successfully"}
