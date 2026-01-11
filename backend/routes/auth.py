from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests
import bcrypt
import logging
import jwt
from datetime import datetime, timedelta
from typing import Optional
from backend.config import GOOGLE_CLIENT_ID
from backend.database import get_user_collection
from backend.schemas import UserSignup, UserLogin, ResetPasswordRequest

# Ensure logger is set up
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])

import requests as req_lib  # Alias to avoid conflict with google.auth.transport.requests

# --- JWT Configuration ---
SECRET_KEY = "your-secret-key-change-this-in-production" # TODO: Move to .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

class GoogleAuthRequest(BaseModel):
    token: Optional[str] = None
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    is_signup: Optional[bool] = False

@router.post("/auth/google")
async def google_login(request: GoogleAuthRequest):
    try:
        token_to_verify = request.token

        # If code is provided, exchange it for an ID token
        if request.code:
            if not request.redirect_uri:
                raise HTTPException(status_code=400, detail="Redirect URI is required for code exchange")
            
            from backend.config import GOOGLE_CLIENT_SECRET
            if not GOOGLE_CLIENT_SECRET:
                raise HTTPException(status_code=500, detail="Server misconfiguration: Missing Google Client Secret")

            # Exchange code for tokens
            token_endpoint = "https://oauth2.googleapis.com/token"
            payload = {
                "code": request.code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": request.redirect_uri,
                "grant_type": "authorization_code"
            }
            
            # Use standard requests library to call Google Payload
            # We use req_lib because 'requests' is imported from google.auth.transport
            resp = req_lib.post(token_endpoint, data=payload)
            token_data = resp.json()
            
            if "error" in token_data:
                logger.error(f"Google Token Exchange Error: {token_data}")
                raise HTTPException(status_code=400, detail=f"Google Token Exchange Failed: {token_data.get('error_description')}")
            
            token_to_verify = token_data.get("id_token")
            if not token_to_verify:
                 raise HTTPException(status_code=400, detail="No ID token returned from Google")

        if not token_to_verify:
             raise HTTPException(status_code=400, detail="No token or code provided")

        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            token_to_verify, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        # ID token is valid. Get the user's Google Account ID from the decoded token.
        userid = idinfo['sub']
        email = idinfo.get('email')
        name = idinfo.get('name')
        picture = idinfo.get('picture')

        # Check if user exists in DB, if not create
        # Check if user exists in DB, if not create
        users_col = get_user_collection()
        if users_col is not None:
            existing_user = await users_col.find_one({"email": email})
            
            # --- NEW CHECK: If signing up but user exists -> Error ---
            if request.is_signup and existing_user:
                raise HTTPException(status_code=400, detail="Account already exists. Please sign in.")
            # ---------------------------------------------------------

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
                logger.info(f"User logged in: {email}")
        
        access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
        access_token = create_access_token(
            data={"sub": email, "name": name, "picture": picture},
            expires_delta=access_token_expires
        )

        return {
            "status": "success",
            "user": {
                "email": email,
                "name": name,
                "picture": picture
            },
            "access_token": access_token
        }

    except ValueError as ve:
        # Invalid token
        logger.error(f"Token verification error: {ve}")
        raise HTTPException(status_code=401, detail="Invalid Google Token")
    except Exception as e:
        logger.error(f"Auth loop error: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

# --- Local Auth ---

def verify_password(plain_password, hashed_password):
    # bcrypt.checkpw requires bytes. 
    # plain_password comes as str, hashed_password comes as str from DB
    if not plain_password or not hashed_password:
        return False
    try:
        # Standard bcrypt limit is 72 bytes. Truncate if necessary or client-side hash.
        # Here we just encode. If it's too long, bcrypt will raise ValueError, so we handle it.
        pwd_bytes = plain_password.encode('utf-8')
        
        # hashed_password from DB is str, needs to be bytes
        hash_bytes = hashed_password.encode('utf-8')
        
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except ValueError:
        return False # Handle potential length errors or invalid format gracefully
    except Exception as e:
        logger.error(f"Bcrypt verify error: {e}")
        return False

def get_password_hash(password):
    # bcrypt.hashpw returns bytes. We decode to utf-8 str for storage in MongoDB.
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

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

    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    access_token = create_access_token(
        data={
            "sub": db_user["email"], 
            "name": db_user.get("name", "User"),
            "picture": db_user.get("picture", "")
        },
        expires_delta=access_token_expires
    )

    return {
        "status": "success", 
        "user": {
            "email": db_user["email"], 
            "name": db_user.get("name", "User"),
            "picture": db_user.get("picture", "")
        },
        "access_token": access_token
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
