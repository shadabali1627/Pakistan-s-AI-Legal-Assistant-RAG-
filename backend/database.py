import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from backend.config import MONGODB_URL, DATABASE_NAME

logger = logging.getLogger(__name__)

client = None

def get_db_client():
    global client
    if client is None:
        if not MONGODB_URL:
            logger.warning("MONGODB_URL not set, cannot connect to database.")
            return None
        logger.info("Initializing MongoDB client...")
        client = AsyncIOMotorClient(MONGODB_URL)
    return client

def get_db():
    client = get_db_client()
    if client:
        return client[DATABASE_NAME]
    return None

def get_chat_collection():
    db = get_db()
    if db is not None:
        return db["chats"]
    return None

def get_user_collection():
    db = get_db()
    if db is not None:
        return db["users"]
    return None

def get_session_collection():
    db = get_db()
    if db is not None:
        return db["sessions"]
    return None
