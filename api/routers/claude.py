"""Claude proxy — each user's own Anthropic API key, stored encrypted server-side.

The browser sends the key once (PUT /api/claude/key) and never sees it again; every Claude call
goes through POST /api/claude/messages, which attaches the caller's key and forwards the body
unchanged to Anthropic's Messages API. Callers are identified by their Supabase access token.

env.env needs:
    CLAUDE_KEY_SECRET  Fernet key, from: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    SUPABASE_URL, SUPABASE_ANON_KEY  same values as the frontend's VITE_SUPABASE_*
"""

from __future__ import annotations
import os
import uuid
from pathlib import Path

import requests
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

load_dotenv('env.env')

router = APIRouter()

ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def _fernet() -> Fernet:
    secret = os.getenv("CLAUDE_KEY_SECRET")
    if not secret:
        raise HTTPException(500, detail="Server is missing CLAUDE_KEY_SECRET")
    return Fernet(secret.encode())


def _key_path(user_id: str) -> Path:
    return Path(os.getenv("STORAGE_DIR", "storage")) / "api_keys" / user_id


def current_user(authorization: str | None = Header(None)) -> str:
    """The caller's Supabase user id, checked against Supabase itself so a forged token fails."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, detail="Not signed in")
    if not os.getenv("SUPABASE_URL"):
        raise HTTPException(500, detail="Server is missing SUPABASE_URL")
    res = requests.get(
        f"{os.getenv('SUPABASE_URL')}/auth/v1/user",
        headers={"apikey": os.getenv("SUPABASE_ANON_KEY", ""), "Authorization": authorization},
        timeout=10,
    )
    if res.status_code != 200:
        raise HTTPException(401, detail="Session expired, sign in again")
    user_id = res.json().get("id", "")
    # the id becomes a filename, so only ever accept a real UUID
    try:
        return str(uuid.UUID(user_id))
    except ValueError:
        raise HTTPException(401, detail="Invalid user")


def _load_key(user_id: str) -> str | None:
    path = _key_path(user_id)
    if not path.exists():
        return None
    try:
        return _fernet().decrypt(path.read_bytes()).decode()
    except InvalidToken:
        return None


def _anthropic_headers(key: str) -> dict[str, str]:
    return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}


def _anthropic_error(res: requests.Response) -> HTTPException:
    try:
        message = res.json()["error"]["message"]
    except Exception:
        message = res.text or res.reason
    return HTTPException(res.status_code, detail=f"Claude API: {message}")


class KeyRequest(BaseModel):
    key: str


@router.get("/claude/key")
def get_key(user_id: str = Depends(current_user)):
    key = _load_key(user_id)
    return {"has_key": key is not None, "last4": key[-4:] if key else None}


@router.put("/claude/key")
def set_key(body: KeyRequest, user_id: str = Depends(current_user)):
    key = body.key.strip()
    # a cheap call that only succeeds with a working key, so typos are caught here, not mid-chat
    res = requests.get(f"{ANTHROPIC_URL}/models", params={"limit": 1}, headers=_anthropic_headers(key), timeout=15)
    if res.status_code != 200:
        raise _anthropic_error(res)
    path = _key_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_fernet().encrypt(key.encode()))
    return {"has_key": True, "last4": key[-4:]}


@router.delete("/claude/key")
def delete_key(user_id: str = Depends(current_user)):
    _key_path(user_id).unlink(missing_ok=True)
    return {"has_key": False, "last4": None}


@router.post("/claude/messages")
def messages(body: dict, user_id: str = Depends(current_user)):
    key = _load_key(user_id)
    if key is None:
        raise HTTPException(400, detail="Add your Claude API key on the dashboard first")
    body.pop("stream", None)  # responses are relayed whole
    res = requests.post(f"{ANTHROPIC_URL}/messages", json=body, headers=_anthropic_headers(key), timeout=600)
    if res.status_code != 200:
        raise _anthropic_error(res)
    return res.json()
