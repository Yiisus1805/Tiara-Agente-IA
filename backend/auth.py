from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET    = os.getenv("JWT_SECRET", "tiara-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

_bearer = HTTPBearer(auto_error=False)


def create_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
    return _decode_token(credentials.credentials)


def require_admin(user: dict = Depends(require_auth)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Acceso restringido a administradores")
    return user


def _verify_password(password: str, expected: str) -> bool:
    if not expected:
        return False
    # Soporta hash bcrypt ($2b$...) o contraseña en texto plano para dev
    if expected.startswith("$2b$") or expected.startswith("$2a$"):
        return bcrypt.checkpw(password.encode(), expected.encode())
    return password == expected


def check_credentials(username: str, password: str) -> str | None:
    """Valida credenciales contra las cuentas configuradas por variable de entorno.
    Retorna el rol ("admin" | "user") si son válidas, None si no."""
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    if username == admin_user and _verify_password(password, os.getenv("ADMIN_PASSWORD", "")):
        return "admin"

    user_user = os.getenv("USER_USERNAME", "")
    if user_user and username == user_user and _verify_password(password, os.getenv("USER_PASSWORD", "")):
        return "user"

    return None
