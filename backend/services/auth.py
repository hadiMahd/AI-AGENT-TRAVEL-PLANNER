"""
Authentication service — password hashing and JWT management.

All auth concerns live here so routers stay thin:
- Password hashing: bcrypt directly (passlib is incompatible with bcrypt 4.1+)
- JWT creation: HS256 signed with the secret key from Settings/Vault
- JWT decoding: validates signature + expiration, returns payload dict

Why bcrypt directly instead of passlib?
- passlib hasn't been updated since 2020 and is incompatible with bcrypt 4.1+
- bcrypt 5.x has a clean, simple API: hashpw() and checkpw()
- No wrapper needed — fewer dependencies, less surface area for bugs
- The bcrypt library handles salt generation, constant-time comparison, etc.

Why python-jose over PyJWT?
- python-jose supports multiple algorithms (HS256, RS256, etc.)
- Built-in expiration handling (exp claim)
- Compatible with FastAPI's OAuth2PasswordBearer
- Same interface as PyJWT with more flexibility

Why HS256 over RS256?
- Symmetric signing (shared secret) is simpler — one key to manage
- RS256 (asymmetric) requires key pair generation and rotation
- HS256 is sufficient for single-service auth (no third-party verification)
- The secret key is stored in Vault for security
"""

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from config import get_settings


def hash_password(plain: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    bcrypt automatically generates a unique salt per hash,
    so identical passwords produce different hashes.
    The cost factor (rounds) defaults to 12 — sufficient for 2024+.

    Args:
        plain: The plaintext password from the user.

    Returns:
        The bcrypt hash string (includes algorithm, cost, salt, hash).
        e.g. "$2b$12$LJ3m4ys3Gz0rOGO5q2N0oO4kF1Ww8Y6H3m9RzA.vK1xJ2bC4dE6fG"
    """
    password_bytes = plain.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a bcrypt hash.

    bcrypt.checkpw performs the comparison in constant time to prevent
    timing attacks — we never roll our own comparison.

    Args:
        plain:  The plaintext password from the login request.
        hashed: The bcrypt hash stored in the database.

    Returns:
        True if the password matches, False otherwise.
    """
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict[str, object], expires_delta: timedelta | None = None) -> str:
    """
    Create a signed JWT access token.

    The token payload includes:
    - "sub": the subject (user ID) — identifies who this token belongs to
    - "exp": expiration timestamp — after this, the token is invalid
    - "iat": issued-at timestamp — when the token was created

    Args:
        data:           Dict with claims to include (must contain "sub": user_id).
        expires_delta:  Optional custom expiration duration.
                        Defaults to Settings.jwt_expiration_minutes (60 min).

    Returns:
        Encoded JWT string (three dot-separated base64 segments).
    """
    settings = get_settings()

    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiration_minutes)

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, object]:
    """
    Decode and validate a JWT access token.

    Validates:
    - Signature matches (proves the token was issued by us)
    - Expiration hasn't passed (proves the token is still valid)
    - Algorithm matches HS256 (prevents algorithm confusion attacks)

    Args:
        token: The encoded JWT string from the Authorization header.

    Returns:
        The decoded payload dict (e.g., {"sub": "user-uuid", "exp": ..., "iat": ...}).

    Raises:
        JWTError: If the token is invalid, expired, or tampered with.
                  The caller (get_current_user) catches this and returns 401.
    """
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
