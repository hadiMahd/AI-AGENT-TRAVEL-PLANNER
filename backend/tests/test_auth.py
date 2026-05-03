"""Auth pipeline tests — password hashing, JWT round-trip, signup/login endpoints."""

from datetime import timedelta

import pytest
from jose import JWTError

from services.auth import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        hashed = hash_password("mysecret")
        assert verify_password("mysecret", hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password("mysecret")
        assert verify_password("wrong", hashed) is False

    def test_identical_passwords_different_hashes(self):
        """Same password, different salts — hashes must differ."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_empty_password_hashes(self):
        """Empty string should still hash (bcrypt handles it)."""
        hashed = hash_password("")
        assert len(hashed) > 0

    def test_verify_empty_password(self):
        hashed = hash_password("")
        assert verify_password("", hashed) is True


class TestJWT:
    def test_create_and_decode_roundtrip(self, test_settings):
        token = create_access_token({"sub": "user-123"})
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert "exp" in payload
        assert "iat" in payload

    def test_expired_token_rejected(self, test_settings):
        token = create_access_token(
            {"sub": "user-123"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(JWTError):
            decode_token(token)

    def test_tampered_token_rejected(self, test_settings):
        token = create_access_token({"sub": "user-123"})
        with pytest.raises(JWTError):
            decode_token(token + "tampered")

    def test_missing_sub_claim(self, test_settings):
        token = create_access_token({"email": "a@b.com"})
        payload = decode_token(token)
        assert "sub" not in payload
        assert "email" in payload
