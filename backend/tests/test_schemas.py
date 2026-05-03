"""Pydantic boundary tests — every schema validated for valid/invalid inputs."""

import pytest
from pydantic import ValidationError

from models.schemas import (
    ChatRequest,
    HistoryMessage,
    LoginRequest,
    SignupRequest,
    TravelStyleFeatures,
)


class TestTravelStyleFeatures:
    """Validates the 8-feature travel style input schema."""

    VALID_FEATURES = {
        "active_movement": 0.8,
        "relaxation": 0.2,
        "cultural_interest": 0.5,
        "cost_sensitivity": 0.3,
        "luxury_preference": 0.7,
        "family_friendliness": 0.1,
        "nature_orientation": 0.9,
        "social_group": 0.4,
    }

    def test_valid_input_constructs(self):
        f = TravelStyleFeatures(**self.VALID_FEATURES)
        assert f.active_movement == 0.8
        assert f.relaxation == 0.2

    def test_boundary_zero_accepted(self):
        f = TravelStyleFeatures(**{**self.VALID_FEATURES, "active_movement": 0.0})
        assert f.active_movement == 0.0

    def test_boundary_one_accepted(self):
        f = TravelStyleFeatures(**{**self.VALID_FEATURES, "active_movement": 1.0})
        assert f.active_movement == 1.0

    def test_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            TravelStyleFeatures(**{**self.VALID_FEATURES, "active_movement": -0.1})

    def test_above_one_rejected(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            TravelStyleFeatures(**{**self.VALID_FEATURES, "active_movement": 1.5})

    def test_missing_field_rejected(self):
        feats = {**self.VALID_FEATURES}
        del feats["relaxation"]
        with pytest.raises(ValidationError):
            TravelStyleFeatures(**feats)

    def test_defaults_are_not_provided(self):
        """No defaults — every field is required."""
        with pytest.raises(ValidationError):
            TravelStyleFeatures()


class TestSignupRequest:
    def test_valid_signup_constructs(self):
        req = SignupRequest(email="user@example.com", password="secret123")
        assert req.email == "user@example.com"

    def test_email_min_length(self):
        with pytest.raises(ValidationError):
            SignupRequest(email="", password="secret123")

    def test_password_min_6_chars(self):
        with pytest.raises(ValidationError, match="at least 6 characters"):
            SignupRequest(email="a@b.com", password="123")

    def test_password_too_long(self):
        with pytest.raises(ValidationError):
            SignupRequest(email="a@b.com", password="x" * 129)

    def test_missing_email_rejected(self):
        with pytest.raises(ValidationError):
            SignupRequest(password="secret123")

    def test_missing_password_rejected(self):
        with pytest.raises(ValidationError):
            SignupRequest(email="a@b.com")


class TestLoginRequest:
    def test_valid_login_constructs(self):
        req = LoginRequest(email="user@example.com", password="secret123")
        assert req.email == "user@example.com"

    def test_empty_email_rejected(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="", password="x")

    def test_empty_password_rejected(self):
        with pytest.raises(ValidationError):
            LoginRequest(email="a@b.com", password="")


class TestChatRequest:
    def test_valid_minimal_request(self):
        req = ChatRequest(query="plan a trip to Bali")
        assert req.query == "plan a trip to Bali"
        assert req.origin_country is None
        assert req.history == []

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 2001)

    def test_with_origin_country(self):
        req = ChatRequest(query="plan a trip", origin_country="Egypt")
        assert req.origin_country == "Egypt"

    def test_with_history(self):
        req = ChatRequest(
            query="what about flights?",
            history=[
                HistoryMessage(role="user", content="plan Bali"),
                HistoryMessage(
                    role="assistant",
                    content="Where are you traveling from?",
                ),
            ],
        )
        assert len(req.history) == 2
        assert req.history[0].role == "user"

    def test_history_message_role_is_string(self):
        """HistoryMessage uses str type for role — description only suggests values."""
        msg = HistoryMessage(role="user", content="hello")
        assert msg.role == "user"
        msg2 = HistoryMessage(role="assistant", content="ok")
        assert msg2.role == "assistant"
