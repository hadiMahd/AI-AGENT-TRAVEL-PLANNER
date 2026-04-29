from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


class TravelStyleFeatures(BaseModel):
	active_movement: float = Field(
		ge=0.0,
		le=1.0,
		description="Preference for physical activities and hiking.",
	)
	relaxation: float = Field(
		ge=0.0,
		le=1.0,
		description="Preference for downtime, wellness, and spa experiences.",
	)
	cultural_interest: float = Field(
		ge=0.0,
		le=1.0,
		description="Interest in museums, history, and local art.",
	)
	cost_sensitivity: float = Field(
		ge=0.0,
		le=1.0,
		description="Importance of staying within a low budget.",
	)
	luxury_preference: float = Field(
		ge=0.0,
		le=1.0,
		description="Preference for high-end services and comfort.",
	)
	family_friendliness: float = Field(
		ge=0.0,
		le=1.0,
		description="Suitability for children and family groups.",
	)
	nature_orientation: float = Field(
		ge=0.0,
		le=1.0,
		description="Focus on outdoor and natural environments.",
	)
	social_group: float = Field(
		ge=0.0,
		le=1.0,
		description="Preference for group activities and nightlife.",
	)

	model_config = ConfigDict(
		json_schema_extra={
			"example": {
				"active_movement": 0.8,
				"relaxation": 0.2,
				"cultural_interest": 0.1,
				"cost_sensitivity": 0.1,
				"luxury_preference": 0.9,
				"family_friendliness": 0.0,
				"nature_orientation": 0.9,
				"social_group": 0.1,
			}
		}
	)


class TravelStylePredictionResponse(BaseModel):
	predicted_style: str
	confidence: float = Field(ge=0.0, le=1.0)
	probabilities: dict[str, float]
	input_features: TravelStyleFeatures

	model_config = ConfigDict(
		json_schema_extra={
			"example": {
				"predicted_style": "Adventure",
				"confidence": 0.91,
				"probabilities": {
					"Adventure": 0.91,
					"Relaxation": 0.04,
					"Culture": 0.03,
					"Budget": 0.02,
				},
				"input_features": {
					"active_movement": 0.8,
					"relaxation": 0.2,
					"cultural_interest": 0.1,
					"cost_sensitivity": 0.1,
					"luxury_preference": 0.9,
					"family_friendliness": 0.0,
					"nature_orientation": 0.9,
					"social_group": 0.1,
				},
			}
		}
	)


class SignupRequest(BaseModel):
	email: str = Field(
		...,
		min_length=1,
		max_length=255,
		description="User email address (unique login identifier)",
	)
	password: str = Field(
		...,
		min_length=6,
		max_length=128,
		description="Plaintext password — will be bcrypt-hashed before storage",
	)

	model_config = ConfigDict(
		json_schema_extra={
			"example": {
				"email": "user@example.com",
				"password": "secret123",
			}
		}
	)


class LoginRequest(BaseModel):
	email: str = Field(
		...,
		min_length=1,
		max_length=255,
		description="User email address",
	)
	password: str = Field(
		...,
		min_length=1,
		max_length=128,
		description="Plaintext password to verify against the stored hash",
	)

	model_config = ConfigDict(
		json_schema_extra={
			"example": {
				"email": "user@example.com",
				"password": "secret123",
			}
		}
	)


class TokenResponse(BaseModel):
	access_token: str = Field(description="JWT access token (HS256 signed)")
	token_type: str = Field(default="bearer", description="Token type — always 'bearer'")

	model_config = ConfigDict(
		json_schema_extra={
			"example": {
				"access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
				"token_type": "bearer",
			}
		}
	)


class UserOut(BaseModel):
	id: str
	email: str
	created_at: datetime | None = None

	model_config = ConfigDict(
		from_attributes=True,
		json_schema_extra={
			"example": {
				"id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
				"email": "user@example.com",
				"created_at": "2026-04-29T12:00:00+00:00",
			}
		}
	)


class APIMessage(BaseModel):
	message: str
	details: dict[str, Any] | None = None