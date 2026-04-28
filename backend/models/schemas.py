from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class APIMessage(BaseModel):
	message: str
	details: dict[str, Any] | None = None