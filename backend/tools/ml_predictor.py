"""
ML predictor tool — wraps the travel style prediction model as a LangGraph tool.

Takes 8 float features (0-1 range) and returns the predicted travel style
with confidence scores. The model is a scikit-learn RandomForest loaded
at startup via joblib.

Per INSTRUCTIONS.md: "Wrap in an async agent tool with Pydantic input
validation. Return structured prediction + confidence."
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from models.schemas import TravelStyleFeatures
from services.ml_inference import infer_travel_style

logger = logging.getLogger(__name__)


class MLToolInput(BaseModel):
    active_movement: float = Field(ge=0.0, le=1.0, description="Preference for physical activities")
    relaxation: float = Field(ge=0.0, le=1.0, description="Preference for downtime and spas")
    cultural_interest: float = Field(ge=0.0, le=1.0, description="Interest in museums and history")
    cost_sensitivity: float = Field(ge=0.0, le=1.0, description="Importance of low budget")
    luxury_preference: float = Field(ge=0.0, le=1.0, description="Preference for high-end comfort")
    family_friendliness: float = Field(ge=0.0, le=1.0, description="Suitability for families")
    nature_orientation: float = Field(ge=0.0, le=1.0, description="Focus on outdoor/nature")
    social_group: float = Field(ge=0.0, le=1.0, description="Preference for group activities")


@tool(args_schema=MLToolInput)
async def ml_predictor(
    active_movement: float = 0.5,
    relaxation: float = 0.5,
    cultural_interest: float = 0.5,
    cost_sensitivity: float = 0.5,
    luxury_preference: float = 0.5,
    family_friendliness: float = 0.5,
    nature_orientation: float = 0.5,
    social_group: float = 0.5,
    model: Any = None,
) -> str:
    """
    Predict a traveler's style based on their preference scores.

    Use this tool when the user describes their travel preferences
    (active vs relaxed, budget vs luxury, cultural vs nature, etc.)
    Returns the predicted travel style with confidence scores.
    """
    if model is None:
        return "[ml_predictor ERROR] ML model is not loaded"

    try:
        features = TravelStyleFeatures(
            active_movement=active_movement,
            relaxation=relaxation,
            cultural_interest=cultural_interest,
            cost_sensitivity=cost_sensitivity,
            luxury_preference=luxury_preference,
            family_friendliness=family_friendliness,
            nature_orientation=nature_orientation,
            social_group=social_group,
        )

        result = await infer_travel_style(model, features)
        return json.dumps(result, indent=2, default=str)

    except Exception as exc:
        logger.error("ml_predictor failed: %s", exc)
        return f"[ml_predictor ERROR] {exc}"
