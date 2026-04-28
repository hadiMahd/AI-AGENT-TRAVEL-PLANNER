from functools import lru_cache
from pathlib import Path
from typing import Any

import anyio
import pandas as pd

from models.schemas import TravelStyleFeatures


MODEL_FEATURES: list[str] = [
	"active_movement",
	"relaxation",
	"cultural_interest",
	"cost_sensitivity",
	"luxury_preference",
	"family_friendliness",
	"nature_orientation",
	"social_group",
]


class ModelNotAvailableError(RuntimeError):
	pass


@lru_cache(maxsize=1)
def get_model_path() -> Path:
	return Path(__file__).resolve().parents[1] / "artifacts" / "ml_model" / "random_forest_travel_model.pkl"


def features_to_dataframe(features: TravelStyleFeatures) -> pd.DataFrame:
	return pd.DataFrame([[getattr(features, feature) for feature in MODEL_FEATURES]], columns=MODEL_FEATURES)


def build_prediction(model: Any, features: TravelStyleFeatures) -> dict[str, object]:
	data_frame = features_to_dataframe(features)
	prediction = model.predict(data_frame)[0]
	probabilities: dict[str, float] = {}

	if hasattr(model, "predict_proba"):
		scores = model.predict_proba(data_frame)[0]
		probabilities = {
			str(label): float(score)
			for label, score in zip(model.classes_, scores, strict=False)
		}

	confidence = probabilities.get(str(prediction), 1.0 if not probabilities else max(probabilities.values()))

	return {
		"predicted_style": str(prediction),
		"confidence": float(confidence),
		"probabilities": probabilities,
		"input_features": features,
	}


async def infer_travel_style(model: Any, features: TravelStyleFeatures) -> dict[str, object]:
	return await anyio.to_thread.run_sync(build_prediction, model, features)
