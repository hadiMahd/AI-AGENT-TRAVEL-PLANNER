from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from dependencies import get_model
from models.schemas import TravelStyleFeatures, TravelStylePredictionResponse
from services.ml_inference import ModelNotAvailableError, infer_travel_style


router = APIRouter(prefix="/ml", tags=["Travel Model"])


@router.post(
	"/predict",
	response_model=TravelStylePredictionResponse,
	summary="Predict a travel style from feature scores",
)
async def predict_travel_style(
	payload: TravelStyleFeatures,
	model: Any = Depends(get_model),
) -> TravelStylePredictionResponse:
	try:
		result = await infer_travel_style(model, payload)
		return TravelStylePredictionResponse.model_validate(result)
	except ModelNotAvailableError as exc:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail=str(exc),
		) from exc
	except AttributeError as exc:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Model failed to load. Ensure scikit-learn version matches the model artifact.",
		) from exc
