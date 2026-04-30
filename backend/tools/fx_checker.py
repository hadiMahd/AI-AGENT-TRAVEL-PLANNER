"""
FX checker tool — wraps the frankfurter.app exchange rate service as a LangGraph tool.

Takes two 3-letter currency codes and returns the current exchange rate.
Also provides a helper to look up currency codes from country names.

Per INSTRUCTIONS.md: "Pydantic validation on every tool input."
"""

import json
import logging

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.fx import get_currency_for_country, get_exchange_rate

logger = logging.getLogger(__name__)


class FXToolInput(BaseModel):
    base_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="3-letter base currency code (e.g. 'USD', 'EGP')",
    )
    target_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="3-letter target currency code (e.g. 'IDR', 'EUR')",
    )


@tool(args_schema=FXToolInput)
async def fx_checker(base_currency: str, target_currency: str) -> str:
    """
    Get the current exchange rate between two currencies.

    Use this tool when the user needs currency conversion for trip planning.
    Provide 3-letter currency codes (ISO 4217).
    Common codes: USD, EUR, GBP, EGP, IDR, THB, JPY, AED, CAD.
    """
    try:
        base = base_currency.upper()
        target = target_currency.upper()

        if base == target:
            return json.dumps({"rate": 1.0, "base": base, "target": target, "note": "Same currency"})

        result = await get_exchange_rate(base, target)
        if result is None:
            return (
                f"[fx_checker] Exchange rate for {base}→{target} is not available "
                f"on frankfurter.app (ECB data). Try a major currency like USD or EUR."
            )
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.error("fx_checker failed: %s", exc)
        return f"[fx_checker ERROR] {exc}"


def get_currency_code(country: str) -> str:
    """
    Convenience function for the agent — look up currency code from country name.

    Used during the routing step to extract currency codes from
    the user's origin country and the RAG destination country.
    """
    return get_currency_for_country(country)
