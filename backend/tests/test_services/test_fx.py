"""FX service tests — mock httpx, retries, caching, via-EUR fallback."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.fx import (
    _cache as fx_cache,
    get_currency_for_country,
    get_exchange_rate,
)


@pytest.fixture(autouse=True)
def clear_fx_cache():
    fx_cache.clear()
    yield
    fx_cache.clear()


class TestCurrencyLookup:
    def test_known_country(self):
        assert get_currency_for_country("Indonesia") == "IDR"
        assert get_currency_for_country("Japan") == "JPY"
        assert get_currency_for_country("Italy") == "EUR"

    def test_unknown_country_falls_back_to_usd(self):
        assert get_currency_for_country("Atlantis") == "USD"
        assert get_currency_for_country("") == "USD"


@pytest.mark.asyncio
async def test_same_currency_returns_one(mocker):
    """No API call when base == target."""
    result = await get_exchange_rate("USD", "USD")

    assert result["rate"] == 1.0
    assert result["base"] == "USD"
    assert result["target"] == "USD"


@pytest.mark.asyncio
async def test_get_exchange_rate_direct(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "rates": {"IDR": 16000},
        "date": "2026-04-29",
    }
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)
    mocker.patch("services.fx.httpx.AsyncClient", return_value=mock_client)

    result = await get_exchange_rate("USD", "IDR")

    assert result["rate"] == 16000
    assert result["base"] == "USD"
    assert result["target"] == "IDR"


@pytest.mark.asyncio
async def test_get_exchange_rate_fallback_via_eur(mocker):
    call_index = [0]

    async def mock_get(url, params=None):
        call_index[0] += 1
        resp = MagicMock()
        if call_index[0] == 1:
            # First call: USD→EGP not supported by frankfurter
            resp.status_code = 400
        elif call_index[0] == 2:
            # EUR→USD = 0.85
            resp.status_code = 200
            resp.json.return_value = {"rates": {"USD": 0.85}}
        else:
            # EUR→EGP = 30
            resp.status_code = 200
            resp.json.return_value = {"rates": {"EGP": 30}}
        return resp

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = mock_get
    mocker.patch("services.fx.httpx.AsyncClient", return_value=mock_client)

    result = await get_exchange_rate("USD", "EGP")

    assert result is not None
    assert call_index[0] == 3
    # 1 USD = (1/0.85) EUR * 30 EGP/EUR ≈ 35.2941
    assert abs(result["rate"] - 35.2941) < 0.01


@pytest.mark.asyncio
async def test_get_exchange_rate_uses_cache(mocker):
    call_count = [0]

    async def mock_get(url, params=None):
        call_count[0] += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"rates": {"EUR": 0.92}, "date": "2026-01-01"}
        return resp

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = mock_get
    mocker.patch("services.fx.httpx.AsyncClient", return_value=mock_client)

    await get_exchange_rate("USD", "EUR")
    await get_exchange_rate("USD", "EUR")

    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_get_exchange_rate_retries_on_failure(mocker):
    call_count = [0]

    async def fail_then_ok(url, params=None):
        call_count[0] += 1
        if call_count[0] < 2:
            raise OSError("Network error")
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"rates": {"IDR": 16000}}
        return resp

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = fail_then_ok
    mocker.patch("services.fx.httpx.AsyncClient", return_value=mock_client)

    result = await get_exchange_rate("USD", "IDR")

    assert result is not None
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_get_exchange_rate_returns_none_after_exhausted_retries(mocker):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(side_effect=OSError("down"))
    mocker.patch("services.fx.httpx.AsyncClient", return_value=mock_client)

    result = await get_exchange_rate("USD", "IDR")

    assert result is None
