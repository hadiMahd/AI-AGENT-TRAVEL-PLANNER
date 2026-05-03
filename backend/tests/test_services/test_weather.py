"""Weather service tests — httpx mocking, retries, caching."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.weather import get_weather, get_weather_by_city, _cache


@pytest.fixture(autouse=True)
def clear_weather_cache():
    _cache.clear()
    yield
    _cache.clear()


@pytest.mark.asyncio
async def test_get_weather_returns_structured_dict(mocker):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "current_condition": [{
            "temp_C": "28",
            "FeelsLikeC": "30",
            "weatherDesc": [{"value": "Sunny"}],
            "humidity": "75",
            "windspeedKmph": "12",
        }],
        "nearest_area": [{"areaName": [{"value": "Bali"}]}],
    }
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)
    mocker.patch("services.weather.httpx.AsyncClient", return_value=mock_client)

    result = await get_weather(-8.34, 115.09)

    assert result["temp_c"] == 28
    assert result["feels_like_c"] == 30
    assert result["condition"] == "Sunny"
    assert result["humidity"] == 75
    assert result["wind_kph"] == 12.0
    assert result["location"] == "Bali"


@pytest.mark.asyncio
async def test_get_weather_retries_on_failure(mocker):
    call_count = 0

    async def failing_then_ok(url, params=None):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise OSError("Connection refused")
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "current_condition": [{
                "temp_C": "20",
                "FeelsLikeC": "18",
                "weatherDesc": [{"value": "Cloudy"}],
                "humidity": "60",
                "windspeedKmph": "5",
            }],
            "nearest_area": [{"areaName": [{"value": "TestCity"}]}],
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = failing_then_ok
    mocker.patch("services.weather.httpx.AsyncClient", return_value=mock_client)

    result = await get_weather(0, 0)

    assert result["temp_c"] == 20
    assert call_count == 2


@pytest.mark.asyncio
async def test_get_weather_returns_none_after_exhausted_retries(mocker):
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(side_effect=OSError("down"))
    mocker.patch("services.weather.httpx.AsyncClient", return_value=mock_client)

    result = await get_weather(0, 0)

    assert result is None


@pytest.mark.asyncio
async def test_get_weather_uses_cache(mocker):
    call_count = 0

    async def track_calls(url, params=None):
        nonlocal call_count
        call_count += 1
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "current_condition": [{
                "temp_C": "25",
                "FeelsLikeC": "25",
                "weatherDesc": [{"value": "Clear"}],
                "humidity": "50",
                "windspeedKmph": "10",
            }],
            "nearest_area": [{"areaName": [{"value": "Cached"}]}],
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = track_calls
    mocker.patch("services.weather.httpx.AsyncClient", return_value=mock_client)

    await get_weather(10, 20)
    result = await get_weather(10, 20)  # should use cache

    assert result is not None
    assert call_count == 1  # only first call hit the API


@pytest.mark.asyncio
async def test_get_weather_by_city(mocker):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "current_condition": [{
            "temp_C": "22",
            "FeelsLikeC": "21",
            "weatherDesc": [{"value": "Rain"}],
            "humidity": "88",
            "windspeedKmph": "15",
        }],
        "nearest_area": [{"areaName": [{"value": "Tokyo"}]}],
    }
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_response)
    mocker.patch("services.weather.httpx.AsyncClient", return_value=mock_client)

    result = await get_weather_by_city("Tokyo")

    assert result["location"] == "Tokyo"
    assert result["condition"] == "Rain"


@pytest.mark.asyncio
async def test_get_weather_by_city_uses_cache(mocker):
    call_count = 0

    async def track_calls(url, params=None):
        nonlocal call_count
        call_count += 1
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "current_condition": [{
                "temp_C": "30",
                "FeelsLikeC": "33",
                "weatherDesc": [{"value": "Hot"}],
                "humidity": "90",
                "windspeedKmph": "5",
            }],
            "nearest_area": [{"areaName": [{"value": "Dubai"}]}],
        }
        return response

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = track_calls
    mocker.patch("services.weather.httpx.AsyncClient", return_value=mock_client)

    await get_weather_by_city("Dubai")
    await get_weather_by_city("Dubai")

    assert call_count == 1
