"""Flight search service tests — mock DDGS, retries, caching."""

import pytest

from services.flights import _cache as flights_cache
from services.flights import search_flights


@pytest.fixture(autouse=True)
def clear_flights_cache():
    flights_cache.clear()
    yield
    flights_cache.clear()


@pytest.mark.asyncio
async def test_search_flights_returns_list(mocker):
    mock_results = [
        {"title": "Cairo to Bali", "snippet": "From $450, 1 stop", "url": "https://example.com/1"},
        {"title": "Cheap flights", "snippet": "From $480, direct", "url": "https://example.com/2"},
    ]
    mocker.patch("services.flights._search_sync", return_value=mock_results)

    result = await search_flights("Cairo", "Bali")

    assert result is not None
    assert len(result) == 2
    assert result[0]["title"] == "Cairo to Bali"
    assert result[0]["snippet"] == "From $450, 1 stop"
    assert result[0]["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_search_flights_uses_cache(mocker):
    call_count = [0]

    def track_calls(query):
        call_count[0] += 1
        return [{"title": "Test", "body": "From $100", "href": "https://test.com"}]

    mocker.patch("services.flights._search_sync", side_effect=track_calls)

    await search_flights("A", "B")
    await search_flights("A", "B")

    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_search_flights_different_routes_different_cache_keys(mocker):
    call_count = [0]

    def track_calls(query):
        call_count[0] += 1
        return [{"title": "Test", "body": "From $100", "href": "https://test.com"}]

    mocker.patch("services.flights._search_sync", side_effect=track_calls)

    await search_flights("Cairo", "Bali")
    await search_flights("London", "Tokyo")

    assert call_count[0] == 2  # different cache keys


@pytest.mark.asyncio
async def test_search_flights_empty_result_returns_empty_list(mocker):
    mocker.patch("services.flights._search_sync", return_value=[])

    result = await search_flights("Nowhere", "Noplace")

    assert result == []


@pytest.mark.asyncio
async def test_search_flights_retries_on_failure(mocker):
    call_count = [0]

    def fail_then_ok(query):
        call_count[0] += 1
        if call_count[0] < 2:
            raise RuntimeError("Search unavailable")
        return [{"title": "OK", "body": "Found", "href": "https://ok.com"}]

    mocker.patch("services.flights._search_sync", side_effect=fail_then_ok)

    result = await search_flights("A", "B")

    assert result is not None
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_search_flights_returns_none_after_exhausted_retries(mocker):
    mocker.patch("services.flights._search_sync", side_effect=RuntimeError("down"))

    result = await search_flights("A", "B")

    assert result is None
