import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from test_poc import test_place_search_api as place_search_poc
from dags_pipelines.agent_search_client import BravePlaceSearchClient, BraveSearchClient


def test_poc_main_uses_requested_church_search(monkeypatch, capsys):
    client = Mock()
    client.search.return_value = {
        "type": "locations",
        "results": [
            {
                "title": "Far Church",
                "coordinates": {"latitude": 10.77, "longitude": 106.628},
            },
            {"title": "Church without coordinates"},
            {
                "title": "Near Church",
                "url": "https://example.com",
                "provider_url": "https://provider.example",
                "description": "Description",
                "coordinates": {"latitude": 10.754, "longitude": 106.629},
                "thumbnail": {"src": "thumbnail.jpg"},
                "postal_address": {"city": "Ho Chi Minh City"},
                "opening_hours": {"sunday": "08:00"},
                "categories": ["church"],
            }
        ],
    }
    monkeypatch.setattr(place_search_poc, "BravePlaceSearchClient", lambda: client)

    place_search_poc.main()

    client.search.assert_called_once_with(
        "church",
        latitude=10.7536097,
        longitude=106.6284595,
        radius=5000,
        count=2,
    )
    assert json.loads(capsys.readouterr().out) == [
        {
            "title": "Near Church",
            "url": "https://example.com",
            "provider_url": "https://provider.example",
            "description": "Description",
            "coordinates": {"latitude": 10.754, "longitude": 106.629},
            "thumbnail": {"src": "thumbnail.jpg"},
            "postal_address": {"city": "Ho Chi Minh City"},
            "opening_hours": {"sunday": "08:00"},
            "categories": ["church"],
        },
        {
            "title": "Far Church",
            "coordinates": {"latitude": 10.77, "longitude": 106.628},
        },
        {"title": "Church without coordinates"},
    ]


def test_search_sends_auth_location_and_returns_decoded_response():
    payload = {"type": "locations", "results": [{"title": "Coffee shop"}]}
    response = Mock()
    response.json.return_value = payload
    session = Mock()
    session.get.return_value = response
    client = BravePlaceSearchClient(api_key="test-token", session=session)

    result = client.search(
        "coffee shops",
        latitude=37.7749,
        longitude=-122.4194,
        radius=1000,
        count=5,
    )

    assert result == payload
    response.raise_for_status.assert_called_once_with()
    session.get.assert_called_once_with(
        "https://api.search.brave.com/res/v1/local/place_search",
        params={
            "count": 5,
            "q": "coffee shops",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "radius": 1000,
        },
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": "test-token",
        },
        timeout=15.0,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latitude": 37.0},
        {"longitude": -122.0},
        {"latitude": 91.0, "longitude": -122.0},
        {"latitude": 37.0, "longitude": -181.0},
        {"latitude": 37.0, "longitude": -122.0, "location": "San Francisco"},
        {"radius": 1000},
        {"count": 0},
        {"count": 101},
        {"query": "  "},
    ],
)
def test_search_rejects_invalid_arguments(kwargs):
    client = BravePlaceSearchClient(api_key="test-token", session=Mock())

    with pytest.raises(ValueError):
        client.search(**kwargs)


def test_context_search_uses_only_grounding_generic():
    generic = {
        "url": "https://parish.example/schedule",
        "title": "Parish",
        "snippets": ["Sunday Mass at 08:00"],
    }
    requests = []

    def handle_request(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "grounding": {"generic": [generic]},
                "answer": "Ignored response data",
                "other": [{"url": "https://ignored.example"}],
            },
        )

    client = BraveSearchClient(
        api_key="test-token",
        transport=httpx.MockTransport(handle_request),
    )

    result = asyncio.run(
        client.search(" Nghia Hoa Church ", count=10, search_lang="vi")
    )

    assert result == [generic]
    assert len(requests) == 1
    request = requests[0]
    assert request.url.host == "api.search.brave.com"
    assert request.url.path == "/res/v1/llm/context"
    assert dict(request.url.params) == {
        "q": "Nghia Hoa Church",
        "count": "10",
        "search_lang": "vi",
    }
    assert request.headers["X-Subscription-Token"] == "test-token"


def test_context_search_rejects_response_without_generic_grounding():
    def handle_request(request):
        assert request.url.path == "/res/v1/llm/context"
        return httpx.Response(200, json={"grounding": {}})

    client = BraveSearchClient(
        api_key="test-token",
        transport=httpx.MockTransport(handle_request),
    )

    with pytest.raises(ValueError, match="grounding.generic"):
        asyncio.run(client.search("Nghia Hoa Church"))


def test_context_search_surfaces_http_failures():
    def handle_request(request):
        assert request.method == "GET"
        return httpx.Response(429, json={"error": "rate limited"})

    client = BraveSearchClient(
        api_key="test-token",
        transport=httpx.MockTransport(handle_request),
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.search("Nghia Hoa Church"))


def test_context_search_reuses_client_connection_with_async_context():
    requests = []

    def handle_request(request):
        requests.append(request)
        return httpx.Response(200, json={"grounding": {"generic": []}})

    async def scenario():
        client = BraveSearchClient(
            api_key="test-token",
            transport=httpx.MockTransport(handle_request),
        )
        async with client:
            assert await client.search("Church One") == []
            assert await client.search("Church Two") == []

    asyncio.run(scenario())

    assert len(requests) == 2
