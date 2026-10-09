"""Client for Brave Search's Place Search API."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import httpx
import requests
from dotenv import load_dotenv

PLACE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/local/place_search"
LLM_SEARCH_API_URL = "https://api.search.brave.com/res/v1/llm/context"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BravePlaceSearchClient:
    """Search Brave's place index and return its decoded JSON response."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        configured_api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not configured_api_key or not configured_api_key.strip():
            raise ValueError(
                "Set BRAVE_API_KEY in the environment or repository .env file."
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.api_key: str = configured_api_key.strip()
        self.timeout = timeout
        self.session = session or requests.Session()

    def search(
        self,
        query: str | None = None,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        location: str | None = None,
        radius: float | None = None,
        count: int = 10,
        country: str | None = None,
        search_lang: str | None = None,
        ui_lang: str | None = None,
        units: str | None = None,
        safesearch: str | None = None,
        spellcheck: bool | None = None,
    ) -> dict[str, Any]:
        """Search by query and optional coordinates or named location.

        ``radius`` biases results within the supplied coordinates; it is not a
        hard distance cutoff. Omitting ``query`` requests a general local search.
        """
        if (latitude is None) != (longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        if latitude is not None and (
            not math.isfinite(latitude) or not -90 <= latitude <= 90
        ):
            raise ValueError("latitude must be between -90 and 90")
        if longitude is not None and (
            not math.isfinite(longitude) or not -180 <= longitude <= 180
        ):
            raise ValueError("longitude must be between -180 and 180")
        if location and latitude is not None:
            raise ValueError("use either location or latitude/longitude, not both")
        if radius is not None and latitude is None:
            raise ValueError("radius requires latitude and longitude")
        if radius is not None and (not math.isfinite(radius) or radius <= 0):
            raise ValueError("radius must be greater than zero")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 100
        ):
            raise ValueError("count must be an integer between 1 and 100")
        if query is not None and not query.strip():
            raise ValueError("query must not be empty")
        if location is not None and not location.strip():
            raise ValueError("location must not be empty")

        params: dict[str, str | float | int | bool] = {"count": count}
        if query is not None:
            params["q"] = query.strip()
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
        if location is not None:
            params["location"] = location.strip()
        if radius is not None:
            params["radius"] = radius

        optional_params = {
            "country": country,
            "search_lang": search_lang,
            "ui_lang": ui_lang,
            "units": units,
            "safesearch": safesearch,
            "spellcheck": spellcheck,
        }
        params.update(
            {key: value for key, value in optional_params.items() if value is not None}
        )

        response = self.session.get(
            PLACE_SEARCH_API_URL,
            params=params,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Brave Place Search returned a non-object JSON response")
        if not isinstance(payload.get("results"), list):
            raise ValueError("Brave Place Search response has no results list")
        return payload


class BraveSearchClient:
    """Asynchronously fetch grounded web results from Brave Search."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        configured_api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not configured_api_key or not configured_api_key.strip():
            raise ValueError(
                "Set BRAVE_API_KEY in the environment or repository .env file."
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.api_key = configured_api_key.strip()
        self.timeout = timeout
        self.transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "BraveSearchClient":
        if self._client is not None:
            raise RuntimeError("BraveSearchClient is already open")
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            transport=self.transport,
            trust_env=False,
        )
        return self

    async def __aexit__(self, *_args: Any) -> None:
        del _args
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def search(
        self,
        query: str,
        *,
        count: int = 5,
        search_lang: str = "vi",
    ) -> list[dict[str, Any]]:
        """Return only Brave's ``grounding.generic`` search results."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError("count must be an integer between 1 and 20")
        if not search_lang.strip():
            raise ValueError("search_lang must not be empty")

        params = {
            "q": query.strip(),
            "count": count,
            "search_lang": search_lang.strip(),
        }
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        if self._client is not None:
            response = await self._client.get(
                LLM_SEARCH_API_URL, params=params, headers=headers
            )
        else:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                transport=self.transport,
                trust_env=False,
            ) as client:
                response = await client.get(
                    LLM_SEARCH_API_URL, params=params, headers=headers
                )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Brave Search returned a non-object JSON response")
        grounding = payload.get("grounding")
        if not isinstance(grounding, dict) or not isinstance(
            grounding.get("generic"), list
        ):
            raise ValueError(
                "Brave Search response has no grounding.generic results list"
            )
        if any(not isinstance(item, dict) for item in grounding["generic"]):
            raise ValueError("Brave Search returned an invalid grounding.generic item")
        return grounding["generic"]