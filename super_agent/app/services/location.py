"""
Location detection using ip-api.com (free, no API key required).

Flow:
  1. Frontend tries browser navigator.geolocation (precise, requires permission).
  2. If denied / unavailable, frontend calls GET /api/v1/location which resolves
     the caller's IP via ip-api.com and returns city/country/lat/lon.
  3. Location is attached to every chat request as `user_location`.
  4. The orchestrator injects it into location-sensitive search queries.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_IP_API_URL = "http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,region,regionName,city,lat,lon,timezone,isp,query"

_LOCATION_RE = re.compile(
    r"\b(near(by|est)?|close to|around here|in my (city|area|town|country|region)|"
    r"local(ly)?|where i am|my location|from here|near me|around me|"
    r"restaurants|hotels|shops?|gym|hospital|pharmacy|supermarket|"
    r"weather|forecast|temperature|today.*weather|weather.*today)\b",
    re.IGNORECASE,
)


def looks_location_sensitive(text: str) -> bool:
    """Return True if the query probably needs location context."""
    return bool(_LOCATION_RE.search(text))


def get_location_from_ip(ip: str | None = None) -> dict[str, Any]:
    """
    Resolve an IP address to location data via ip-api.com.
    Pass ip=None or ip="" to auto-detect (the server's own outgoing IP will be used).
    Returns a dict with keys: city, regionName, country, countryCode, lat, lon, timezone, isp, ip.
    Returns {"error": "..."} on failure.
    """
    target = ip.strip() if ip else ""
    url = _IP_API_URL.format(ip=target) if target and target not in ("127.0.0.1", "::1", "localhost") else "http://ip-api.com/json/?fields=status,message,country,countryCode,region,regionName,city,lat,lon,timezone,isp,query"
    try:
        with httpx.Client(timeout=6.0) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning("ip-api.com lookup failed: %s", exc)
        return {"error": str(exc)}

    if data.get("status") != "success":
        return {"error": data.get("message", "lookup failed")}

    return {
        "city": data.get("city", ""),
        "region": data.get("regionName", ""),
        "country": data.get("country", ""),
        "country_code": data.get("countryCode", ""),
        "lat": data.get("lat"),
        "lon": data.get("lon"),
        "timezone": data.get("timezone", ""),
        "isp": data.get("isp", ""),
        "ip": data.get("query", ip or ""),
    }


def format_location_for_prompt(loc: dict[str, Any]) -> str:
    """Turn a location dict into a terse string for injection into prompts."""
    if not loc or loc.get("error"):
        return ""
    parts = []
    if loc.get("city"):
        parts.append(loc["city"])
    if loc.get("region"):
        parts.append(loc["region"])
    if loc.get("country"):
        parts.append(loc["country"])
    coords = ""
    if loc.get("lat") is not None and loc.get("lon") is not None:
        coords = f" (lat={loc['lat']:.3f}, lon={loc['lon']:.3f})"
    tz = f", timezone={loc['timezone']}" if loc.get("timezone") else ""
    return ", ".join(parts) + coords + tz


def enrich_query_with_location(query: str, loc: dict[str, Any] | None) -> str:
    """
    If the query is location-sensitive and we have location data,
    append a location hint so web searches return local results.
    """
    if not loc or loc.get("error"):
        return query
    if not looks_location_sensitive(query):
        return query
    loc_str = format_location_for_prompt(loc)
    if loc_str and loc_str.lower() not in query.lower():
        return f"{query} [User location: {loc_str}]"
    return query
