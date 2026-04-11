from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

from .exceptions import ExternalSignalError

LOGGER = logging.getLogger(__name__)


def fetch_google_news(asset_name: str, limit: int = 5) -> list[str]:
    query = quote_plus(f"{asset_name} crypto")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    titles: list[str] = []
    for item in root.findall("./channel/item/title"):
        title = (item.text or "").strip()
        if title:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def get_live_news(asset_name: str, limit: int = 5) -> str:
    LOGGER.info("Fetching external news context for %s", asset_name)
    titles = fetch_google_news(asset_name, limit=limit)
    if not titles:
        raise ExternalSignalError(f"No verifiable news titles returned for asset {asset_name}")
    return " | ".join(titles)
