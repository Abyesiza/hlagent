"""
Research Tool — web scraping and search for continuous HDC model training.

Pipeline
========
1. Query DuckDuckGo (free, no API key) for a topic
2. Fetch page HTML via httpx
3. Extract clean article text via trafilatura (falls back to BeautifulSoup)
4. Clean and filter the text
5. Return structured ScrapedDocument objects for the training pipeline

Also provides Wikipedia article fetching for high-quality training data.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────────────────────

try:
    import trafilatura  # type: ignore[import-untyped]
    _TRAFILATURA = True
except ImportError:
    trafilatura = None  # type: ignore[assignment]
    _TRAFILATURA = False

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    _BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]
    _BS4 = False

try:
    try:
        from ddgs import DDGS  # type: ignore[import-untyped]  # new package name
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]  # legacy name
    _DDGS = True
except ImportError:
    DDGS = None  # type: ignore[assignment]
    _DDGS = False


# ── constants ─────────────────────────────────────────────────────────────────

_USER_AGENT = (
    "Mozilla/5.0 (compatible; HDCResearchBot/1.0; +https://github.com/hlagent)"
)
_MIN_TEXT_LENGTH = 200          # discard pages shorter than this
_MAX_TEXT_LENGTH = 50_000       # truncate very long articles
_REQUEST_TIMEOUT = 15.0
_RATE_LIMIT_DELAY = 1.5         # seconds between requests

# Domains to skip (login walls, video-only, etc.)
_BLOCKED_DOMAINS = frozenset({
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "tiktok.com", "reddit.com", "youtube.com", "linkedin.com",
    "pinterest.com", "snapchat.com",
})


# ── data types ────────────────────────────────────────────────────────────────


@dataclass
class ScrapedDocument:
    url: str
    title: str
    text: str
    source: str          # "web" | "wikipedia" | "ddg_snippet"
    topic: str = ""
    word_count: int = 0
    scraped_at: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())
        if not self.scraped_at:
            self.scraped_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class ResearchResult:
    topic: str
    documents: list[ScrapedDocument] = field(default_factory=list)
    total_words: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.total_words = sum(d.word_count for d in self.documents)


# ── text cleaning ─────────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s{2,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw: str) -> str:
    """Normalise whitespace and strip control characters."""
    text = html.unescape(raw)
    text = _CONTROL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def extract_text_trafilatura(html_bytes: bytes, url: str) -> str:
    """Use trafilatura for clean main-content extraction."""
    if not _TRAFILATURA:
        return ""
    result = trafilatura.extract(
        html_bytes,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    return clean_text(result) if result else ""


def extract_text_bs4(html_bytes: bytes) -> str:
    """Fallback: use BeautifulSoup to extract paragraph text."""
    if not _BS4:
        return ""
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(separator=" ") for p in paragraphs)
    return clean_text(text)


def extract_text_fallback(html_bytes: bytes, url: str) -> str:
    """Try trafilatura then bs4 then give up."""
    text = extract_text_trafilatura(html_bytes, url)
    if len(text) >= _MIN_TEXT_LENGTH:
        return text[:_MAX_TEXT_LENGTH]
    text = extract_text_bs4(html_bytes)
    return text[:_MAX_TEXT_LENGTH] if len(text) >= _MIN_TEXT_LENGTH else ""


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _is_blocked(url: str) -> bool:
    domain = urlparse(url).netloc.lower().lstrip("www.")
    return domain in _BLOCKED_DOMAINS


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": _USER_AGENT},
        timeout=_REQUEST_TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_connections=5),
    )


# ── DuckDuckGo search ─────────────────────────────────────────────────────────


def search_ddg(query: str, max_results: int = 8) -> list[dict]:
    """
    Search DuckDuckGo and return a list of result dicts with keys:
    title, href, body.
    """
    if not _DDGS:
        logger.warning("duckduckgo_search not installed — using snippet fallback")
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as exc:
        logger.warning("DDG search error for '%s': %s", query, exc)
        return []


# ── Wikipedia ─────────────────────────────────────────────────────────────────


def fetch_wikipedia(topic: str, sentences: int = 10) -> ScrapedDocument | None:
    """
    Fetch a Wikipedia article summary via the REST API.
    High-quality, clean text — ideal training data.
    """
    slug = topic.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    try:
        with httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=10.0) as client:
            r = client.get(url)
        if r.status_code != 200:
            return None
        data = r.json()
        text = clean_text(data.get("extract", ""))
        title = data.get("title", topic)
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", url)
        if len(text) < _MIN_TEXT_LENGTH:
            return None
        return ScrapedDocument(
            url=page_url,
            title=title,
            text=text,
            source="wikipedia",
            topic=topic,
        )
    except Exception as exc:
        logger.warning("Wikipedia fetch failed for '%s': %s", topic, exc)
        return None


def fetch_wikipedia_full(topic: str) -> ScrapedDocument | None:
    """Fetch the full parsed Wikipedia article (more training data)."""
    slug = topic.strip().replace(" ", "_")
    wiki_api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": slug,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "exlimit": 1,
    }
    try:
        with httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=20.0) as client:
            r = client.get(wiki_api, params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        text = clean_text(page.get("extract", ""))
        title = page.get("title", topic)
        if len(text) < _MIN_TEXT_LENGTH:
            return None
        return ScrapedDocument(
            url=f"https://en.wikipedia.org/wiki/{slug}",
            title=title,
            text=text[:_MAX_TEXT_LENGTH],
            source="wikipedia",
            topic=topic,
        )
    except Exception as exc:
        logger.warning("Wikipedia full fetch failed for '%s': %s", topic, exc)
        return None


def fetch_wikipedia_links(topic: str, max_links: int = 6) -> list[ScrapedDocument]:
    """
    Fetch the top linked articles from a Wikipedia page.
    This gives us many *different* articles per topic on repeated runs,
    defeating the deduplication cache with genuinely new content.
    """
    slug = topic.strip().replace(" ", "_")
    wiki_api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": slug,
        "prop": "links",
        "pllimit": str(max_links * 3),  # fetch extra, filter to real articles
        "plnamespace": "0",
        "format": "json",
    }
    docs: list[ScrapedDocument] = []
    try:
        with httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=15.0) as client:
            r = client.get(wiki_api, params=params)
        if r.status_code != 200:
            return docs
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return docs
        page = next(iter(pages.values()))
        links = [lk["title"] for lk in page.get("links", [])][:max_links]
        for linked_title in links:
            try:
                doc = fetch_wikipedia_full(linked_title)
                if doc:
                    doc.topic = topic   # tag with parent topic
                    docs.append(doc)
                    time.sleep(0.3)     # be polite to Wikipedia
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Wikipedia links fetch failed for '%s': %s", topic, exc)
    return docs


# ── web page scraper ──────────────────────────────────────────────────────────


def scrape_url(url: str, topic: str = "", client: httpx.Client | None = None) -> ScrapedDocument | None:
    """
    Fetch and extract text from a web URL.
    Returns None if the page is unscrapable or too short.
    """
    if _is_blocked(url):
        logger.debug("Blocked domain: %s", url)
        return None

    close = client is None
    if client is None:
        client = _make_client()
    try:
        r = client.get(url)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("content-type", "")
        if "text/html" not in ctype and "text/plain" not in ctype:
            return None

        text = extract_text_fallback(r.content, url)
        if len(text) < _MIN_TEXT_LENGTH:
            return None

        # Try to get a title from the URL slug
        title = urlparse(url).path.split("/")[-1].replace("-", " ").replace("_", " ").strip()
        if not title:
            title = url

        return ScrapedDocument(
            url=url,
            title=title,
            text=text,
            source="web",
            topic=topic,
        )
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        logger.debug("Scrape failed (%s): %s", type(exc).__name__, url)
        return None
    except Exception as exc:
        logger.warning("Unexpected scrape error for %s: %s", url, exc)
        return None
    finally:
        if close and client:
            client.close()


# ── main research function ────────────────────────────────────────────────────


def research_topic(
    topic: str,
    max_pages: int = 5,
    include_wikipedia: bool = True,
) -> ResearchResult:
    """
    Research a topic:
    1. Fetch Wikipedia article (high quality)
    2. DuckDuckGo search → fetch top pages
    Returns a ResearchResult with all scraped documents.
    """
    t0 = time.time()
    result = ResearchResult(topic=topic)

    # 1. Wikipedia first (cleanest, most topical data)
    if include_wikipedia:
        try:
            wiki_doc = fetch_wikipedia_full(topic)
            if wiki_doc:
                result.documents.append(wiki_doc)
                logger.info("Wikipedia main: %d words for '%s'", wiki_doc.word_count, topic)
        except Exception as exc:
            result.errors.append(f"wikipedia: {exc}")

    # 2. DuckDuckGo search
    ddg_results = search_ddg(f"{topic} explained overview", max_results=max_pages + 3)

    # Add DDG snippets as lightweight training data (always available)
    for item in ddg_results:
        snippet = clean_text(item.get("body", ""))
        title = item.get("title", "")
        if len(snippet) >= 50:
            result.documents.append(ScrapedDocument(
                url=item.get("href", ""),
                title=title,
                text=f"{title}. {snippet}",
                source="ddg_snippet",
                topic=topic,
            ))

    # 3. Scrape top URLs for full article text
    urls_to_scrape = [
        item["href"]
        for item in ddg_results
        if item.get("href") and not _is_blocked(item["href"])
    ][:max_pages]

    with _make_client() as client:
        for url in urls_to_scrape:
            time.sleep(_RATE_LIMIT_DELAY)
            doc = scrape_url(url, topic=topic, client=client)
            if doc:
                result.documents.append(doc)
                logger.info("Scraped: %d words from %s", doc.word_count, url)

    result.elapsed_seconds = time.time() - t0
    result.total_words = sum(d.word_count for d in result.documents)
    logger.info(
        "Research '%s': %d docs, %d words in %.1fs",
        topic, len(result.documents), result.total_words, result.elapsed_seconds,
    )
    return result


async def research_topic_async(
    topic: str,
    max_pages: int = 5,
    include_wikipedia: bool = True,
) -> ResearchResult:
    """Async wrapper — runs the blocking research in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: research_topic(topic, max_pages=max_pages, include_wikipedia=include_wikipedia),
    )


# ── topic suggestions ─────────────────────────────────────────────────────────

DEFAULT_SEED_TOPICS: list[str] = [
    "Machine learning",
    "Artificial intelligence",
    "Natural language processing",
    "Deep learning",
    "Neural networks",
    "Computer science",
    "Mathematics",
    "Linguistics",
    "Cognitive science",
    "Information theory",
    "Statistics",
    "Algorithms",
    "Hyperdimensional computing",
    "Vector symbolic architectures",
    "Neuro-symbolic AI",
    "Associative memory",
    "Distributed representations",
    "Knowledge representation",
    "Neuroscience",
    "Philosophy of mind",
]
