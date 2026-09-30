"""Вежливый краулер публичной Базы знаний Webim (только https://webim.ru/kb/).

* обнаружение: живой sitemap.xml (+ опционально ссылки внутри статей, строго в пределах /kb/);
* канонизация URL и защита области (scope);
* кэш на диске + условные запросы (ETag / Last-Modified);
* ограничение частоты запросов, ретраи, ручная обработка редиректов (каждый переход проверяется на scope).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from .config import KB_BASE, KB_HOST, KB_SITEMAP, USER_AGENT

_SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|webp|svg|ico|pdf|zip|gz|css|js|json|xml|woff2?|ttf|mp4|webm|txt)$", re.I)


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Возвращает нормализованный URL внутри KB или None, если URL вне области краулинга."""
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    host = (p.hostname or "").lower()
    if host not in (KB_HOST, "www." + KB_HOST):
        return None
    path = p.path or "/"
    # нормализуем ../ и //
    parts: list[str] = []
    for seg in path.split("/"):
        if seg == "..":
            if parts:
                parts.pop()
        elif seg not in ("", "."):
            parts.append(seg)
    path = "/" + "/".join(parts)
    last = parts[-1] if parts else ""
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    elif last.endswith(".html"):
        pass
    elif "." in last:  # картинки, css, js и прочие файлы — не страницы
        return None
    elif not path.endswith("/"):
        path += "/"
    if not (path == "/kb/" or path.startswith("/kb/")):
        return None
    if p.query:  # у KB нет query-страниц; такие ссылки — поиск/трекинг
        return None
    if _SKIP_EXT.search(path):
        return None
    return f"https://{KB_HOST}{path}"


def in_scope(url: str) -> bool:
    return canonicalize(url) is not None


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int  # 200, 304 (из кэша), 404, 0 (ошибка сети)
    html: str | None
    etag: str | None = None
    last_modified: str | None = None
    from_cache: bool = False
    error: str | None = None


class Crawler:
    def __init__(self, cache_dir: Path, delay: float = 0.3, client: httpx.Client | None = None, max_retries: int = 3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.max_retries = max_retries
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.5"},
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
        )
        self._last = 0.0
        self.requests_made = 0

    # ---------- низкоуровневое ----------
    def _throttle(self) -> None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str, headers: dict | None = None) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                self.requests_made += 1
                r = self.client.get(url, headers=headers or {})
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    last_exc = RuntimeError(f"HTTP {r.status_code}")
                    continue
                return r
            except httpx.HTTPError as e:  # сеть/таймаут
                last_exc = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"не удалось получить {url}: {last_exc}")

    # ---------- кэш ----------
    def _paths(self, url: str) -> tuple[Path, Path]:
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / f"{h}.html", self.cache_dir / f"{h}.json"

    def negative_cached(self, url: str, ttl: float = 7 * 86400) -> bool:
        _, mp = self._paths(url)
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            return m.get("status") in (404, 410) and time.time() - m.get("fetched_at", 0) < ttl
        return False

    def cached(self, url: str) -> FetchResult | None:
        hp, mp = self._paths(url)
        if hp.exists() and mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            return FetchResult(url, m.get("final_url", url), m.get("status", 200), hp.read_text(encoding="utf-8"),
                               m.get("etag"), m.get("last_modified"), from_cache=True)
        return None

    # ---------- публичное ----------
    def discover(self, extra_sitemaps: list[Path] | None = None) -> dict[str, str | None]:
        """Возвращает {canonical_url: lastmod}. Основной источник — живой sitemap; локальные — только запасной вариант."""
        found: dict[str, str | None] = {}
        try:
            r = self._get(KB_SITEMAP)
            if r.status_code == 200:
                found.update(parse_sitemap(r.text))
        except RuntimeError:
            pass
        if not found:  # запасной вариант: файлы из репозитория (могут быть устаревшими)
            for p in extra_sitemaps or []:
                if Path(p).exists():
                    found.update(parse_sitemap(Path(p).read_text(encoding="utf-8")))
        found.setdefault(KB_BASE, None)
        return found

    def fetch(self, url: str, use_cache: bool = True, force: bool = False) -> FetchResult:
        """Загружает страницу (в пределах scope). Использует условный GET, если есть кэш."""
        cu = canonicalize(url)
        if cu is None:
            return FetchResult(url, url, 0, None, error="вне области /kb/")
        if use_cache and not force and self.negative_cached(cu):
            return FetchResult(cu, cu, 404, None, error="HTTP 404 (негативный кэш)")
        cached = self.cached(cu) if use_cache else None
        if cached and not force:
            return cached
        headers = {}
        if cached and cached.etag:
            headers["If-None-Match"] = cached.etag
        if cached and cached.last_modified:
            headers["If-Modified-Since"] = cached.last_modified
        cur = cu
        try:
            for _ in range(5):  # ручные редиректы с проверкой scope
                r = self._get(cur, headers)
                if r.status_code in (301, 302, 303, 307, 308):
                    nxt = canonicalize(r.headers.get("location", ""), base=cur)
                    if nxt is None:
                        return FetchResult(cu, cur, r.status_code, None, error="редирект за пределы /kb/")
                    cur = nxt
                    continue
                break
            else:
                return FetchResult(cu, cur, 0, None, error="слишком много редиректов")
        except RuntimeError as e:
            if cached:  # сеть недоступна — работаем по кэшу
                return cached
            return FetchResult(cu, cu, 0, None, error=str(e))
        if r.status_code == 304 and cached:
            return cached
        if r.status_code != 200:
            if r.status_code in (404, 410):  # негативный кэш: не долбим несуществующие страницы каждую ночь
                _, mp = self._paths(cu)
                mp.write_text(json.dumps({"status": r.status_code, "fetched_at": time.time()}), encoding="utf-8")
            return FetchResult(cu, cur, r.status_code, None, error=f"HTTP {r.status_code}")
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype:
            return FetchResult(cu, cur, r.status_code, None, error=f"неожиданный content-type {ctype}")
        html = r.text
        hp, mp = self._paths(cu)
        hp.write_text(html, encoding="utf-8")
        mp.write_text(json.dumps({"final_url": cur, "status": 200, "etag": r.headers.get("etag"),
                                  "last_modified": r.headers.get("last-modified"), "fetched_at": time.time()}), encoding="utf-8")
        return FetchResult(cu, cur, 200, html, r.headers.get("etag"), r.headers.get("last-modified"))


def parse_sitemap(xml: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for m in re.finditer(r"<url>(.*?)</url>", xml, re.S):
        loc = re.search(r"<loc>\s*([^<\s]+)\s*</loc>", m.group(1))
        if not loc:
            continue
        cu = canonicalize(loc.group(1).replace("&amp;", "&"))
        if not cu:
            continue
        lm = re.search(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", m.group(1))
        out[cu] = (lm.group(1) if lm else None) or out.get(cu)
    return out
