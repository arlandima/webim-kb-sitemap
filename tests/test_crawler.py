import httpx

from webim_ai.crawler import Crawler, canonicalize, in_scope, parse_sitemap


def test_canonicalize_scope():
    assert canonicalize("https://webim.ru/kb/a/b.html#x") == "https://webim.ru/kb/a/b.html"
    assert canonicalize("https://webim.ru/kb/faq") == "https://webim.ru/kb/faq/"
    assert canonicalize("https://webim.ru/kb/faq/index.html") == "https://webim.ru/kb/faq/"
    assert canonicalize("../x.html", base="https://webim.ru/kb/dev/api/a.html") == "https://webim.ru/kb/dev/x.html"
    # вне области
    assert canonicalize("https://evil.com/kb/x.html") is None
    assert canonicalize("https://webim.ru/blog/post.html") is None
    assert canonicalize("https://webim.ru/kb/../admin/") is None
    assert canonicalize("https://webim.ru/kb/a.html?s=query") is None
    assert canonicalize("https://webim.ru/kb/img/pic.webp") is None
    assert canonicalize("https://webim.ru/kb/img/pic.webp/") is None
    assert canonicalize("javascript:alert(1)") is None
    assert not in_scope("ftp://webim.ru/kb/a.html")


def test_parse_sitemap_dedup_and_scope():
    xml = """<urlset><url><loc>https://webim.ru/kb/a.html</loc><lastmod>2026-01-01</lastmod></url>
    <url><loc>https://webim.ru/kb/a.html#dup</loc></url><url><loc>https://webim.ru/blog/x</loc></url></urlset>"""
    assert parse_sitemap(xml) == {"https://webim.ru/kb/a.html": "2026-01-01"}


def _crawler(tmp_path, handler):
    c = Crawler(tmp_path, delay=0, client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False))
    return c


def test_fetch_ok_cache_and_conditional(tmp_path):
    calls = []

    def h(req):
        calls.append(dict(req.headers))
        if req.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, text="<html>ok</html>", headers={"content-type": "text/html", "etag": '"v1"'})

    c = _crawler(tmp_path, h)
    r = c.fetch("https://webim.ru/kb/a.html")
    assert r.status == 200 and r.html and not r.from_cache
    assert c.fetch("https://webim.ru/kb/a.html").from_cache  # из кэша без запроса
    assert len(calls) == 1
    r3 = c.fetch("https://webim.ru/kb/a.html", force=True)  # условный запрос -> 304 -> кэш
    assert r3.html == "<html>ok</html>" and len(calls) == 2


def test_redirect_within_scope_and_out_of_scope(tmp_path):
    def h(req):
        if req.url.path == "/kb/old.html":
            return httpx.Response(301, headers={"location": "/kb/new.html"})
        if req.url.path == "/kb/evil.html":
            return httpx.Response(302, headers={"location": "https://evil.com/steal"})
        return httpx.Response(200, text="<html>new</html>", headers={"content-type": "text/html"})

    c = _crawler(tmp_path, h)
    r = c.fetch("https://webim.ru/kb/old.html")
    assert r.status == 200 and r.final_url == "https://webim.ru/kb/new.html"
    r = c.fetch("https://webim.ru/kb/evil.html")
    assert r.status != 200 and r.html is None and "за пределы" in (r.error or "")


def test_failed_requests_retry_and_negative_cache(tmp_path, monkeypatch):
    n = {"503": 0, "404": 0}

    def h(req):
        if req.url.path == "/kb/flaky.html":
            n["503"] += 1
            return httpx.Response(503) if n["503"] < 2 else httpx.Response(200, text="<html>x</html>", headers={"content-type": "text/html"})
        if req.url.path == "/kb/gone.html":
            n["404"] += 1
            return httpx.Response(404)
        raise httpx.ConnectError("boom")

    c = _crawler(tmp_path, h)
    c.max_retries = 3
    import webim_ai.crawler as cm
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    assert c.fetch("https://webim.ru/kb/flaky.html").status == 200 and n["503"] == 2
    assert c.fetch("https://webim.ru/kb/gone.html").status == 404
    assert c.fetch("https://webim.ru/kb/gone.html").status == 404 and n["404"] == 1  # негативный кэш
    r = c.fetch("https://webim.ru/kb/net.html")
    assert r.status == 0 and r.html is None  # сетевой сбой не роняет краулер


def test_out_of_scope_never_requested(tmp_path):
    def h(req):
        raise AssertionError("не должно быть запросов")

    c = _crawler(tmp_path, h)
    assert c.fetch("https://evil.com/kb/x.html").status == 0
    assert c.fetch("https://webim.ru/blog/x.html").status == 0


def test_discover_uses_live_sitemap_then_fallback(tmp_path, monkeypatch):
    def h(req):
        if req.url.path == "/kb/sitemap.xml":
            return httpx.Response(200, text="<urlset><url><loc>https://webim.ru/kb/z.html</loc></url></urlset>")
        return httpx.Response(404)

    c = _crawler(tmp_path, h)
    assert "https://webim.ru/kb/z.html" in c.discover()

    def down(req):
        return httpx.Response(500)

    import webim_ai.crawler as cm
    monkeypatch.setattr(cm.time, "sleep", lambda s: None)
    c2 = _crawler(tmp_path / "2", down)
    f = tmp_path / "fallback.xml"
    f.write_text("<urlset><url><loc>https://webim.ru/kb/old.html</loc></url></urlset>")
    assert "https://webim.ru/kb/old.html" in c2.discover([f])
