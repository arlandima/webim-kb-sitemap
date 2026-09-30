import httpx

from conftest import FakeEmbedder, make_page
from webim_ai.config import Settings
from webim_ai.crawler import Crawler
from webim_ai.store import Store
from webim_ai.sync import run_sync

BASE = "https://webim.ru/kb/"


class Site:
    """Мини-«сайт» для MockTransport: страницы и sitemap можно менять между запусками."""
    def __init__(self):
        self.pages = {}
        self.lastmod = {}
        self.hits = []

    def handler(self, req: httpx.Request):
        self.hits.append(req.url.path)
        if req.url.path == "/kb/sitemap.xml":
            items = "".join(f"<url><loc>https://webim.ru{p}</loc><lastmod>{self.lastmod.get(p, '2026-01-01')}</lastmod></url>" for p in self.pages if p != "/kb/")
            return httpx.Response(200, text=f"<urlset>{items}</urlset>")
        if req.url.path in self.pages:
            return httpx.Response(200, text=self.pages[req.url.path], headers={"content-type": "text/html"})
        return httpx.Response(404)


def setup(tmp_path):
    site = Site()
    site.pages["/kb/"] = make_page("Главная", [], canonical=BASE)
    site.pages["/kb/a.html"] = make_page("Статья А", [("Раздел один", "s1", "Текст первого раздела статьи А про чаты и операторов.")], canonical=BASE + "a.html")
    site.pages["/kb/b.html"] = make_page("Статья Б", [("Раздел два", "s2", "Текст второго раздела статьи Б про боты и сценарии.")], canonical=BASE + "b.html")
    crawler = Crawler(tmp_path / "cache", delay=0, client=httpx.Client(transport=httpx.MockTransport(site.handler), follow_redirects=False))
    store = Store(":memory:")
    settings = Settings(data_dir=tmp_path)
    return site, crawler, store, settings


def sync(site, crawler, store, settings, emb, **kw):
    return run_sync(settings, store, crawler, emb, follow_links=False, log=lambda *_: None, extra_sitemaps=[], **kw)


def test_full_incremental_cycle(tmp_path):
    site, crawler, store, settings = setup(tmp_path)
    emb = FakeEmbedder()

    s1 = sync(site, crawler, store, settings, emb)
    assert s1["new"] == 3 and s1["updated"] == 0 and s1["failed"] == 0  # главная + А + Б
    n_chunks = store.counts()["chunks"]
    assert n_chunks >= 3 and emb.calls == store.counts()["embedded_chunks"]

    # ничего не изменилось: ни перезагрузки страниц, ни повторных эмбеддингов
    calls_before, hits_before = emb.calls, len(site.hits)
    s2 = sync(site, crawler, store, settings, emb)
    assert s2["new"] == 0 and s2["updated"] == 0 and s2["unchanged"] == 3 and s2["chunks_updated"] == 0
    assert emb.calls == calls_before
    assert site.hits[hits_before:] == ["/kb/sitemap.xml"]  # только sitemap

    # страница Б изменилась (lastmod другой) -> обновляется только она, эмбеддится только изменённое
    site.pages["/kb/b.html"] = make_page("Статья Б", [("Раздел два", "s2", "Совершенно новый текст про интеграции и вебхуки.")], canonical=BASE + "b.html")
    site.lastmod["/kb/b.html"] = "2026-02-02"
    s3 = sync(site, crawler, store, settings, emb)
    assert s3["updated"] == 1 and s3["new"] == 0 and s3["unchanged"] == 2
    assert 0 < emb.calls - calls_before <= 2
    hit = store.lexical("вебхуки")
    assert hit and not store.lexical("сценарии")  # старый текст исчез из индекса

    # новая страница
    site.pages["/kb/c.html"] = make_page("Статья В", [("Раздел три", "s3", "Про каналы Telegram и WhatsApp.")], canonical=BASE + "c.html")
    s4 = sync(site, crawler, store, settings, emb)
    assert s4["new"] == 1 and s4["updated"] == 0

    # удалённая страница
    del site.pages["/kb/a.html"]
    s5 = sync(site, crawler, store, settings, emb)
    assert s5["removed"] == 1
    assert BASE + "a.html" not in store.all_article_urls()
    assert not store.lexical("операторов")
    assert store.counts()["articles"] == 3


def test_failed_page_is_kept_not_deleted(tmp_path):
    site, crawler, store, settings = setup(tmp_path)
    sync(site, crawler, store, settings, None, embed=False)
    # страница «падает» (500), но остаётся в sitemap -> данные не теряем
    orig = site.handler

    def flaky(req):
        if req.url.path == "/kb/a.html":
            return httpx.Response(500)
        return orig(req)

    site.lastmod["/kb/a.html"] = "2026-03-03"
    crawler.client = httpx.Client(transport=httpx.MockTransport(flaky), follow_redirects=False)
    import webim_ai.crawler as cm
    cm.time.sleep = lambda s: None
    crawler.cache_dir  # кэш остаётся, но force-обновление по новому lastmod упадёт
    s = sync(site, crawler, store, settings, None, embed=False)
    assert BASE + "a.html" in store.all_article_urls()
    assert s["removed"] == 0


def test_alias_duplicate_not_indexed_twice(tmp_path):
    site, crawler, store, settings = setup(tmp_path)
    site.pages["/kb/alias.html"] = site.pages["/kb/a.html"]  # то же содержимое и canonical
    s = sync(site, crawler, store, settings, None, embed=False)
    assert store.counts()["articles"] == 3
    assert s["duplicates"] + s["new"] >= 3


def test_rechunk_from_stored_articles_without_network(tmp_path):
    from webim_ai.sync import rechunk_all
    site, crawler, store, settings = setup(tmp_path)
    sync(site, crawler, store, settings, None, embed=False)
    before = {r["text"] for r in store.conn.execute("SELECT text FROM chunks")}
    hits = len(site.hits)
    assert rechunk_all(store, log=lambda *_: None) == 3
    after = {r["text"] for r in store.conn.execute("SELECT text FROM chunks")}
    assert before == after and len(site.hits) == hits  # сеть не использовалась
    assert store.lexical("вебхуки") == [] and store.lexical("боты")
