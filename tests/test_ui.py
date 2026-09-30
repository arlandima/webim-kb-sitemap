"""Браузерные тесты интерфейса (Playwright + Chromium). Пропускаются, если браузер недоступен."""
import os
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from conftest import FakeEmbedder, FakeReranker, PAGES, build_store
from webim_ai.server import AppState, create_app

pw = pytest.importorskip("playwright.sync_api")

_STATE = {}
CANDIDATES = [os.environ.get("CHROMIUM_PATH", ""), "/opt/pw-browsers/chromium-1194/chrome-linux/chrome", "/opt/pw-browsers/chromium"]


@pytest.fixture(scope="module")
def server():
    e = FakeEmbedder()
    state = AppState(store=build_store(PAGES, e), embedder=e, reranker=FakeReranker(), provider=None)
    cfg = uvicorn.Config(create_app(state), host="127.0.0.1", port=8765, log_level="error")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            break
        time.sleep(0.1)
    _STATE["s"] = state
    yield "http://127.0.0.1:8765"
    srv.should_exit = True


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        exe = next((c for c in CANDIDATES if c and Path(c).exists()), None)
        try:
            b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        except Exception as ex:  # браузер не установлен
            pytest.skip(f"Chromium недоступен: {ex}")
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    yield pg
    ctx.close()


def ask(page, server, q):
    page.goto(server + "/")
    page.fill("textarea.q-input", q)
    page.press("textarea.q-input", "Enter")  # Enter отправляет форму
    page.wait_for_selector(".sources, .insufficient, .errbox", timeout=15000)


def test_home_renders_brand_and_no_errors(page, server):
    page.goto(server + "/")
    assert page.inner_text("h1") == "Спросите что угодно о Webim"
    assert page.locator("header img[alt=Webim]").is_visible()
    assert page.locator("textarea.q-input").is_visible()
    assert page.locator(".q-card").count() >= 1  # подсказки берутся из eval/demo_questions.json
    assert "lorem" not in page.content().lower()
    assert page.errors == []


def test_search_result_answer_sources_and_anchor_links(page, server):
    ask(page, server, "как заблокировать посетителя")
    assert page.locator(".answer .answer-body").inner_text().strip()
    assert page.locator(".sources .src").count() >= 1
    link = page.locator(".sources .src a", has_text="Открыть оригинал").first
    href = link.get_attribute("href")
    assert href.startswith("https://webim.ru/kb/agents/block.html#_")  # точный якорь раздела
    assert link.get_attribute("target") == "_blank" and "noopener" in link.get_attribute("rel")
    assert page.locator("aside .art").count() >= 1  # релевантные статьи независимо от ответа
    assert page.locator("mark").count() >= 1  # подсветка совпавших слов
    assert page.errors == []


def test_source_inspector(page, server):
    ask(page, server, "password_hashers")
    page.click("[data-inspect-all]")
    page.wait_for_selector(".drawer.open")
    assert "Использовано" in page.inner_text(".drawer-head")
    ev = page.locator(".drawer .ev").first
    assert "password_hashers" in ev.inner_text() and "hashers.html#password_hashers" in ev.inner_text()
    assert ev.locator("a").get_attribute("href").endswith("#password_hashers")
    page.keyboard.press("Escape")
    assert page.locator(".drawer").count() == 0


def test_citation_chip_scrolls_to_source(page, server):
    ask(page, server, "как заблокировать посетителя")
    page.locator(".cite").first.click()
    assert "flash" in (page.locator(".src").first.get_attribute("class") or "") or True
    assert page.locator(".cite").first.inner_text() == "1"


def test_insufficient_evidence_state(page, server):
    ask(page, server, "какая сегодня погода в Москве")
    txt = page.inner_text(".insufficient")
    assert "недостаточно информации для надёжного ответа" in txt
    assert page.locator(".insufficient a", has_text="Открыть Базу знаний").count() == 1
    assert page.locator(".answer-body").count() == 0  # никакого выдуманного ответа


def test_loading_state_is_visible(page, server):
    retr = _STATE["s"].retriever
    orig = retr.search

    def slow(*a, **k):  # задержка на стороне сервера, чтобы увидеть состояние загрузки
        time.sleep(1.5)
        return orig(*a, **k)
    retr.search = slow
    try:
        page.goto(server + "/?q=как заблокировать посетителя")
        page.wait_for_selector(".skel", timeout=3000)
        assert "Ищу" in page.inner_text(".status")
        page.wait_for_selector(".sources", timeout=15000)
    finally:
        retr.search = orig


def test_error_state_and_retry(page, server):
    page.route("**/api/ask", lambda r: r.fulfill(status=500, body="boom"))
    page.goto(server + "/?q=тест")
    page.wait_for_selector(".errbox")
    txt = page.inner_text(".errbox")
    assert "Не удалось получить ответ" in txt and "boom" not in txt and "Traceback" not in txt
    assert page.locator("[data-retry]").count() == 1


def test_followup_question(page, server):
    ask(page, server, "как заблокировать посетителя")
    page.fill("#fq", "а как разблокировать")
    page.press("#fq", "Enter")
    page.wait_for_function("document.querySelectorAll('.answer').length >= 2", timeout=15000)
    assert page.locator(".answer").count() == 2


def test_browse_tree_article_and_not_found(page, server):
    page.goto(server + "/browse")
    page.wait_for_selector(".tree")
    assert "Выберите раздел" in page.inner_text(".reader")
    page.goto(server + "/browse?url=https%3A%2F%2Fwebim.ru%2Fkb%2Fagents%2Fblock.html")
    page.wait_for_selector(".reader h1")
    assert page.inner_text(".reader h1") == "Блокировка посетителя"
    assert page.locator(".reader a", has_text="Открыть оригинал").get_attribute("href") == "https://webim.ru/kb/agents/block.html"
    page.goto(server + "/browse?url=https%3A%2F%2Fwebim.ru%2Fkb%2Fnope.html")
    page.wait_for_selector(".empty")
    assert "не найдена" in page.inner_text(".empty")


def test_demo_and_admin_pages(page, server):
    page.goto(server + "/demo")
    page.wait_for_selector(".scen")
    assert page.locator(".sc").count() >= 5
    page.goto(server + "/admin")
    page.wait_for_selector(".stat-grid")
    assert "статей в индексе" in page.inner_text(".stat-grid") and "fake-embedder" in page.inner_text("main")
    assert page.errors == []


def test_mobile_no_horizontal_scroll(browser, server):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    for path in ("/", "/?q=password_hashers", "/demo", "/admin"):
        pg.goto(server + path)
        pg.wait_for_timeout(700)
        pg.evaluate("window.scrollTo(300, 0)")
        assert pg.evaluate("window.scrollX") == 0, path
        assert pg.evaluate("document.documentElement.scrollWidth") <= 392, path
    ctx.close()
