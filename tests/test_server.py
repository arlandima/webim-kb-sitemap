import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, FakeReranker, PAGES, build_store
from webim_ai.server import AppState, create_app


@pytest.fixture(scope="module")
def client():
    e = FakeEmbedder()
    st = AppState(store=build_store(PAGES, e), embedder=e, reranker=FakeReranker(), provider=None)
    with TestClient(create_app(st)) as c:
        yield c


def sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        ev = block.split("\n")[0][7:]
        out.append((ev, json.loads(block.split("\n")[1][6:])))
    return out


def test_search_api(client):
    d = client.get("/api/search", params={"q": "password_hashers"}).json()
    assert d["confident"] and d["evidence"][0]["source_url"].endswith("#password_hashers")
    assert d["articles"][0]["title"].startswith("Настройка хеширования")
    assert "total_ms" in d["timings"]


def test_ask_stream_events_and_extractive_answer(client):
    r = client.post("/api/ask", json={"q": "как заблокировать посетителя"})
    ev = sse(r.text)
    names = [n for n, _ in ev]
    assert names[0] == "retrieval" and "final" in names and names[-1] == "done"
    fin = dict(ev)["final"]
    assert fin["mode"] == "extractive" and fin["cited"] and "[1]" in fin["text"]
    assert dict(ev)["retrieval"]["evidence"][0]["n"] == 1


def test_ask_insufficient(client):
    ev = dict(sse(client.post("/api/ask", json={"q": "какая погода в Москве"}).text))
    assert ev["final"]["mode"] == "insufficient" and "недостаточно" in ev["final"]["text"]
    assert ev["retrieval"]["articles"] is not None  # похожие материалы всё равно есть


def test_ask_followup_uses_history_for_retrieval(client):
    hist = [{"role": "user", "content": "как заблокировать посетителя"}, {"role": "assistant", "content": "…"}]
    ev = dict(sse(client.post("/api/ask", json={"q": "а как разблокировать", "history": hist}).text))
    assert ev["retrieval"]["evidence"][0]["url"].endswith("block.html")


def test_ask_validation(client):
    assert client.post("/api/ask", json={"q": ""}).status_code == 422
    assert client.post("/api/ask", json={"q": "x" * 601}).status_code == 422


def test_article_and_tree(client):
    a = client.get("/api/article", params={"url": "https://webim.ru/kb/agents/block.html"}).json()
    assert a["title"] == "Блокировка посетителя" and a["sections"][1]["anchor"] == "_1"
    assert client.get("/api/article", params={"url": "https://evil.com/x"}).status_code == 400
    assert client.get("/api/article", params={"url": "https://webim.ru/kb/nope.html"}).status_code == 404
    tree = client.get("/api/tree").json()
    assert tree and tree[0]["children"]


def test_status(client):
    s = client.get("/api/status").json()
    assert s["counts"]["articles"] == 5 and s["embedding_model"] == "fake-embedder" and s["llm"]["mode"] == "extractive"


def test_spa_fallback_and_api_404(client):
    assert client.get("/demo").status_code == 200 and "<title>" in client.get("/browse").text
    assert client.get("/api/unknown").status_code == 404


def test_sections_and_scoped_ask(client):
    secs = client.get("/api/sections").json()
    assert isinstance(secs, list) and all({"key", "title", "count"} <= set(x) for x in secs)
    ev = dict(sse(client.post("/api/ask", json={"q": "подключение кластера", "section": "devops"}).text))
    assert all("/kb/devops/" in e["url"] for e in ev["retrieval"]["evidence"])
    assert client.post("/api/ask", json={"q": "x", "section": "../etc"}).status_code == 422
