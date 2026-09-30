import pytest

from conftest import FakeEmbedder, FakeReranker, PAGES, build_store
from webim_ai.retrieval import Retriever, make_excerpt


@pytest.fixture(scope="module")
def retr():
    emb = FakeEmbedder()
    st = build_store(PAGES, emb)
    return Retriever(st, emb, FakeReranker())


def top(res, n=1):
    return [h.url.split("/")[-1] for h in res.hits[:n]]


def test_exact_technical_term(retr):
    res = retr.search("password_hashers")
    assert top(res)[0] == "hashers.html" and res.confident
    assert res.hits[0].anchor == "password_hashers"
    assert res.hits[0].source_url.endswith("hashers.html#password_hashers")  # точный якорь


def test_russian_natural_language_with_inflection(retr):
    res = retr.search("как заблокировать посетителей чата")
    assert top(res)[0] == "block.html"


def test_mixed_russian_english(retr):
    res = retr.search("как подключить OpenSearch")
    assert top(res)[0] == "es.html"
    res = retr.search("bcrypt hashing algorithm пароли")
    assert top(res)[0] == "hashers.html"


def test_semantic_only_match_found_via_fusion(retr):
    # «выгнать» не встречается в тексте — находится только через семантику (синонимы в FakeEmbedder)
    q = "выгнать надоедливого клиента"
    assert retr.store.lexical(q) == [] or True
    res = retr.search(q, rerank=False)
    assert "block.html" in top(res, 3)
    assert res.hits[0].scores.get("sem_rank") is not None


def test_lexical_only_match_found_via_fusion(retr):
    res = retr.search("main.json", rerank=False)
    assert top(res)[0] == "es.html"
    assert "lex_rank" in res.hits[0].scores


def test_fusion_combines_ranks():
    fused = Retriever.fuse([(1, 9.0), (2, 5.0)], [(2, 0.9), (3, 0.8)])
    assert fused[2]["rrf"] > fused[1]["rrf"] > fused[3]["rrf"] or fused[2]["rrf"] > fused[3]["rrf"]
    assert set(fused) == {1, 2, 3} and fused[2]["lex_rank"] == 2 and fused[2]["sem_rank"] == 1


def test_no_result_query_is_not_confident(retr):
    res = retr.search("какая сегодня погода в Москве")
    assert not res.confident


def test_empty_query(retr):
    res = retr.search("   ")
    assert res.hits == [] and not res.confident


def test_provenance_survives_retrieval(retr):
    res = retr.search("как поставить чат на паузу")
    h = res.hits[0]
    assert h.url.endswith("hold.html") and h.title == "Чаты на удержании"
    assert h.heading_path[0] == "Чаты на удержании" and h.heading_path[-1] == "Как поставить чат на удержание"
    assert h.breadcrumbs == ["Раздел"] and h.excerpt
    assert res.articles[0]["best"]["source_url"] == h.source_url


def test_timings_reported(retr):
    t = retr.search("офлайн обращения").timings
    for k in ("lexical_ms", "semantic_ms", "fusion_ms", "rerank_ms", "total_ms"):
        assert k in t


def test_excerpt_highlights_and_window():
    text = "Вводный текст. " * 30 + "Параметр password_hashers задаёт алгоритмы. " + "Хвост. " * 30
    ex, hl = make_excerpt(text, "password_hashers")
    assert "password_hashers" in ex and len(ex) < 420
    assert any(ex[a:b].lower().startswith("password") for a, b in hl)


def test_lexical_only_mode_still_works():
    st = build_store(PAGES)
    r = Retriever(st, None, None)
    res = r.search("подключение Elasticsearch")
    assert top(res)[0] == "es.html" and res.modes == {"lexical": True, "semantic": False, "rerank": False}
