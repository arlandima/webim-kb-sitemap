from webim_ai.chunker import chunk_article
from webim_ai.parser import parse_article, parse_ru_date


def test_parse_article_structure(fixture_html):
    a = parse_article(fixture_html, "https://webim.ru/kb/control-panel/buttons-and-locations/chat-window.html")
    assert a.title == 'Раздел "Окно чата"'
    assert a.url == "https://webim.ru/kb/control-panel/buttons-and-locations/chat-window.html"  # канонический
    assert [c["title"] for c in a.breadcrumbs] == ["Панель управления", "Кнопки и размещения"]
    assert a.breadcrumbs[0]["url"] == "https://webim.ru/kb/control-panel/"
    assert a.updated == "2024-12-05"
    heads = [(s.level, s.heading, s.anchor) for s in a.sections]
    assert heads == [(1, 'Раздел "Окно чата"', "chat-window"), (2, "Экран «Первый вопрос»", "_1"), (3, "Начало диалога", "start")]
    sub = a.sections[2]
    assert sub.path == ['Раздел "Окно чата"', "Экран «Первый вопрос»", "Начало диалога"]  # иерархия H1 > H2 > H3
    types = [b["type"] for b in sub.blocks]
    assert types == ["list", "list", "code", "table", "figure", "p"]
    assert sub.blocks[0]["items"][2] == {"text": "после отправки", "depth": 1}
    assert sub.blocks[1]["ordered"] is True
    assert sub.blocks[2]["text"] == '{"first_question_required": true}'
    assert sub.blocks[3]["rows"][1] == ["mode", "strict"]


def test_parser_drops_scripts_and_headerlinks(fixture_html):
    a = parse_article(fixture_html, "https://webim.ru/kb/x.html")
    text = "\n".join(s.text() for s in a.sections)
    assert "alert" not in text and "evil = 1" not in text and "¶" not in text
    assert "Изменения применяются сразу" in text  # admonition сохранён


def test_links_filtered_to_scope(fixture_html):
    a = parse_article(fixture_html, "https://webim.ru/kb/control-panel/buttons-and-locations/chat-window.html")
    assert a.links == ["https://webim.ru/kb/for-admins/dept.html"]  # внешняя и картинка отброшены


def test_content_hash_stable_and_sensitive(fixture_html):
    a1 = parse_article(fixture_html, "https://webim.ru/kb/x.html")
    a2 = parse_article(fixture_html, "https://webim.ru/kb/x.html")
    a3 = parse_article(fixture_html.replace("Чат создаётся", "Чат создаётся позже"), "https://webim.ru/kb/x.html")
    assert a1.content_hash == a2.content_hash != a3.content_hash


def test_parse_ru_date():
    assert parse_ru_date("16 марта 2023 г.") == "2023-03-16"
    assert parse_ru_date("мусор") is None


def test_chunk_provenance(fixture_html):
    a = parse_article(fixture_html, "https://webim.ru/kb/control-panel/buttons-and-locations/chat-window.html")
    chunks = chunk_article(a)
    assert chunks
    c = next(c for c in chunks if "Клиент вводит вопрос" in c.text)
    assert c.anchor  # якорь сохранён
    assert c.source_url.startswith("https://webim.ru/kb/control-panel/buttons-and-locations/chat-window.html#")
    assert c.heading_path[0] == 'Раздел "Окно чата"' and c.heading_path[-1] == "Начало диалога"
    assert c.title in c.embed_text and "Начало диалога" in c.embed_text  # контекст в тексте для эмбеддинга


def test_large_section_split_keeps_heading():
    from webim_ai.parser import Article, Section
    long = [{"type": "p", "text": ("Предложение номер %d про настройку. " % i) * 6} for i in range(30)]
    a = Article("https://webim.ru/kb/x.html", "T", [], None, [Section(1, "T", "t", ["T"], []), Section(2, "Большой", "big", ["T", "Большой"], long)], [], "h")
    chunks = chunk_article(a)
    assert len(chunks) > 3
    assert all(c.anchor == "big" and c.heading_path == ["T", "Большой"] for c in chunks)
    assert all(len(c.text) <= 1700 for c in chunks)
    assert [c.part for c in chunks] == list(range(len(chunks)))


def test_tiny_parent_merges_with_child():
    from webim_ai.parser import Article, Section
    a = Article("https://webim.ru/kb/x.html", "T", [], None, [
        Section(1, "T", "t", ["T"], [{"type": "p", "text": "Короткое введение."}]),
        Section(2, "Подраздел", "sub", ["T", "Подраздел"], [{"type": "p", "text": "Достаточно длинный текст подраздела. " * 8}])], [], "h")
    chunks = chunk_article(a)
    assert len(chunks) == 1 and "Короткое введение" in chunks[0].text and chunks[0].anchor == "sub"  # ссылка ведёт к конкретному подразделу
