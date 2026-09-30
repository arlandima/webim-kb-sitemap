import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webim_ai.text import tokens  # noqa: E402


class FakeEmbedder:
    """Детерминированные «эмбеддинги»: хешированный мешок стеммов + небольшая таблица синонимов — для тестов слияния без загрузки модели."""
    name = "fake-embedder"
    dim = 128
    SYN = {"пожаловаться": "блок", "выгнать": "блок", "заблокировать": "блок", "бан": "блок", "ночью": "офлайн", "пауза": "удержан"}

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokens(text):
            for w in (t, self.SYN.get(t, "")):
                if w:
                    v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts, batch=16):
        self.calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    name = "fake-reranker"

    def score(self, query, docs, batch=8):
        q = set(tokens(query))
        out = []
        for d in docs:
            dt = set(tokens(d))
            out.append(min(1.0, len(q & dt) / max(1, len(q)) * 1.2))
        return out


@pytest.fixture
def fixture_html():
    return (Path(__file__).parent / "fixtures" / "article.html").read_text(encoding="utf-8")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def make_page(title, sections, canonical=None, crumbs=""):
    """Собирает мини-страницу в формате MkDocs Material: sections = [(h2, id, text)]."""
    body = "".join(f'<h2 id="{i}">{h}</h2><p>{t}</p>' for h, i, t in sections)
    canon = f'<link rel="canonical" href="{canonical}">' if canonical else ""
    return f'<html><head>{canon}</head><body><article class="md-content__inner"><h1 id="top">{title}</h1><p>Вводный текст статьи про {title}.</p>{body}</article></body></html>'


def build_store(pages: dict, embedder=None):
    """pages: {url: (title, [(h2,id,text)], crumbs_html)} -> Store в памяти с чанками и эмбеддингами."""
    from webim_ai.chunker import chunk_article
    from webim_ai.parser import parse_article
    from webim_ai.store import Store
    from webim_ai.sync import embed_hash, embed_missing

    st = Store(":memory:")
    for url, (title, secs) in pages.items():
        a = parse_article(make_page(title, secs, canonical=url), url)
        a.breadcrumbs = [{"title": "Раздел", "url": "https://webim.ru/kb/raздел/"}]
        ch = chunk_article(a)
        st.upsert_article(a, None, ch, [embed_hash(c.embed_text) for c in ch])
    if embedder is not None:
        embed_missing(st, embedder, log=lambda *_: None)
    return st


PAGES = {
    "https://webim.ru/kb/agents/block.html": ("Блокировка посетителя", [
        ("Блокировка во время диалога", "_1", "Оператор может заблокировать посетителя прямо во время диалога. Заблокированный посетитель не сможет написать в чат снова."),
        ("Разблокировка", "_2", "Чтобы разблокировать посетителя, откройте список заблокированных.")]),
    "https://webim.ru/kb/devops/hashers.html": ("Настройка хеширования паролей операторов (password_hashers)", [
        ("Параметр password_hashers", "password_hashers", "Параметр password_hashers задаёт список алгоритмов хеширования: bcrypt, pbkdf2. Значение по умолчанию — bcrypt.")]),
    "https://webim.ru/kb/devops/es.html": ("Подключение Elasticsearch или OpenSearch", [
        ("Настройка", "setup", "Для подключения укажите адрес кластера Elasticsearch в файле main.json. Поддерживается OpenSearch версии 2."),
        ("Проверка", "check", "Проверьте доступность кластера командой curl.")]),
    "https://webim.ru/kb/agents/offline.html": ("Работа с офлайн-обращениями", [
        ("Как приходят обращения", "_1", "Когда все операторы не в сети, сообщение посетителя сохраняется как офлайн-обращение и доступно в разделе обращений.")]),
    "https://webim.ru/kb/agents/hold.html": ("Чаты на удержании", [
        ("Как поставить чат на удержание", "_1", "Чтобы поставить диалог на паузу, нажмите кнопку удержания. Чат появится в списке удержанных диалогов.")]),
}
