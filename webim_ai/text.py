"""Токенизация и нормализация для лексического поиска (русский + английский + технические идентификаторы)."""
from __future__ import annotations

import re
import unicodedata

import snowballstemmer

_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[_\-.][0-9A-Za-zА-Яа-яЁё]+)*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CYR = re.compile(r"[А-Яа-яЁё]")

STOP_RU = set(
    """а без более бы был была были было быть в вам вас весь во вот все всё всего вы где да даже для до его ее её ей ему если есть еще ещё
    же за здесь и из или им их к как какая какие какой какую когда конечно кто ли либо мне может можно мой мы на над надо наш не него нее нет ни них но ну о об однако он она они оно
    от очень по под после потому при про раз с со совсем так также такой там те тем то того тоже той только том ты у уж уже хорошо хотя чего чем через что чтобы чтоб эти этого этой этом этот эту я
    нужно надо хочу хотим хочет можете могу подскажите пожалуйста""".split()
)
STOP_EN = set("a an the of to in on for is are be how do i can what which with and or".split())


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("­", "").replace(" ", " ")
    return s.lower().replace("ё", "е")


def stem_token(t: str) -> str:
    if _CYR.search(t):
        return _ru.stemWord(t)
    if t.isdigit():
        return t
    return _en.stemWord(t)


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    """Стеммированные токены. Составные идентификаторы (a_b, a.b, camelCase) дают и целое, и части."""
    out: list[str] = []
    raw = text or ""
    for m in _WORD.finditer(raw):
        w = m.group(0)
        parts = [w]
        # camelCase -> части
        camel = _CAMEL.split(w)
        if len(camel) > 1:
            parts += camel
        if re.search(r"[_\-.]", w):
            parts += re.split(r"[_\-.]", w)
        for p in parts:
            n = normalize(p)
            if not n or (len(n) < 2 and not n.isdigit()):
                continue
            if drop_stop and (n in STOP_RU or n in STOP_EN):
                continue
            out.append(stem_token(n))
    return out


def fts_index_text(text: str) -> str:
    return " ".join(tokens(text, drop_stop=False))


def query_terms(text: str) -> list[str]:
    seen, res = set(), []
    for t in tokens(text):
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res


def fts_query(text: str) -> str:
    terms = query_terms(text)
    # экранирование: каждый токен в кавычках (защита от синтаксиса FTS5)
    return " OR ".join('"' + t.replace('"', "") + '"' for t in terms)


def highlight_terms(text: str, query: str) -> list[tuple[int, int]]:
    """Диапазоны для подсветки слов запроса в исходном тексте (по совпадению стеммов)."""
    qset = set(query_terms(query))
    spans = []
    for m in re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+", text):
        n = normalize(m.group(0))
        if len(n) >= 2 and stem_token(n) in qset:
            spans.append((m.start(), m.end()))
    return spans
