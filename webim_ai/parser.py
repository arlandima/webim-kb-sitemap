"""Разбор HTML статьи MkDocs Material -> структурированная статья с секциями, якорями и происхождением (provenance)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict

from bs4 import BeautifulSoup, NavigableString, Tag

from .crawler import canonicalize

MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8,
          "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
_DROP = ("script", "style", "nav", "svg", "noscript", "iframe", "form", "button", "template")


@dataclass
class Section:
    level: int  # 1 = h1, 2 = h2 ...
    heading: str
    anchor: str  # id заголовка; "" для введения без id
    path: list[str]  # заголовки от h1 до текущего
    blocks: list[dict] = field(default_factory=list)

    def text(self) -> str:
        return blocks_to_text(self.blocks)


@dataclass
class Article:
    url: str
    title: str
    breadcrumbs: list[dict]  # [{title,url}] от корня к родителю (без самой статьи)
    updated: str | None
    sections: list[Section]
    links: list[str]
    content_hash: str
    description: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _clean(s: str) -> str:
    return re.sub(r"[ \t\r\f\v  ]+", " ", s or "").strip()


def inline_text(el: Tag) -> str:
    parts = []
    for c in el.descendants:
        if isinstance(c, NavigableString):
            p = c.parent
            if p is not None and p.name in _DROP:
                continue
            parts.append(str(c))
        elif isinstance(c, Tag) and c.name == "br":
            parts.append(" ")
    return _clean(re.sub(r"\s+", " ", "".join(parts)))


def parse_ru_date(s: str) -> str | None:
    m = re.search(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", (s or "").lower())
    if m and m.group(2) in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s or "")
    return m.group(0) if m else None


def blocks_to_text(blocks: list[dict]) -> str:
    out = []
    for b in blocks:
        t = b["type"]
        if t in ("p", "note", "figure", "quote"):
            pre = f"{b['title']}: " if b.get("title") else ""
            out.append(pre + b["text"])
        elif t == "list":
            for i, it in enumerate(b["items"]):
                out.append(("  " * it.get("depth", 0)) + (f"{i + 1}. " if b.get("ordered") and not it.get("depth") else "- ") + it["text"])
        elif t == "code":
            out.append("```\n" + b["text"] + "\n```")
        elif t == "table":
            for row in b["rows"]:
                out.append(" | ".join(row))
    return "\n".join(out)


class _Walker:
    def __init__(self, title: str):
        self.sections: list[Section] = [Section(1, title, "", [title])]
        self.stack: list[Section] = [self.sections[0]]
        self.links: list[str] = []

    @property
    def cur(self) -> Section:
        return self.sections[-1]

    def add(self, block: dict) -> None:
        self.cur.blocks.append(block)

    def heading(self, el: Tag) -> None:
        level = int(el.name[1])
        text = inline_text(el)
        if not text:
            return
        anchor = el.get("id", "") or ""
        while self.stack and self.stack[-1].level >= level:
            self.stack.pop()
        parent_path = self.stack[-1].path if self.stack else []
        sec = Section(level, text, anchor, parent_path + [text])
        self.sections.append(sec)
        self.stack.append(sec)

    def walk(self, node: Tag) -> None:
        for el in node.children:
            if isinstance(el, NavigableString):
                continue
            if not isinstance(el, Tag):
                continue
            n = el.name
            cls = el.get("class") or []
            if n in _DROP or "headerlink" in cls or "md-source-file" in cls or "md-content__button" in cls:
                continue
            if n in ("h2", "h3", "h4", "h5", "h6"):
                self.heading(el)
            elif n == "p":
                t = inline_text(el)
                if t:
                    self.add({"type": "p", "text": t})
                self._links(el)
            elif n in ("ul", "ol"):
                items: list[dict] = []
                self._list(el, items, 0)
                if items:
                    self.add({"type": "list", "ordered": n == "ol", "items": items})
                self._links(el)
            elif n == "pre":
                code = el.get_text("\n") if not el.find("code") else el.find("code").get_text()
                code = re.sub(r"\n{3,}", "\n\n", code).strip("\n")
                if code.strip():
                    self.add({"type": "code", "text": code[:6000]})
            elif n == "table":
                rows = []
                for tr in el.find_all("tr"):
                    cells = [inline_text(c) for c in tr.find_all(["th", "td"])]
                    if any(cells):
                        rows.append(cells)
                if rows:
                    self.add({"type": "table", "rows": rows})
                self._links(el)
            elif n == "figure":
                cap = el.find("figcaption")
                img = el.find("img")
                t = inline_text(cap) if cap else (img.get("alt", "") if img else "")
                if t:
                    self.add({"type": "figure", "text": _clean(t)})
            elif n == "img":
                alt = _clean(el.get("alt", ""))
                if alt:
                    self.add({"type": "figure", "text": alt})
            elif n == "blockquote":
                t = inline_text(el)
                if t:
                    self.add({"type": "quote", "text": t})
            elif n in ("div", "details", "section", "aside") and ("admonition" in cls or n == "details"):
                title_el = el.find(class_="admonition-title") or el.find("summary")
                title = inline_text(title_el) if title_el else ""
                if title_el is not None:
                    title_el.extract()
                # вложенные списки/код внутри примечаний тоже сохраняем текстом
                t = "\n".join(x for x in (self._flat(el)) if x)
                if t:
                    self.add({"type": "note", "title": title, "text": t})
                self._links(el)
            elif n == "dl":
                for dt in el.find_all(["dt", "dd"]):
                    t = inline_text(dt)
                    if t:
                        self.add({"type": "p", "text": t})
            else:
                self.walk(el)

    def _flat(self, el: Tag) -> list[str]:
        out = []
        for c in el.children:
            if isinstance(c, NavigableString):
                continue
            if not isinstance(c, Tag) or c.name in _DROP:
                continue
            if c.name in ("ul", "ol"):
                items: list[dict] = []
                self._list(c, items, 0)
                out += [("- " + i["text"]) for i in items]
            elif c.name == "pre":
                out.append(c.get_text().strip())
            elif c.name == "table":
                for tr in c.find_all("tr"):
                    out.append(" | ".join(inline_text(x) for x in tr.find_all(["th", "td"])))
            elif c.name in ("p", "span", "code", "strong", "em", "a", "h2", "h3", "h4", "figure"):
                out.append(inline_text(c))
            else:
                out += self._flat(c)
        return out

    def _list(self, ul: Tag, items: list[dict], depth: int) -> None:
        for li in ul.find_all("li", recursive=False):
            direct = []
            for c in li.children:
                if isinstance(c, Tag) and c.name in ("ul", "ol"):
                    continue
                if isinstance(c, Tag) and c.name in _DROP:
                    continue
                direct.append(inline_text(c) if isinstance(c, Tag) else _clean(str(c)))
            txt = _clean(" ".join(x for x in direct if x))
            if txt:
                items.append({"text": txt, "depth": depth})
            for sub in li.find_all(["ul", "ol"], recursive=False):
                self._list(sub, items, depth + 1)

    def _links(self, el: Tag) -> None:
        for a in el.find_all("a", href=True):
            self.links.append(a["href"])


def parse_article(html: str, url: str) -> Article | None:
    soup = BeautifulSoup(html, "lxml")
    art = soup.select_one("article.md-content__inner") or soup.find("article")
    if art is None:
        return None
    canon = soup.find("link", rel="canonical")
    canon_url = canonicalize(canon["href"]) if canon and canon.get("href") else None
    url = canon_url or canonicalize(url) or url

    for el in art.select(".headerlink, .md-source-file__fact a"):  # значок «¶» не должен попадать в заголовки
        el.decompose()
    h1 = art.find("h1")
    title = inline_text(h1) if h1 else ""
    if not title:
        t = soup.find("title")
        title = _clean(t.get_text()).split(" - ")[0] if t else url
    h1_id = (h1.get("id") if h1 else "") or ""

    # дата последнего обновления
    updated = None
    src = art.select_one(".md-source-file .git-revision-date-localized-plugin-date")
    if src:
        updated = parse_ru_date(src.get("title", "") or src.get_text())

    # хлебные крошки: активные предки в боковой навигации
    crumbs: list[dict] = []
    nav = soup.select_one("nav.md-nav--primary")
    if nav:
        for li in nav.select("li.md-nav__item--active.md-nav__item--nested"):
            a = li.select_one(":scope > div.md-nav__container > a.md-nav__link") or li.select_one(":scope > a.md-nav__link")
            if a and a.get("href") is not None:
                cu = canonicalize(a["href"], base=url)
                if cu and cu != url:
                    crumbs.append({"title": inline_text(a), "url": cu})
    desc = soup.find("meta", attrs={"name": "description"})

    for el in art.select(".md-source-file, .headerlink, .md-content__button, script, style"):
        el.decompose()
    w = _Walker(title)
    w.sections[0].anchor = h1_id
    w.walk(art)
    sections = [s for s in w.sections if s.blocks or s.level > 1]
    if not sections:
        sections = [w.sections[0]]

    links = []
    for href in w.links:
        cu = canonicalize(href, base=url)
        if cu and cu not in links and cu != url:
            links.append(cu)

    body = "\n".join(f"{'#' * s.level} {s.heading}\n{s.text()}" for s in sections)
    return Article(url=url, title=title, breadcrumbs=crumbs, updated=updated, sections=sections, links=links,
                   content_hash=hashlib.sha256(body.encode()).hexdigest(),
                   description=_clean(desc["content"]) if desc and desc.get("content") else "")
