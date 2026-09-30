"""Секционное разбиение: чанк = раздел статьи (H1/H2/H3…) с полным путём заголовков и якорем.
Маленькие разделы объединяются со своими подразделами, большие режутся по границам блоков."""
from __future__ import annotations

from dataclasses import dataclass

from .parser import Article, Section, blocks_to_text

CHUNKER_VERSION = 3  # при изменении правил разбиения индекс пересобирается из сохранённых статей без обращения к сети
MAX_CHARS = 1600
TARGET_CHARS = 1100
MIN_CHARS = 110


@dataclass
class Chunk:
    url: str
    ord: int
    anchor: str
    heading_path: list[str]
    title: str
    text: str
    part: int = 0  # номер части в разделе, если раздел разрезан

    @property
    def source_url(self) -> str:
        return self.url + (f"#{self.anchor}" if self.anchor else "")

    @property
    def head_text(self) -> str:
        """Заголовок статьи + путь разделов — короткий текст для отдельного «заголовочного» эмбеддинга."""
        path = " › ".join(self.heading_path[1:]) if len(self.heading_path) > 1 else ""
        return f"{self.title}" + (f" › {path}" if path else "")

    @property
    def embed_text(self) -> str:
        path = " › ".join(self.heading_path[1:]) if len(self.heading_path) > 1 else ""
        head = f"{self.title}" + (f" › {path}" if path else "")
        return f"{head}\n{self.text}"


def _split_block(b: dict, limit: int) -> list[str]:
    text = blocks_to_text([b])
    if len(text) <= limit:
        return [text]
    lines, out, cur = text.split("\n"), [], ""
    for ln in lines:
        while len(ln) > limit:  # очень длинная строка (например, минифицированный JSON)
            if cur:
                out.append(cur)
                cur = ""
            out.append(ln[:limit])
            ln = ln[limit:]
        if len(cur) + len(ln) + 1 > limit and cur:
            out.append(cur)
            cur = ln
        else:
            cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        out.append(cur)
    return out


def _pack(section_blocks: list[dict]) -> list[str]:
    units: list[str] = []
    for b in section_blocks:
        units += _split_block(b, MAX_CHARS)
    parts, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > TARGET_CHARS and len(cur) >= MIN_CHARS:
            parts.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        parts.append(cur)
    return parts


def chunk_article(a: Article) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: list[tuple[Section, str]] = []  # маленькие разделы, ожидающие слияния с дочерним

    def emit(sec: Section, body: str, part: int = 0, prefix: str = "") -> None:
        text = (prefix + body).strip()
        if text:
            chunks.append(Chunk(a.url, len(chunks), sec.anchor, sec.path, a.title, text, part))

    secs = a.sections
    for i, sec in enumerate(secs):
        body = blocks_to_text(sec.blocks)
        nxt = secs[i + 1] if i + 1 < len(secs) else None
        if not body.strip():
            # пустой раздел — его заголовок попадёт в путь дочерних; но переносим для слияния
            if nxt and nxt.level > sec.level:
                pending.append((sec, ""))
            continue
        if len(body) < MIN_CHARS and nxt and nxt.level > sec.level:
            pending.append((sec, body))
            continue
        prefix = ""
        anchor_sec = sec
        if pending:
            # склеиваем накопленные короткие «родительские» разделы перед текущим
            lines = []
            for ps, pb in pending:
                if pb:
                    lines.append(f"{ps.heading}\n{pb}" if ps.level > 1 else pb)
            prefix = ("\n".join(lines) + "\n") if lines else ""
            pending = []  # якорь остаётся у самого конкретного (дочернего) раздела — ссылка ведёт точно к нему
        parts = _pack([{"type": "p", "text": prefix.rstrip()}] + sec.blocks) if prefix else _pack(sec.blocks)
        for pi, p in enumerate(parts):
            emit(anchor_sec, p, pi)
    if pending:  # хвост: короткие разделы без потомков
        lines = [f"{ps.heading}\n{pb}" for ps, pb in pending if pb]
        if lines:
            emit(pending[0][0], "\n".join(lines))
    # дедупликация полностью одинаковых частей
    return chunks
