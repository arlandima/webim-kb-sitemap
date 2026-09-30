"""Гибридный поиск: лексика (FTS5/BM25 по стеммам) + семантика (эмбеддинги) -> RRF -> реранкер -> доказательства с provenance."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .store import Store
from .text import highlight_terms, normalize, query_terms, stem_token

_ENTITY = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    url: str
    source_url: str  # URL с якорем на точный раздел
    anchor: str
    title: str
    heading_path: list[str]
    breadcrumbs: list[str]
    text: str
    excerpt: str
    highlights: list[list[int]]
    updated: str | None
    scores: dict = field(default_factory=dict)
    score: float = 0.0  # итоговая релевантность 0..1 (реранкер или эвристика)

    def to_dict(self, full_text: bool = True) -> dict:
        d = asdict(self)
        if not full_text:
            d.pop("text")
        return d


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]  # чанки, упорядоченные по релевантности
    articles: list[dict]  # статьи (лучший чанк на статью)
    confident: bool
    confidence: float
    timings: dict
    modes: dict
    unknown_terms: list[str] = field(default_factory=list)  # названия из вопроса, которых нет нигде в Базе знаний


def make_excerpt(text: str, query: str, width: int = 340) -> tuple[str, list[list[int]]]:
    """Окно текста вокруг наибольшего скопления слов запроса + диапазоны подсветки."""
    text = text.strip()
    flat = re.sub(r"\s+", " ", text)
    spans = highlight_terms(flat, query)
    if len(flat) <= width:
        return flat, [[a, b] for a, b in spans]
    best, best_start = -1, 0
    starts = [0] + [a for a, _ in spans]
    for st in starts:
        st = max(0, st - 60)
        cnt = sum(1 for a, b in spans if st <= a < st + width)
        if cnt > best:
            best, best_start = cnt, st
    st = best_start
    if st > 0:  # выравниваем по границе слова/предложения
        m = re.search(r"[.!?]\s+", flat[max(0, st - 80):st + 20])
        if m:
            st = max(0, st - 80) + m.end()
        else:
            sp = flat.find(" ", st)
            st = sp + 1 if 0 <= sp < st + 30 else st
    end = min(len(flat), st + width)
    sp = flat.rfind(" ", st, end)
    if end < len(flat) and sp > st + width * 0.6:
        end = sp
    ex = flat[st:end]
    pre = "… " if st > 0 else ""
    post = " …" if end < len(flat) else ""
    hl = [[a - st + len(pre), b - st + len(pre)] for a, b in spans if a >= st and b <= end]
    return pre + ex + post, hl


def _coverage(qterms: set[str], text: str) -> float:
    """Доля (стеммированных) слов запроса, присутствующих в тексте."""
    if not qterms:
        return 0.0
    from .text import tokens
    have = set(tokens(text))
    return len(qterms & have) / len(qterms)


class Retriever:
    def __init__(self, store: Store, embedder=None, reranker=None, rerank_candidates: int = 20):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.ids: list[int] = []
        self.matrix: np.ndarray | None = None
        self.head_ids: list[int] = []
        self.head_matrix: np.ndarray | None = None
        self.reload()

    def reload(self) -> None:
        if self.embedder is not None:
            self.ids, self.matrix = self.store.load_matrix(self.embedder.name)
            self.head_ids, self.head_matrix = self.store.load_matrix(self.embedder.name + "#head", "head_hash")
        else:
            self.ids, self.matrix = [], None
            self.head_ids, self.head_matrix = [], None
        self._meta = {}

    # ---------- этапы ----------
    def semantic(self, query: str, limit: int = 40) -> tuple[list[tuple[int, float]], float, list[tuple[int, float]]]:
        if self.embedder is None or self.matrix is None or not len(self.ids):
            return [], 0.0, []
        t = time.perf_counter()
        qv = self.embedder.embed_query(query)
        t_embed = (time.perf_counter() - t) * 1000
        sims = self.matrix @ qv
        top = np.argpartition(-sims, min(limit, len(sims) - 1))[:limit]
        top = top[np.argsort(-sims[top])]
        head: list[tuple[int, float]] = []
        if self.head_matrix is not None:  # заголовочный сигнал; считается тем же вектором запроса
            hs = self.head_matrix @ qv
            ht = np.argpartition(-hs, min(limit, len(hs) - 1))[:limit]
            ht = ht[np.argsort(-hs[ht])]
            head = [(self.head_ids[i], float(hs[i])) for i in ht]
        return [(self.ids[i], float(sims[i])) for i in top], t_embed, head

    @staticmethod
    def fuse(lex: list[tuple[int, float]], sem: list[tuple[int, float]], w_lex: float = 1.0, w_sem: float = 1.0,
             head: list[tuple[int, float]] | None = None, w_head: float = 0.8) -> dict[int, dict]:
        """Reciprocal Rank Fusion: сумма w/(k+rank) по обоим спискам."""
        out: dict[int, dict] = {}
        for r, (cid, s) in enumerate(lex, 1):
            e = out.setdefault(cid, {"rrf": 0.0})
            e["rrf"] += w_lex / (RRF_K + r)
            e["lex_rank"], e["lex_score"] = r, s
        for r, (cid, s) in enumerate(sem, 1):
            e = out.setdefault(cid, {"rrf": 0.0})
            e["rrf"] += w_sem / (RRF_K + r)
            e["sem_rank"], e["cos"] = r, s
        for r, (cid, s) in enumerate(head or [], 1):
            e = out.setdefault(cid, {"rrf": 0.0})
            e["rrf"] += w_head / (RRF_K + r)
            e["head_rank"], e["head_cos"] = r, s
        return out

    def search(self, query: str, k: int = 8, rerank: bool = True, articles_n: int = 8) -> SearchResult:
        t_all = time.perf_counter()
        query = re.sub(r"\s+", " ", query or "").strip()[:500]
        timings: dict = {}
        if not query:
            return SearchResult(query, [], [], False, 0.0, {}, {})

        t = time.perf_counter()
        lex = self.store.lexical(query, 40)
        timings["lexical_ms"] = round((time.perf_counter() - t) * 1000, 1)

        t = time.perf_counter()
        sem, t_embed, head = self.semantic(query, 40)
        timings["semantic_ms"] = round((time.perf_counter() - t) * 1000, 1)
        timings["query_embedding_ms"] = round(t_embed, 1)

        t = time.perf_counter()
        fused = self.fuse(lex, sem, head=head)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1]["rrf"])
        cand_ids = [cid for cid, _ in ranked[:30]]
        rows = self.store.chunk_rows(cand_ids)
        timings["fusion_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # реранкинг
        rr_scores: dict[int, float] = {}
        if rerank and self.reranker is not None and cand_ids:
            t = time.perf_counter()
            top = [c for c in cand_ids[: self.rerank_candidates] if c in rows]
            docs = [self._rerank_doc(rows[c]) for c in top]
            for c, s in zip(top, self.reranker.score(query, docs)):
                rr_scores[c] = s
            timings["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)

        hits = self._build_hits(query, cand_ids, rows, fused, rr_scores)
        hits.sort(key=lambda h: -h.score)
        conf = self._confidence(query, hits, bool(rr_scores))
        unknown = self.unknown_terms(query)
        if unknown:  # вопрос про сущность, которой нет в документации — не отвечаем по «похожему»
            conf = (False, conf[1])
        hits = self._diversify(hits, k)

        art: dict[str, dict] = {}
        for h in sorted(self._all_hits(query, cand_ids, rows, fused, rr_scores), key=lambda h: -h.score):
            a = art.get(h.url)
            if a is None:
                if len(art) >= articles_n:
                    continue
                art[h.url] = {"url": h.url, "title": h.title, "breadcrumbs": h.breadcrumbs, "updated": h.updated, "score": h.score,
                              "best": h.to_dict(full_text=False), "sections": [{"heading_path": h.heading_path, "source_url": h.source_url}]}
            elif len(a["sections"]) < 3 and all(s["source_url"] != h.source_url for s in a["sections"]):
                a["sections"].append({"heading_path": h.heading_path, "source_url": h.source_url})
        timings["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
        modes = {"lexical": True, "semantic": self.matrix is not None, "rerank": bool(rr_scores)}
        return SearchResult(query, hits, list(art.values()), conf[0], conf[1], timings, modes, unknown)

    def unknown_terms(self, query: str) -> list[str]:
        """Латинские «названия» в вопросе (с заглавной буквы не в начале или с цифрой), которых нет во всём индексе: Bitrix24, Signal…"""
        out = []
        for m in _ENTITY.finditer(query):
            w = m.group(0)
            if not (any(c.isdigit() for c in w) or (w[0].isupper() and m.start() > 0)):
                continue
            tok = stem_token(normalize(w))
            if self._df(tok) == 0:
                out.append(w)
        return out

    def _df(self, tok: str) -> int:
        cache = self.__dict__.setdefault("_df_cache", {})
        if tok not in cache:
            try:
                cache[tok] = self.store.conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ('"%s"' % tok.replace('"', ""),)).fetchone()[0]
            except Exception:
                cache[tok] = 1
        return cache[tok]

    # ---------- внутреннее ----------
    def _rerank_doc(self, r) -> str:
        hp = json.loads(r["heading_path"])
        return f"{r['title']} › {' › '.join(hp[1:])}\n{r['text']}" if len(hp) > 1 else f"{r['title']}\n{r['text']}"

    def _crumbs(self, url: str) -> tuple[list[str], str | None]:
        if not hasattr(self, "_meta"):
            self._meta = {}
        m = self._meta.get(url)
        if m is None:
            r = self.store.conn.execute("SELECT breadcrumbs, updated FROM articles WHERE url=?", (url,)).fetchone()
            m = ([c["title"] for c in json.loads(r["breadcrumbs"])], r["updated"]) if r else ([], None)
            self._meta[url] = m
        return m

    def _build_hits(self, query, cand_ids, rows, fused, rr) -> list[Hit]:
        return self._all_hits(query, cand_ids, rows, fused, rr)

    def _all_hits(self, query, cand_ids, rows, fused, rr) -> list[Hit]:
        hits = []
        max_rrf = max((fused[c]["rrf"] for c in cand_ids), default=1.0) or 1.0
        qterms = set(query_terms(query))
        for c in cand_ids:
            r = rows.get(c)
            if r is None:
                continue
            f = fused[c]
            hp = json.loads(r["heading_path"])
            crumbs, updated = self._crumbs(r["url"])
            ex, hl = make_excerpt(r["text"], query)
            if c in rr:
                score = rr[c]
            else:  # без реранкера: эвристика из семантики и лексики
                cos = f.get("cos")
                lex = f["rrf"] / max_rrf
                if self.matrix is None:  # только лексика: доля слов запроса, найденных в тексте чанка
                    cov = _coverage(qterms, r["title"] + " " + " ".join(hp) + " " + r["text"])
                    f["coverage"] = round(cov, 2)
                    score = 0.7 * cov + 0.3 * lex
                else:
                    sem = 0.0 if cos is None else max(0.0, min(1.0, (cos - 0.75) / 0.15))
                    score = 0.6 * sem + 0.4 * lex * (1.0 if "lex_rank" in f else 0.5)
            # чанки, не вошедшие в реранкинг, не могут обогнать проранжированные
            if rr and c not in rr:
                score = min(score, 0.001)
            hits.append(Hit(c, r["url"], r["url"] + (f"#{r['anchor']}" if r["anchor"] else ""), r["anchor"], r["title"], hp, crumbs,
                            r["text"], ex, hl, updated, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}, round(score, 4)))
        return hits

    def _confidence(self, query: str, hits: list[Hit], reranked: bool) -> tuple[bool, float]:
        if not hits:
            return False, 0.0
        top = hits[0]
        if reranked:
            sc = top.scores
            backed = sc.get("lex_rank", 99) <= 2 or sc.get("sem_rank", 99) == 1 or sc.get("head_rank", 99) == 1
            return top.score >= RERANK_MIN or (top.score >= RERANK_WEAK and backed), top.score
        if self.matrix is None:
            return top.scores.get("coverage", 0.0) >= 0.6 and "lex_rank" in top.scores, top.score
        cos = top.scores.get("cos", 0.0)
        return (cos >= 0.80 and "lex_rank" in top.scores) or cos >= 0.86, top.score

    @staticmethod
    def _diversify(hits: list[Hit], k: int) -> list[Hit]:
        out, per_url = [], {}
        for h in hits:
            if per_url.get(h.url, 0) >= 2:
                continue
            per_url[h.url] = per_url.get(h.url, 0) + 1
            out.append(h)
            if len(out) >= k:
                break
        return out


RERANK_MIN = 0.12  # порог уверенного попадания; калибровка по eval/questions.json (см. docs/ARCHITECTURE.md)
RERANK_WEAK = 0.05  # слабый сигнал реранкера допустим, только если лучший чанк независимо подтверждён лексикой/семантикой
