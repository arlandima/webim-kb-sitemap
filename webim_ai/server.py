"""HTTP API + статический интерфейс. Запуск: python run.py  (или uvicorn webim_ai.server:app)."""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .answer import INSUFFICIENT, answer_stream, get_provider
from .config import ROOT, get_settings
from .crawler import canonicalize
from .models import load_embedder, load_reranker
from .retrieval import Retriever, RERANK_MIN, RERANK_WEAK
from .store import Store

log = logging.getLogger("webim_ai")
WEB = ROOT / "web"


class AppState:
    def __init__(self, settings=None, store=None, embedder="auto", reranker="auto", provider="auto"):
        self.settings = settings or get_settings()
        self.store = store or Store(self.settings.db_path)
        self.warnings: list[str] = []
        self.embedder = self._load(embedder, lambda: load_embedder(self.settings.embed_model), "эмбеддинги")
        self.reranker = self._load(reranker, lambda: load_reranker(self.settings.rerank_model), "реранкер")
        self.provider = get_provider(self.settings) if provider == "auto" else provider
        self.retriever = Retriever(self.store, self.embedder, self.reranker, self.settings.rerank_candidates)
        self.started = time.time()
        self.recent = deque(maxlen=50)
        self._tree = None

    def _load(self, given, fn, what):
        if given != "auto":
            return given
        try:
            return fn()
        except Exception as e:  # модель недоступна — работаем без неё, но честно сообщаем
            log.warning("%s отключены: %s", what, e)
            self.warnings.append(f"{what}: {e}")
            return None

    def tree(self) -> list[dict]:
        if self._tree is None:
            self._tree = build_tree(self.store)
        return self._tree


def build_tree(store: Store) -> list[dict]:
    nodes: dict[str, dict] = {}
    roots: list[dict] = []
    rows = store.conn.execute("SELECT url,title,breadcrumbs,nchunks FROM articles ORDER BY rowid").fetchall()
    titles = {r["url"]: r["title"] for r in rows}

    def node(url, title):
        n = nodes.get(url)
        if n is None:
            n = nodes[url] = {"url": url, "title": titles.get(url, title), "children": [], "indexed": url in titles}
        return n

    for r in rows:
        chain = json.loads(r["breadcrumbs"]) + [{"title": r["title"], "url": r["url"]}]
        parent = None
        for c in chain:
            is_new = c["url"] not in nodes
            n = node(c["url"], c["title"])
            if is_new:
                (parent["children"] if parent else roots).append(n)
            parent = n
    # корневая страница KB («Введение») не нужна как папка — оставляем её как есть, но выносим разделы на верхний уровень
    for n in roots:
        if n["url"].rstrip("/").endswith("/kb") and n["children"] and False:
            pass

    def prune(ns):
        for n in ns:
            prune(n["children"])
            n["count"] = (1 if n["indexed"] and not n["children"] else 0) + sum(c["count"] for c in n["children"])
    prune(roots)
    return roots


class AskBody(BaseModel):
    q: str = Field(..., min_length=1, max_length=600)
    history: list[dict] = []
    k: int = 5
    section: str | None = Field(None, max_length=60, pattern=r"^[a-z0-9-]+$")


def create_app(state: AppState | None = None) -> FastAPI:
    holder: dict = {"s": state}

    @asynccontextmanager
    async def lifespan(_app):
        S()  # модели загружаются при старте, а не при первом запросе
        yield

    app = FastAPI(title="Webim AI Knowledge", version=__version__, docs_url="/api/docs", redoc_url=None, lifespan=lifespan)

    def S() -> AppState:
        if holder["s"] is None:
            holder["s"] = AppState()
        return holder["s"]

    # ---------- поиск ----------
    def evidence_for(res, k: int):
        """Лучший фрагмент берём всегда; остальные — только если они заметно релевантны (не тянем случайные разделы в ответ)."""
        hits = res.hits[:k]
        if res.modes.get("rerank") and hits:
            top = hits[0].score
            hits = hits[:1] + [h for h in hits[1:] if h.score >= max(RERANK_WEAK, 0.25 * top)]
        return hits[: min(k, 5)]

    def relevant_articles(arts):
        """Список статей без «шума»: слабые совпадения скрываем, но минимум 3 оставляем (пользователю всегда есть куда перейти)."""
        if not arts:
            return arts
        top = arts[0]["score"]
        keep = [a for a in arts if a["score"] >= max(0.02, 0.15 * top)]
        return keep if len(keep) >= 3 else arts[:3]

    def retrieval_payload(res, evidence):
        return {
            "query": res.query, "confident": res.confident, "unknown_terms": res.unknown_terms, "confidence": round(res.confidence, 3), "modes": res.modes,
            "timings": res.timings,
            "evidence": [dict(h.to_dict(), n=i + 1) for i, h in enumerate(evidence)],
            "articles": relevant_articles(res.articles),
        }

    def retrieval_query(q: str, history: list[dict]) -> str:
        """Уточняющий вопрос без контекста добавляем к предыдущему вопросу пользователя (только для поиска)."""
        prev = [t["content"] for t in history if t.get("role") == "user" and isinstance(t.get("content"), str)]
        if prev and len(q.split()) <= 5:
            return prev[-1][:300] + " " + q
        return q

    @app.get("/api/search")
    def api_search(q: str = Query(..., min_length=1, max_length=600), k: int = 8, section: str | None = Query(None, pattern=r"^[a-z0-9-]+$")):
        s = S()
        res = s.retriever.search(q, k=max(1, min(k, 20)), section=section)
        s.recent.append({"q": q, **res.timings, "t": time.time()})
        return retrieval_payload(res, res.hits)

    @app.post("/api/ask")
    def api_ask(body: AskBody):
        s = S()

        def gen():
            def ev(name, data):
                return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            try:
                rq = retrieval_query(body.q, body.history)
                res = s.retriever.search(rq, k=max(3, min(body.k, 8)), section=body.section)
                evidence = evidence_for(res, body.k)
                s.recent.append({"q": body.q, **res.timings, "t": time.time()})
                yield ev("retrieval", retrieval_payload(res, evidence))
                t0 = time.perf_counter()
                for e in answer_stream(body.q, evidence, res.confident, s.provider, body.history):
                    if e["type"] == "token":
                        yield ev("token", {"text": e["text"]})
                    else:
                        e = dict(e, generation_ms=round((time.perf_counter() - t0) * 1000))
                        e.pop("type")
                        yield ev("final", e)
            except Exception as exc:  # не показываем трассировки пользователю
                log.exception("ask failed")
                yield ev("error", {"message": "Не удалось выполнить поиск. Попробуйте ещё раз.", "detail": type(exc).__name__})
            yield ev("done", {})

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------- справочники ----------
    @app.get("/api/tree")
    def api_tree():
        return S().tree()

    @app.get("/api/article")
    def api_article(url: str):
        cu = canonicalize(url)
        if not cu:
            raise HTTPException(400, "URL вне Базы знаний")
        r = S().store.conn.execute("SELECT * FROM articles WHERE url=?", (cu,)).fetchone()
        if not r:
            raise HTTPException(404, "Статья не найдена в индексе")
        return {"url": r["url"], "title": r["title"], "updated": r["updated"], "breadcrumbs": json.loads(r["breadcrumbs"]),
                "sections": json.loads(r["doc"])}

    @app.get("/api/sections")
    def api_sections():
        """Разделы верхнего уровня для фильтра «искать в…» (ключ — первый сегмент пути в /kb/)."""
        out = []
        for n in S().tree():
            for c in (n["children"] if n["url"].rstrip("/").endswith("/kb") else [n]):
                seg = c["url"].replace("https://webim.ru/kb/", "").strip("/").split("/")[0]
                if seg and not seg.endswith(".html") and c["children"]:
                    out.append({"key": seg, "title": c["title"], "count": c["count"]})
        return out

    @app.get("/api/suggestions")
    def api_suggestions():
        p = ROOT / "eval" / "demo_questions.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"featured": [], "demo": [], "topics": []}

    @app.get("/api/status")
    def api_status():
        s = S()
        c = s.store.counts()
        last = s.store.get_meta("last_sync")
        lat = [r["total_ms"] for r in s.recent if "total_ms" in r]
        return {
            "version": __version__, "counts": c, "last_sync": last, "sync_history": s.store.last_sync_runs(5),
            "embedding_model": s.embedder.name if s.embedder else None, "reranker": s.reranker.name if s.reranker else None,
            "llm": {"configured": s.provider is not None, "label": s.provider.label if s.provider else None,
                    "mode": "llm" if s.provider else "extractive"},
            "warnings": s.warnings, "semantic_ready": bool(s.retriever.matrix is not None),
            "healthy": c["chunks"] > 0,
            "recent_latency_ms": {"n": len(lat), "avg": round(sum(lat) / len(lat)) if lat else None, "max": round(max(lat)) if lat else None},
            "recent": list(s.recent)[-8:], "rerank_threshold": RERANK_MIN, "uptime_s": round(time.time() - s.started),
        }

    @app.get("/api/health")
    def health():
        return {"ok": True}

    # ---------- статика ----------
    if WEB.exists():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str, request: Request):
            if path.startswith("api/"):
                raise HTTPException(404)
            return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
