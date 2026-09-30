"""SQLite-хранилище: статьи, чанки, FTS5 (лексический индекс), кэш эмбеддингов, история синхронизаций."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .text import fts_index_text, fts_query

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles(
  url TEXT PRIMARY KEY, title TEXT, breadcrumbs TEXT, updated TEXT, lastmod TEXT,
  content_hash TEXT, doc TEXT, nchunks INTEGER DEFAULT 0, fetched_at REAL, description TEXT, links TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL, ord INTEGER, part INTEGER DEFAULT 0, anchor TEXT,
  heading_path TEXT, title TEXT, text TEXT, embed_hash TEXT, head_hash TEXT);
CREATE INDEX IF NOT EXISTS chunks_url ON chunks(url);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(title, headings, body, tokenize='unicode61 remove_diacritics 0');
CREATE TABLE IF NOT EXISTS embeddings(embed_hash TEXT, model TEXT, dim INTEGER, vec BLOB, PRIMARY KEY(embed_hash, model));
CREATE TABLE IF NOT EXISTS sync_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, started REAL, finished REAL, stats TEXT);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        try:  # миграция старых индексов
            self.conn.execute('ALTER TABLE chunks ADD COLUMN head_hash TEXT')
        except sqlite3.OperationalError:
            pass

    def close(self) -> None:
        self.conn.close()

    # ---------- meta ----------
    def set_meta(self, k: str, v) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))
            self.conn.commit()

    def get_meta(self, k: str, default=None):
        r = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    # ---------- статьи ----------
    def get_article_meta(self, url: str):
        return self.conn.execute("SELECT url, content_hash, lastmod, fetched_at, links FROM articles WHERE url=?", (url,)).fetchone()

    def load_article(self, url: str):
        """Восстанавливает Article из сохранённой структуры (для пересборки чанков без сети)."""
        from .parser import Article, Section
        r = self.conn.execute("SELECT * FROM articles WHERE url=?", (url,)).fetchone()
        if not r:
            return None
        secs = [Section(s["level"], s["heading"], s["anchor"], s["path"], s["blocks"]) for s in json.loads(r["doc"])]
        return Article(r["url"], r["title"], json.loads(r["breadcrumbs"]), r["updated"], secs, json.loads(r["links"] or "[]"),
                       r["content_hash"], r["description"] or "")

    def all_article_urls(self) -> set[str]:
        return {r["url"] for r in self.conn.execute("SELECT url FROM articles")}

    def touch_article(self, url: str, lastmod: str | None) -> None:
        with self.lock:
            self.conn.execute("UPDATE articles SET lastmod=?, fetched_at=? WHERE url=?", (lastmod, time.time(), url))
            self.conn.commit()

    def upsert_article(self, a, lastmod: str | None, chunks: list[Chunk], embed_hashes: list[str], head_hashes: list[str] | None = None) -> None:
        """Атомарно заменяет статью и все её чанки (провенанс сохраняется в каждом чанке)."""
        with self.lock, self.conn:
            self._delete_chunks(a.url)
            self.conn.execute(
                """INSERT INTO articles(url,title,breadcrumbs,updated,lastmod,content_hash,doc,nchunks,fetched_at,description,links)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(url) DO UPDATE SET title=excluded.title, breadcrumbs=excluded.breadcrumbs, updated=excluded.updated,
                     lastmod=excluded.lastmod, content_hash=excluded.content_hash, doc=excluded.doc, nchunks=excluded.nchunks,
                     fetched_at=excluded.fetched_at, description=excluded.description, links=excluded.links""",
                (a.url, a.title, json.dumps(a.breadcrumbs, ensure_ascii=False), a.updated, lastmod, a.content_hash,
                 json.dumps([{"level": s.level, "heading": s.heading, "anchor": s.anchor, "path": s.path, "blocks": s.blocks}
                             for s in a.sections], ensure_ascii=False),
                 len(chunks), time.time(), a.description, json.dumps(a.links)))
            head_hashes = head_hashes or [None] * len(chunks)
            for c, eh, hh in zip(chunks, embed_hashes, head_hashes):
                cur = self.conn.execute(
                    "INSERT INTO chunks(url,ord,part,anchor,heading_path,title,text,embed_hash,head_hash) VALUES(?,?,?,?,?,?,?,?,?)",
                    (c.url, c.ord, c.part, c.anchor, json.dumps(c.heading_path, ensure_ascii=False), c.title, c.text, eh, hh))
                heads = " ".join(c.heading_path[1:]) if len(c.heading_path) > 1 else c.heading_path[0]
                self.conn.execute("INSERT INTO chunks_fts(rowid,title,headings,body) VALUES(?,?,?,?)",
                                  (cur.lastrowid, fts_index_text(c.title), fts_index_text(heads), fts_index_text(c.text)))

    def _delete_chunks(self, url: str) -> None:
        ids = [r["id"] for r in self.conn.execute("SELECT id FROM chunks WHERE url=?", (url,))]
        for i in ids:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (i,))
        self.conn.execute("DELETE FROM chunks WHERE url=?", (url,))

    def delete_article(self, url: str) -> None:
        with self.lock, self.conn:
            self._delete_chunks(url)
            self.conn.execute("DELETE FROM articles WHERE url=?", (url,))

    def gc_embeddings(self) -> int:
        with self.lock, self.conn:
            cur = self.conn.execute("DELETE FROM embeddings WHERE embed_hash NOT IN (SELECT embed_hash FROM chunks UNION SELECT head_hash FROM chunks WHERE head_hash IS NOT NULL)")
            return cur.rowcount

    # ---------- эмбеддинги ----------
    def get_embeddings(self, hashes: list[str], model: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(hashes), 500):
            part = hashes[i:i + 500]
            q = ",".join("?" * len(part))
            for r in self.conn.execute(f"SELECT embed_hash, vec FROM embeddings WHERE model=? AND embed_hash IN ({q})", [model, *part]):
                out[r["embed_hash"]] = np.frombuffer(r["vec"], dtype=np.float16)
        return out

    def put_embeddings(self, items: dict[str, np.ndarray], model: str) -> None:
        with self.lock, self.conn:
            for h, v in items.items():
                self.conn.execute("INSERT OR REPLACE INTO embeddings(embed_hash,model,dim,vec) VALUES(?,?,?,?)",
                                  (h, model, len(v), np.asarray(v, dtype=np.float16).tobytes()))

    # ---------- поиск ----------
    def lexical(self, query: str, limit: int = 40) -> list[tuple[int, float]]:
        q = fts_query(query)
        if not q:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts, 6.0, 4.0, 1.0) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
                (q, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["rowid"], -r["s"]) for r in rows]

    def chunk_rows(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        return {r["id"]: r for r in self.conn.execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids)}

    def load_matrix(self, model: str, col: str = "embed_hash") -> tuple[list[int], np.ndarray | None]:
        assert col in ("embed_hash", "head_hash")
        rows = self.conn.execute(
            f"SELECT c.id, e.vec FROM chunks c JOIN embeddings e ON e.embed_hash=c.{col} AND e.model=? ORDER BY c.id", (model,)).fetchall()
        if not rows:
            return [], None
        ids = [r["id"] for r in rows]
        m = np.stack([np.frombuffer(r["vec"], dtype=np.float16) for r in rows]).astype(np.float32)
        return ids, m

    # ---------- статистика ----------
    def counts(self) -> dict:
        c = self.conn
        return {
            "articles": c.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
            "chunks": c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "embedded_chunks": c.execute("SELECT COUNT(DISTINCT embed_hash) FROM chunks WHERE embed_hash IN (SELECT embed_hash FROM embeddings)").fetchone()[0],
        }

    def add_sync_run(self, started: float, finished: float, stats: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO sync_runs(started,finished,stats) VALUES(?,?,?)", (started, finished, json.dumps(stats, ensure_ascii=False)))

    def last_sync_runs(self, n: int = 5) -> list[dict]:
        return [{"id": r["id"], "started": r["started"], "finished": r["finished"], **json.loads(r["stats"])}
                for r in self.conn.execute("SELECT * FROM sync_runs ORDER BY id DESC LIMIT ?", (n,))]
