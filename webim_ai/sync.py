"""Инкрементальная синхронизация: обнаружение -> условная загрузка -> разбор -> сравнение хешей -> обновление индекса и эмбеддингов.

Запуск:  python -m webim_ai.sync [--force] [--no-embed] [--full-check] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .chunker import CHUNKER_VERSION, chunk_article
from .config import ROOT, get_settings
from .crawler import Crawler, canonicalize
from .parser import parse_article
from .store import Store


def embed_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def head_hash(text: str) -> str:
    return hashlib.sha1(("H|" + text).encode("utf-8")).hexdigest()


def hashes(chunks) -> tuple[list[str], list[str]]:
    return [embed_hash(c.embed_text) for c in chunks], [head_hash(c.head_text) for c in chunks]


def run_sync(settings=None, store: Store | None = None, crawler: Crawler | None = None, embedder=None, *, force: bool = False,
             full_check: bool = False, follow_links: bool = True, limit: int | None = None, embed: bool = True,
             log=print, extra_sitemaps: list[Path] | None = None) -> dict:
    settings = settings or get_settings()
    store = store or Store(settings.db_path)
    crawler = crawler or Crawler(settings.cache_dir, delay=settings.request_delay)
    t0 = time.time()
    stats = dict(discovered=0, unchanged=0, updated=0, new=0, removed=0, failed=0, duplicates=0, chunks_updated=0,
                 embedded=0, fetched=0, skipped_by_lastmod=0, errors=[], link_discovered=0)

    if store.counts()["articles"] and store.get_meta("chunker_version") != CHUNKER_VERSION:
        n = rechunk_all(store, log)
        stats["rechunked_articles"] = n
    store.set_meta("chunker_version", CHUNKER_VERSION)

    sitemap = crawler.discover(extra_sitemaps if extra_sitemaps is not None else [ROOT / "kb_sitemap.xml", ROOT / "kb_sitemap_1.xml"])
    urls = list(sitemap.keys())
    if limit:
        urls = urls[:limit]
    queue = list(urls)
    seen: set[str] = set()
    hashes_seen: dict[str, str] = {}
    log(f"Обнаружено URL в sitemap: {len(sitemap)}")

    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        prev = store.get_article_meta(url)
        lastmod = sitemap.get(url)
        cached_ok = crawler.cached(url) is not None
        if prev and not force and not full_check and lastmod and prev["lastmod"] == lastmod and cached_ok:
            stats["skipped_by_lastmod"] += 1
            stats["unchanged"] += 1
            hashes_seen[prev["content_hash"]] = url
            if follow_links and not limit:  # ссылки неизменённой статьи берём из индекса
                for l in filter(None, (canonicalize(x) for x in json.loads(prev["links"] or "[]"))):
                    if l not in seen and l not in queue and l not in sitemap:
                        queue.append(l)
                        stats["link_discovered"] += 1
            continue
        res = crawler.fetch(url, force=force or bool(prev and (full_check or lastmod != prev["lastmod"])) )
        if not res.from_cache:
            stats["fetched"] += 1
        if res.status != 200 or not res.html:
            if res.status == 404 and prev is None and url not in sitemap:
                continue  # ссылка из статьи ведёт в 404 — не считаем сбоем индекса
            stats["failed"] += 1
            stats["errors"].append(f"{url}: {res.error}")
            if prev:
                seen.add(url)  # не удаляем статью при временном сбое
            continue
        art = parse_article(res.html, res.final_url or url)
        if art is None:
            stats["failed"] += 1
            stats["errors"].append(f"{url}: не найден контент статьи")
            continue
        if art.url != url:  # канонический URL другой -> это алиас; индексируем только канонический
            if art.url in seen or art.url in sitemap:
                continue
            url = art.url
            seen.add(url)
            prev = store.get_article_meta(url)
        if art.content_hash in hashes_seen and hashes_seen[art.content_hash] != url:
            stats["duplicates"] += 1
            continue
        hashes_seen[art.content_hash] = url
        if prev and prev["content_hash"] == art.content_hash:
            stats["unchanged"] += 1
            store.touch_article(url, lastmod)
        else:
            chunks = chunk_article(art)
            store.upsert_article(art, lastmod, chunks, *hashes(chunks))
            stats["chunks_updated"] += len(chunks)
            stats["new" if prev is None else "updated"] += 1
            log(f"  {'+' if prev is None else '~'} {art.title}  ({len(chunks)} чанков)")
        if follow_links and not limit:
            for l in art.links:
                if l not in seen and l not in queue and l not in sitemap:
                    queue.append(l)
                    stats["link_discovered"] += 1

    stats["discovered"] = stats["unchanged"] + stats["updated"] + stats["new"]

    # удалённые страницы
    stored = store.all_article_urls()
    gone = stored - seen if not limit else set()
    if gone and len(seen) >= 0.5 * max(len(stored), 1):
        for u in sorted(gone):
            store.delete_article(u)
            log(f"  - удалена: {u}")
        stats["removed"] = len(gone)
    elif gone:
        stats["errors"].append(f"пропущено удаление {len(gone)} статей: обнаружено подозрительно мало страниц")

    # эмбеддинги — только для новых текстов
    if embed and embedder is not None:
        stats["embedded"] = embed_missing(store, embedder, log)
    store.gc_embeddings()
    c = store.counts()
    stats.update(indexed_articles=c["articles"], indexed_chunks=c["chunks"], embedded_chunks=c["embedded_chunks"])
    stats["duration_s"] = round(time.time() - t0, 1)
    stats["requests"] = crawler.requests_made
    store.add_sync_run(t0, time.time(), stats)
    store.set_meta("last_sync", {"finished": time.time(), **{k: v for k, v in stats.items() if k != "errors"}})
    return stats


def rechunk_all(store: Store, log=print) -> int:
    """Пересобирает чанки всех статей из сохранённой структуры (правила разбиения изменились)."""
    urls = sorted(store.all_article_urls())
    for u in urls:
        a = store.load_article(u)
        meta = store.get_article_meta(u)
        chunks = chunk_article(a)
        store.upsert_article(a, meta["lastmod"], chunks, *hashes(chunks))
    log(f"Чанки пересобраны для {len(urls)} статей (версия разбиения {CHUNKER_VERSION})")
    return len(urls)


def embed_missing(store: Store, embedder, log=print, batch: int = 16) -> int:
    return _embed_chunks(store, embedder, log, batch) + _embed_headings(store, embedder, log)


def _embed_headings(store: Store, embedder, log=print) -> int:
    """Короткие эмбеддинги «Статья › Раздел» — отдельный сигнал для вопросов, сформулированных как название темы."""
    rows = store.conn.execute("SELECT DISTINCT head_hash, title, heading_path FROM chunks WHERE head_hash IS NOT NULL").fetchall()
    key = embedder.name + "#head"
    have = store.get_embeddings([r["head_hash"] for r in rows], key)
    todo = [r for r in rows if r["head_hash"] not in have]
    if not todo:
        return 0
    log(f"Эмбеддинги заголовков: {len(todo)}")
    for i in range(0, len(todo), 128):
        part = todo[i:i + 128]
        texts = []
        for r in part:
            hp = json.loads(r["heading_path"])
            texts.append(r["title"] + (" › " + " › ".join(hp[1:]) if len(hp) > 1 else ""))
        store.put_embeddings({r["head_hash"]: v for r, v in zip(part, embedder.embed_passages(texts, 32))}, key)
    return len(todo)


def _embed_chunks(store: Store, embedder, log=print, batch: int = 16) -> int:
    rows = store.conn.execute("SELECT DISTINCT embed_hash, title, heading_path, text FROM chunks").fetchall()
    have = store.get_embeddings([r["embed_hash"] for r in rows], embedder.name)
    todo = [r for r in rows if r["embed_hash"] not in have]
    if not todo:
        return 0
    log(f"Эмбеддинги ({embedder.name}): нужно посчитать {len(todo)} из {len(rows)}")
    t0 = time.time()
    done = 0
    for i in range(0, len(todo), 64):
        part = todo[i:i + 64]
        texts = []
        for r in part:
            hp = json.loads(r["heading_path"])
            head = r["title"] + (" › " + " › ".join(hp[1:]) if len(hp) > 1 else "")
            texts.append(f"{head}\n{r['text']}")
        vecs = embedder.embed_passages(texts, batch)
        store.put_embeddings({r["embed_hash"]: v for r, v in zip(part, vecs)}, embedder.name)
        done += len(part)
        el = time.time() - t0
        log(f"  эмбеддинги {done}/{len(todo)}  ({el:.0f}с, ~{el / done * (len(todo) - done):.0f}с осталось)")
    return done


def format_summary(s: dict) -> str:
    return "\n".join([
        f"Articles discovered: {s['discovered']}",
        f"Unchanged: {s['unchanged']}",
        f"Updated: {s['updated']}",
        f"New: {s['new']}",
        f"Removed: {s['removed']}",
        f"Failed: {s['failed']}",
        f"Chunks updated: {s['chunks_updated']}",
        f"Embeddings computed: {s['embedded']}",
        f"HTTP requests: {s.get('requests', 0)}",
        f"Duration: {s['duration_s']}s",
        f"Index now: {s['indexed_articles']} articles, {s['indexed_chunks']} chunks",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Синхронизация индекса Webim AI Knowledge с публичной Базой знаний")
    ap.add_argument("--force", action="store_true", help="перекачать все страницы")
    ap.add_argument("--full-check", action="store_true", help="проверить все страницы условным GET (не полагаться на lastmod)")
    ap.add_argument("--no-embed", action="store_true", help="не считать эмбеддинги (только лексический индекс)")
    ap.add_argument("--no-links", action="store_true", help="не обходить ссылки внутри статей")
    ap.add_argument("--limit", type=int, help="ограничить число страниц (для отладки)")
    a = ap.parse_args(argv)
    st = get_settings()
    from .models import load_embedder
    emb = None
    if not a.no_embed:
        try:
            emb = load_embedder(st.embed_model)
        except Exception as e:  # модель недоступна — индекс всё равно обновляется лексически
            print(f"[!] эмбеддинги отключены: {e}", file=sys.stderr)
    stats = run_sync(st, embedder=emb, force=a.force, full_check=a.full_check, follow_links=not a.no_links, limit=a.limit,
                     embed=not a.no_embed)
    print("\n" + format_summary(stats))
    for e in stats["errors"][:20]:
        print("  !", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
