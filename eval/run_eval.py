"""Оценка качества поиска на eval/questions.json.

    python eval/run_eval.py                 # гибрид + реранкер (как в продукте)
    python eval/run_eval.py --compare       # сравнить: только лексика / гибрид / гибрид + реранкер
    python eval/run_eval.py --verbose       # показать неудачи подробно
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webim_ai.config import get_settings  # noqa: E402
from webim_ai.models import load_embedder, load_reranker  # noqa: E402
from webim_ai.retrieval import Retriever  # noqa: E402
from webim_ai.store import Store  # noqa: E402

BASE = "https://webim.ru/kb/"


def evaluate(retr: Retriever, questions: list[dict], rerank: bool, verbose: bool = False) -> dict:
    ranks, rows, lat = [], [], []
    none_ok = none_total = 0
    for q in questions:
        t = time.perf_counter()
        res = retr.search(q["q"], k=8, rerank=rerank)
        lat.append((time.perf_counter() - t) * 1000)
        urls = [BASE + e for e in q.get("expect", [])]
        top_urls = []
        for a in res.articles:
            top_urls.append(a["url"])
        if q.get("expect_none"):
            none_total += 1
            ok = not res.confident
            none_ok += ok
            rows.append((q["id"], q["cat"], "OK " if ok else "BAD", f"confident={res.confident} conf={res.confidence:.2f} top={res.articles[0]['title'] if res.articles else '-'}"))
            continue
        rank = next((i + 1 for i, u in enumerate(top_urls) if u in urls), None)
        ranks.append(rank)
        rows.append((q["id"], q["cat"], f"@{rank}" if rank else "MISS", f"{q['q'][:60]} → {res.articles[0]['title'] if res.articles else '-'} (conf {res.confidence:.2f}, {'confident' if res.confident else 'NOT confident'})"))
    n = len(ranks)
    return {
        "n": n,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "none_ok": none_ok, "none_total": none_total,
        "abstain_on_answerable": sum(1 for q, row in zip([x for x in questions if not x.get("expect_none")], [r for r in rows if not r[3].startswith("confident=")]) if "NOT confident" in row[3]),
        "avg_ms": sum(lat) / len(lat), "rows": rows,
    }


def show(name: str, m: dict, verbose: bool) -> None:
    print(f"\n=== {name} ===")
    print(f"вопросов с ожидаемой статьёй: {m['n']}   hit@1 {m['hit@1']:.0%}   hit@3 {m['hit@3']:.0%}   hit@5 {m['hit@5']:.0%}   MRR {m['mrr']:.3f}")
    print(f"вопросов без ответа в KB: корректный отказ {m['none_ok']}/{m['none_total']};  отказ на отвечаемые вопросы: {m['abstain_on_answerable']}/{m['n']};  среднее время {m['avg_ms']:.0f} мс")
    for r in m["rows"]:
        if verbose or r[2] in ("MISS", "BAD") or (r[2].startswith("@") and r[2] != "@1"):
            print(f"  {r[0]:<4} {r[1]:<10} {r[2]:<5} {r[3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", help="сохранить результаты в файл")
    a = ap.parse_args()
    st = get_settings()
    qs = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))["questions"]
    store = Store(st.db_path)
    emb, rr = load_embedder(st.embed_model), load_reranker(st.rerank_model)
    out = {}
    if a.compare:
        for name, e, r, do_rr in (("только лексика (BM25)", None, None, False), ("гибрид (лексика + семантика, RRF)", emb, None, False),
                                  ("гибрид + реранкер", emb, rr, True)):
            m = evaluate(Retriever(store, e, r), qs, do_rr, a.verbose)
            show(name, m, a.verbose)
            out[name] = {k: v for k, v in m.items() if k != "rows"}
    else:
        m = evaluate(Retriever(store, emb, rr), qs, True, a.verbose)
        show("гибрид + реранкер", m, a.verbose)
        out["hybrid+rerank"] = {k: v for k, v in m.items() if k != "rows"}
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
