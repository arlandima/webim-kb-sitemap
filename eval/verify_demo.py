"""Проверяет каждый демо-сценарий на живом индексе: правильный раздел в топ-3 источников (URL + якорь) либо корректный отказ.

    python eval/verify_demo.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from webim_ai.config import get_settings  # noqa: E402
from webim_ai.models import load_embedder, load_reranker  # noqa: E402
from webim_ai.retrieval import Retriever  # noqa: E402
from webim_ai.store import Store  # noqa: E402


def main() -> int:
    st = get_settings()
    R = Retriever(Store(st.db_path), load_embedder(st.embed_model), load_reranker(st.rerank_model))
    demo = json.loads((ROOT / "eval" / "demo_questions.json").read_text(encoding="utf-8"))
    bad = 0
    for d in demo["demo"] + [{"q": f["q"], "expect": None} for f in demo["featured"]]:
        res = R.search(d["q"], k=5)
        if d.get("expect_none"):
            ok = not res.confident
            print(("OK  " if ok else "FAIL"), d["q"], "→ отказ" if ok else "→ уверенный ответ!")
        elif d.get("expect"):
            e = d["expect"]
            want = e["url"] + (f"#{e['anchor']}" if e["anchor"] else "")
            got = [h.source_url for h in res.hits[:3]]
            ok = any(g == want or (not e["anchor"] and g.split("#")[0] == e["url"]) for g in got) and res.confident
            print(("OK  " if ok else "FAIL"), d["q"], "→", got[0].replace("https://webim.ru/kb/", ""))
        else:  # featured: достаточно уверенного ответа
            ok = res.confident
            print(("OK  " if ok else "FAIL"), "(featured)", d["q"], "→", res.hits[0].source_url.replace("https://webim.ru/kb/", "") if res.hits else "-")
        bad += not ok
    print("\nВсе сценарии подтверждены" if not bad else f"\nПровалов: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
