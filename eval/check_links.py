"""Проверка ссылок на источники: (1) якорь каждого чанка существует в сохранённом HTML страницы;
(2) URL демо-сценариев отвечают 200 на живом сайте (--live).

    python eval/check_links.py [--live]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from webim_ai.config import get_settings  # noqa: E402
from webim_ai.crawler import Crawler  # noqa: E402
from webim_ai.store import Store  # noqa: E402


def main() -> int:
    st = get_settings()
    store = Store(st.db_path)
    cr = Crawler(st.cache_dir, delay=0.3)
    ids_by_url: dict[str, set[str]] = {}
    bad = total = 0
    for r in store.conn.execute("SELECT url, anchor FROM chunks"):
        total += 1
        if not r["anchor"]:
            continue
        ids = ids_by_url.get(r["url"])
        if ids is None:
            c = cr.cached(r["url"])
            ids = ids_by_url[r["url"]] = set(re.findall(r'\bid="([^"]+)"', c.html)) if c else set()
        if r["anchor"] not in ids:
            bad += 1
            print("нет якоря:", r["url"], "#" + r["anchor"])
    print(f"чанков: {total}, якорей не найдено: {bad}")
    if "--live" in sys.argv:
        demo = json.loads((ROOT / "eval" / "demo_questions.json").read_text(encoding="utf-8"))
        urls = sorted({d["expect"]["url"] for d in demo["demo"] if d.get("expect")})
        for u in urls:
            r = cr._get(u)
            ok = r.status_code == 200
            has = True
            a = next((d["expect"]["anchor"] for d in demo["demo"] if d.get("expect") and d["expect"]["url"] == u and d["expect"]["anchor"]), "")
            if a:
                has = f'id="{a}"' in r.text
            print(("OK  " if ok and has else "FAIL"), r.status_code, u, ("#" + a) if a else "")
            bad += not (ok and has)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
