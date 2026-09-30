#!/usr/bin/env python3
"""Единая точка запуска Webim AI Knowledge (Windows / macOS / Linux).

    python run.py            # создаст .venv, поставит зависимости, скачает модели (один раз), запустит сервер
    python run.py --sync     # предварительно обновить индекс с webim.ru/kb
    python run.py --no-open  # не открывать браузер
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def sh(*cmd: str) -> int:
    return subprocess.call(list(cmd), cwd=ROOT)


def bootstrap() -> None:
    """Создаёт виртуальное окружение и перезапускает этот же скрипт внутри него."""
    if Path(sys.prefix).resolve() == VENV.resolve():
        return
    if os.environ.get("WEBIM_AI_NO_VENV") == "1":
        return
    if sys.version_info < (3, 10):
        sys.exit("Нужен Python 3.10 или новее (https://www.python.org/downloads/).")
    if not PY.exists():
        print("→ Создаю виртуальное окружение .venv …")
        if sh(sys.executable, "-m", "venv", str(VENV)) != 0:
            sys.exit("Не удалось создать .venv")
    stamp = VENV / ".req-stamp"
    want = hashlib.sha1((ROOT / "requirements.txt").read_bytes()).hexdigest()
    if not stamp.exists() or stamp.read_text() != want:
        print("→ Устанавливаю зависимости (один раз, 1–3 минуты) …")
        if sh(str(PY), "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", "requirements.txt") != 0:
            sys.exit("Не удалось установить зависимости (проверьте доступ в интернет к pypi.org)")
        stamp.write_text(want)
    sys.exit(subprocess.call([str(PY), str(ROOT / "run.py"), *sys.argv[1:]], cwd=ROOT))


def main() -> None:
    bootstrap()
    sys.path.insert(0, str(ROOT))
    from webim_ai.config import get_settings
    from webim_ai.store import Store

    st = get_settings()
    args = set(sys.argv[1:])
    if "--sync" in args or not st.db_path.exists() or Store(st.db_path).counts()["chunks"] == 0:
        print("→ Синхронизация индекса с https://webim.ru/kb/ …")
        if sh(sys.executable, "-m", "webim_ai.sync") != 0:
            sys.exit("Синхронизация не удалась")
    else:
        from webim_ai.models import EMBEDDERS, RERANKERS, ensure_files
        for kind, table, name in (("эмбеддинги", EMBEDDERS, st.embed_model), ("реранкер", RERANKERS, st.rerank_model)):
            if name in table:
                try:
                    ensure_files(table[name], st.models_dir, download=False)
                except FileNotFoundError:
                    print(f"→ Скачиваю модель ({kind}: {name}) — один раз, может занять несколько минут …")
                    try:
                        ensure_files(table[name], st.models_dir)
                    except Exception as e:
                        print(f"[!] не удалось скачать {name}: {e}\n    Поиск продолжит работать в упрощённом режиме.")
    url = f"http://{'localhost' if st.host in ('127.0.0.1', '0.0.0.0') else st.host}:{st.port}"
    print(f"\n✔ Webim AI Knowledge: {url}\n   Демо: {url}/demo   Статус индекса: {url}/admin\n   Остановить: Ctrl+C\n")
    if "--no-open" not in args:
        try:
            import threading
            threading.Timer(2.5, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    import uvicorn
    uvicorn.run("webim_ai.server:app", host=st.host, port=st.port, log_level="warning")


if __name__ == "__main__":
    main()
