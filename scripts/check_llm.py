"""Проверка подключения LLM: python scripts/check_llm.py  (настройки берутся из .env / переменных окружения).
Отправляет реальный запрос через тот же провайдер, что использует приложение, и показывает ответ по реальному вопросу."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webim_ai.answer import LLMError, get_provider, answer_stream  # noqa: E402
from webim_ai.config import get_settings  # noqa: E402
from webim_ai.models import load_embedder, load_reranker  # noqa: E402
from webim_ai.retrieval import Retriever  # noqa: E402
from webim_ai.store import Store  # noqa: E402


def main() -> int:
    st = get_settings()
    prov = get_provider(st)
    if prov is None:
        print("LLM не настроен. Задайте LLM_BASE_URL и LLM_MODEL (см. docs/LOCAL_LLM.md).")
        return 2
    print("Провайдер:", prov.label)
    q = sys.argv[1] if len(sys.argv) > 1 else "Как поставить диалог на паузу, пока я ищу ответ для клиента?"
    r = Retriever(Store(st.db_path), load_embedder(st.embed_model), load_reranker(st.rerank_model))
    res = r.search(q, k=5)
    ev = res.hits[:3]
    t0, first = time.time(), None
    try:
        for e in answer_stream(q, ev, res.confident, prov):
            if e["type"] == "token":
                first = first or time.time() - t0
                print(e["text"], end="", flush=True)
            else:
                print(f"\n\n[режим: {e['mode']}, цитируются источники: {e['cited']}, первый токен через {first and round(first, 2)} с, всего {round(time.time() - t0, 1)} с]")
                if e.get("llm_error"):
                    print("Ошибка LLM:", e["llm_error"])
                    return 1
    except LLMError as ex:
        print("Ошибка:", ex)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
