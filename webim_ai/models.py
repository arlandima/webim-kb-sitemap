"""Модульные эмбеддинги и реранкер на ONNX Runtime (без PyTorch): легко заменяются через EMBED_MODEL / RERANK_MODEL."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    files: tuple[str, ...]  # что скачать (относительно репозитория)
    onnx: str  # какой файл — сама модель
    tokenizer: str = "tokenizer.json"
    query_prefix: str = ""
    passage_prefix: str = ""
    max_len: int = 512
    dim: int = 0


EMBEDDERS = {
    "multilingual-e5-large": ModelSpec("multilingual-e5-large", "intfloat/multilingual-e5-large",
                                       ("onnx/model.onnx", "onnx/model.onnx_data", "onnx/tokenizer.json"), "onnx/model.onnx",
                                       "onnx/tokenizer.json", "query: ", "passage: ", 512, 1024),
    "multilingual-e5-small": ModelSpec("multilingual-e5-small", "intfloat/multilingual-e5-small",
                                       ("onnx/model.onnx", "onnx/tokenizer.json"), "onnx/model.onnx",
                                       "onnx/tokenizer.json", "query: ", "passage: ", 512, 384),
}
RERANKERS = {
    "bge-reranker-v2-m3": ModelSpec("bge-reranker-v2-m3", "onnx-community/bge-reranker-v2-m3-ONNX",
                                    ("onnx/model_int8.onnx", "tokenizer.json"), "onnx/model_int8.onnx", "tokenizer.json", max_len=320),
    "jina-reranker-v2": ModelSpec("jina-reranker-v2", "jinaai/jina-reranker-v2-base-multilingual",
                                  ("onnx/model_int8.onnx", "tokenizer.json"), "onnx/model_int8.onnx", "tokenizer.json", max_len=512),
}


def ensure_files(spec: ModelSpec, models_dir: Path, download: bool = True) -> Path:
    """Скачивает файлы модели в обычную папку (без симлинков HF-кэша — надёжно на Windows)."""
    d = Path(models_dir) / spec.name
    missing = [f for f in spec.files if not (d / f).exists()]
    if missing:
        if not download:
            raise FileNotFoundError(f"модель {spec.name}: нет файлов {missing}")
        from huggingface_hub import hf_hub_download
        for f in missing:
            hf_hub_download(spec.repo, f, local_dir=str(d))
    return d


def _session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "0")) or (os.cpu_count() or 4)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ["CPUExecutionProvider"]
    avail = ort.get_available_providers()
    if os.environ.get("ORT_USE_GPU", "1") == "1":  # автоматически используем GPU, если установлен onnxruntime-gpu / directml / coreml
        for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"):
            if p in avail:
                prov.insert(0, p)
                break
    return ort.InferenceSession(str(path), so, providers=prov)


class Embedder:
    def __init__(self, spec: ModelSpec, models_dir: Path | None = None, download: bool = True):
        from tokenizers import Tokenizer
        self.spec = spec
        d = ensure_files(spec, models_dir or get_settings().models_dir, download)
        self.tok = Tokenizer.from_file(str(d / spec.tokenizer))
        self.tok.enable_truncation(spec.max_len)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 0, pad_token="<pad>")
        self.sess = _session(d / spec.onnx)
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.spec.name

    def _embed(self, texts: list[str], prefix: str, batch: int = 16) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.spec.dim), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok.encode_batch([prefix + texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            with self._lock:
                hid = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (hid * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
            v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
            out[idx] = v
        return out

    def embed_passages(self, texts: list[str], batch: int = 16) -> np.ndarray:
        return self._embed(texts, self.spec.passage_prefix, batch)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], self.spec.query_prefix, 1)[0]


class Reranker:
    def __init__(self, spec: ModelSpec, models_dir: Path | None = None, download: bool = True):
        from tokenizers import Tokenizer
        self.spec = spec
        d = ensure_files(spec, models_dir or get_settings().models_dir, download)
        self.tok = Tokenizer.from_file(str(d / spec.tokenizer))
        self.tok.enable_truncation(spec.max_len)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 1, pad_token="<pad>")
        self.sess = _session(d / spec.onnx)
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.spec.name

    def score(self, query: str, docs: list[str], batch: int = 8) -> list[float]:
        """Вероятность релевантности (sigmoid логита) для каждой пары (запрос, документ)."""
        scores = [0.0] * len(docs)
        order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            enc = self.tok.encode_batch([(query, docs[i]) for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            with self._lock:
                logits = self.sess.run(None, feed)[0].reshape(len(idx), -1)[:, 0]
            for i, lg in zip(idx, logits):
                scores[i] = float(1.0 / (1.0 + np.exp(-float(lg))))
        return scores


def load_embedder(name: str, download: bool = True) -> Embedder | None:
    if not name or name.lower() in ("off", "none"):
        return None
    if name not in EMBEDDERS:
        raise ValueError(f"неизвестная модель эмбеддингов {name}; доступны: {', '.join(EMBEDDERS)}")
    return Embedder(EMBEDDERS[name], download=download)


def load_reranker(name: str, download: bool = True) -> Reranker | None:
    if not name or name.lower() in ("off", "none"):
        return None
    if name not in RERANKERS:
        raise ValueError(f"неизвестный реранкер {name}; доступны: {', '.join(RERANKERS)}")
    return Reranker(RERANKERS[name], download=download)
