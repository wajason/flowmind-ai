"""
flowmind.embeddings — 可抽換的向量化後端
=============================================================================
預設走本地 Ollama 的 bge-m3（HTTP），而不是 sentence-transformers。

這個決定是被硬體逼出來的，也值得寫進技術文件：
開發機是 RTX 4060 Laptop，8GB VRAM。sentence-transformers 載入 bge-m3 (FP16)
會固定佔住約 2.4GB；此時 Ollama 想再載入 6.6~9.6GB 的 LLM 就會超過顯存，
Ollama 的處理方式是「靜默退回部分 CPU 推論」—— 程式不會報錯，只會突然慢 5~8 倍，
而且你完全不知道為什麼。

改成兩者都由 Ollama 管理後，它會自己做模型的載入/卸載排程，
把「顯存超賣」變成 Ollama 的問題而不是我們的問題。
副作用是這個專案根本不需要安裝 PyTorch（省下 2.5GB 安裝體積與大量相依衝突）。

bge-m3 的選擇理由：多語言、對中英混雜的金融文件（「Factoring 應收帳款承購」）
表現穩定，1024 維，且與 AnalogGenie 既有資產同一個模型，方法論可直接沿用。
"""

from __future__ import annotations

from typing import Iterable

import httpx

from . import config

_st_model = None


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    # /api/embed（新版）一次可收整批；相較舊的 /api/embeddings 逐條呼叫快很多
    r = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": texts},
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    vectors = data.get("embeddings")
    if vectors is None:
        raise RuntimeError(f"Ollama 未回傳 embeddings：{data}")
    return vectors


def _embed_st(texts: list[str]) -> list[list[float]]:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        name = config.EMBED_MODEL
        # .env 裡為了 Ollama 寫的是短名 'bge-m3'，換後端時補回 HuggingFace 全名
        if "/" not in name:
            name = "BAAI/bge-m3"
        _st_model = SentenceTransformer(name, device=config.EMBED_DEVICE)
    return _st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def embed(texts: Iterable[str], batch_size: int = 64) -> list[list[float]]:
    # batch 從 16 提高到 64：實測入庫 300 頁的白皮書時，
    # 瓶頸不是 GPU 算力而是每次 HTTP 往返的固定開銷。
    # 64 是在「批次夠大」與「單次請求不會撐爆 num_ctx」之間的折衷。
    """把文字批次轉成向量。回傳順序保證與輸入相同。"""
    items = list(texts)
    if not items:
        return []

    fn = _embed_ollama if config.EMBED_BACKEND == "ollama" else _embed_st

    out: list[list[float]] = []
    for i in range(0, len(items), batch_size):
        out.extend(fn(items[i:i + batch_size]))

    if out and len(out[0]) != config.EMBED_DIM:
        raise RuntimeError(
            f"Embedding 維度不符：模型回傳 {len(out[0])} 維，但 schema 是 {config.EMBED_DIM} 維。\n"
            f"換 embedding 模型時必須同步改 EMBED_DIM 並重建向量欄位，"
            f"否則舊向量與新向量在同一個索引裡比較是沒有意義的。"
        )
    return out


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def to_pgvector(vec: list[float]) -> str:
    """pgvector 的文字輸入格式。"""
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"
