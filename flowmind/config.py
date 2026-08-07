"""
flowmind.config — 單一設定來源
=============================================================================
所有模組都從這裡讀設定，不各自呼叫 os.getenv()。
理由：這個系統會同時被 CLI、評測腳本、未來的 API server 使用，
設定散落各處時，「評測用的模型」和「上線用的模型」很容易在不知不覺中不一致，
那樣的評測數字就沒有意義了。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── LLM ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LITELLM_BASE_URL  = os.getenv("LITELLM_BASE_URL", f"{OLLAMA_BASE_URL}/v1")
LITELLM_API_KEY   = os.getenv("LITELLM_API_KEY", "ollama")
LITELLM_PROVIDER  = os.getenv("LITELLM_PROVIDER", "openai")

EXTRACT_MODEL = os.getenv("LLM_EXTRACT_MODEL") or os.getenv("LLM_MODEL", "gemma4:e4b")
ADVISOR_MODEL = os.getenv("LLM_ADVISOR_MODEL") or os.getenv("LLM_MODEL", "gemma4:e4b")
SYNTH_MODEL   = os.getenv("LLM_SYNTH_MODEL")   or ADVISOR_MODEL

# ── Embedding ─────────────────────────────────────────────────────────────
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama").lower()
EMBED_MODEL   = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM     = int(os.getenv("EMBED_DIM", "1024"))
EMBED_DEVICE  = os.getenv("EMBED_DEVICE", "cpu")

# ── PostgreSQL ────────────────────────────────────────────────────────────
PGHOST     = os.getenv("PGHOST", "localhost")
PGPORT     = os.getenv("PGPORT", "5433")
PGDATABASE = os.getenv("PGDATABASE", "flowmind")
PGUSER     = os.getenv("PGUSER", "flowmind_app")
PGPASSWORD = os.getenv("PGPASSWORD", "flowmind_app_pw")

PGADMIN_USER     = os.getenv("PGADMIN_USER", "flowmind")
PGADMIN_PASSWORD = os.getenv("PGADMIN_PASSWORD", "flowmind_dev_pw")


def db_url(admin: bool = False) -> str:
    """admin=True 只給建表/遷移/稽核驗證用；它會繞過 RLS，日常查詢一律 False。"""
    if os.getenv("DATABASE_URL") and not admin:
        return os.environ["DATABASE_URL"]
    user = PGADMIN_USER if admin else PGUSER
    pw = PGADMIN_PASSWORD if admin else PGPASSWORD
    return f"postgresql://{user}:{pw}@{PGHOST}:{PGPORT}/{PGDATABASE}"


# ── 稽核與風險門檻 ────────────────────────────────────────────────────────
ACTOR = os.getenv("FLOWMIND_ACTOR", "dev@flowmind.ai")
CONFIDENCE_ABSTAIN_THRESHOLD = float(os.getenv("CONFIDENCE_ABSTAIN_THRESHOLD", "0.45"))
HUMAN_REVIEW_AMOUNT_TWD = float(os.getenv("HUMAN_REVIEW_AMOUNT_TWD", "5000000"))

# ── 目錄 ──────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"          # data/raw/<tenant_id>/
PROCESSED_DIR = DATA_DIR / "processed"    # data/processed/<tenant_id>/
BENCH_DIR     = DATA_DIR / "benchmarks"
OUT_DIR       = PROJECT_ROOT / "out"

# ── Chunking（沿用 AnalogGenie 已驗證參數，不重新發明）────────────────────
PARENT_MAX_SIZE      = 2500
PARENT_FALLBACK_SIZE = 1500
PARENT_OVERLAP       = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 50
