"""
flowmind.llm — LLM 呼叫層（角色分工 + 受約束解碼）
=============================================================================
兩條路徑，刻意分開：

1. `chat()` 走 LiteLLM。用於顧問問答那種自由文字輸出。
   保留 LiteLLM 而不直接綁 Ollama SDK，是為了讓「本地離線 demo」與
   「未來上雲用 Claude / Azure OpenAI」之間切換只需要改 .env，程式碼零修改。
   對一個要進企業 POC 的產品來說，被單一模型供應商綁死是實質風險。

2. `extract_json()` 走 Ollama 原生 /api/generate 並帶 format=json。
   這不是 prompt 技巧，是 grammar-constrained decoding：
   解碼時直接把不合法 JSON 的 token 機率壓成 0，所以輸出「必然」是合法 JSON。
   文件抽取這一段不能靠「請你只輸出 JSON，謝謝」然後再寫正則去救，
   那在 demo 現場就是一顆定時炸彈。

順帶處理 thinking 類模型：部分本地模型會吐 <think>…</think>，
在 JSON 模式下會直接破格。這裡統一剝除。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import httpx

from . import config

_THINK = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)

_ROLE_MODEL = {
    "extract": lambda: config.EXTRACT_MODEL,
    "advisor": lambda: config.ADVISOR_MODEL,
    "synth":   lambda: config.SYNTH_MODEL,
}


def resolve_model(role: str = "advisor", override: Optional[str] = None) -> str:
    if override:
        return override
    return _ROLE_MODEL.get(role, _ROLE_MODEL["advisor"])()


def strip_thinking(text: str) -> str:
    return decode_byte_fallback(_THINK.sub("", text or "").strip())


# ── byte-fallback token 還原 ──────────────────────────────────────────────
#
# llama.cpp / Ollama 的分詞器在多位元組字被切在 token 邊界時，
# 會吐出「位元組後備 token」，字面上長這樣：<0xE8><0xB3><0x92>
# 那正好是「賒」的 UTF-8 三個位元組（E8 B3 92）。
#
# 【這個 bug 是使用者在畫面上看到才發現的】
# 語料檔案與資料庫裡都是正確的「賒」字，完全沒有污染 ——
# **污染是模型產生的，不是語料壞掉**。
# 一開始的假設（HTML→MD 轉換壞了）指向錯的方向，查證後才確定來源。
#
# 危險之處在於它不會拋錯：那串 <0x..> 就是普通字元，
# 一路通過 JSON 解析、通過引用驗證（它出現在敘述文字而非引用裡）、
# 通過信心計分，最後原封不動印在使用者眼前。
_BYTE_FALLBACK = re.compile(r"(?:<0x[0-9A-Fa-f]{2}>)+")


def decode_byte_fallback(text: str) -> str:
    """把連續的 <0xNN> 還原成原本的字元。無法解碼的原樣保留。"""
    if not text or "<0x" not in text:
        return text or ""

    def _sub(m: re.Match) -> str:
        raw = bytes(int(h, 16) for h in
                    re.findall(r"<0x([0-9A-Fa-f]{2})>", m.group(0)))
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            # 解不出來就保留原樣 —— 硬猜一個字比留著明顯的亂碼危險，
            # 因為亂碼看得出有問題，猜錯的字看不出來。
            return m.group(0)

    return _BYTE_FALLBACK.sub(_sub, text)


def undecoded_byte_tokens(text: str) -> list[str]:
    """回傳仍未解碼的 <0xNN> 片段。空清單代表乾淨。"""
    return [m.group(0) for m in _BYTE_FALLBACK.finditer(text or "")]


# ── 1. 自由文字：走 LiteLLM ────────────────────────────────────────────────

def chat(
    messages: list[dict],
    *,
    role: str = "advisor",
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 600,
    retries: int = 2,
) -> str:
    # 延遲載入：litellm 是重套件，CI 的核心測試只需要 monkeypatch
    # extract_json 就能測 run_verifin 的錯誤處理，不該逼它把 litellm
    # 也裝起來（跟 embeddings.py 延遲載入 sentence_transformers 同理）。
    from litellm import completion          # noqa: PLC0415

    name = resolve_model(role, model)
    # LiteLLM 需要 provider 前綴才知道要走哪套 API schema
    litellm_name = name if "/" in name else f"openai/{name}"

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = completion(
                model=litellm_name,
                api_key=config.LITELLM_API_KEY,
                base_url=config.LITELLM_BASE_URL,
                custom_llm_provider=config.LITELLM_PROVIDER,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return strip_thinking(resp.choices[0].message.content or "")
        except Exception as e:                       # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"LLM 呼叫失敗（{litellm_name}）：{last_err}")


# ── 2. 結構化抽取：走 Ollama 受約束解碼 ────────────────────────────────────

def extract_json(
    prompt: str,
    *,
    schema: Optional[dict] = None,
    model: Optional[str] = None,
    system: Optional[str] = None,
    num_ctx: int = 8192,
    timeout: int = 600,
    retries: int = 1,
) -> tuple[Any, dict]:
    """
    回傳 (解析後的物件, 診斷資訊)。

    schema 若提供，會直接送 JSON Schema 給 Ollama 做結構化輸出，
    連欄位名稱與型別都由解碼器保證，比在 prompt 裡描述 schema 可靠得多。

    診斷資訊裡的 `strict` 欄位很重要：它記錄「第一次解析就成功」與否。
    我們在評測報告裡會如實揭露這個數字，因為一個需要靠正則搶救的抽取器，
    在真實客戶現場的失敗率會比 demo 高得多。
    """
    name = resolve_model("extract", model)
    payload: dict[str, Any] = {
        "model": name,
        "prompt": prompt,
        "stream": False,
        # temperature=0：抽取必須可重現。同一份發票跑兩次得到不同統編，
        # 在會被稽核的場域是不可接受的。
        "options": {"temperature": 0.0, "num_ctx": num_ctx},
        "format": schema if schema else "json",
        "think": False,
        # 實測：gemma4:26b 冷啟動載入要 123.7 秒，熱啟動只要 9 秒。
        # Ollama 預設閒置 5 分鐘就卸載模型 —— demo 現場只要停下來講兩句話，
        # 下一題就要當著評審的面等兩分鐘。keep_alive 從設定檔帶入。
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    if system:
        payload["system"] = system

    diag: dict[str, Any] = {"model": name, "strict": False, "raw": "", "error": None}

    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            r = httpx.post(f"{config.OLLAMA_BASE_URL}/api/generate",
                           json=payload, timeout=timeout)
            # 舊版模型/舊版 Ollama 不吃 think 參數 → 拿掉重試，而不是整個放棄
            if r.status_code == 400 and "think" in payload:
                payload.pop("think")
                r = httpx.post(f"{config.OLLAMA_BASE_URL}/api/generate",
                               json=payload, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            text = strip_thinking(body.get("response", ""))
            diag["raw"] = text
            diag["latency_s"] = round(time.time() - t0, 2)
            diag["out_tokens"] = body.get("eval_count", 0)

            try:
                obj = json.loads(text)
                diag["strict"] = True
                return obj, diag
            except json.JSONDecodeError:
                m = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
                if m:
                    obj = json.loads(m.group(0))
                    diag["strict"] = False
                    return obj, diag
                raise
        except Exception as e:                       # noqa: BLE001
            diag["error"] = str(e)
            if attempt < retries:
                time.sleep(2)

    return None, diag


def chat_local(
    messages: list[dict],
    *,
    role: str = "synth",
    model: Optional[str] = None,
    temperature: float = 0.2,
    num_ctx: int = 32768,
    num_predict: int = 8192,
    timeout: int = 1800,
) -> str:
    """
    長 context 的本地生成，走 Ollama 原生 /api/chat。

    為什麼不能用上面的 chat()：LiteLLM 走的是 OpenAI 相容端點，
    而 Ollama 的 `num_ctx`（context window 大小）只吃原生 API 的 options。
    走相容端點時 Ollama 會套用預設 context（常見是 4096），
    **超過的部分直接從 prompt 前面截掉，而且不會報錯**。

    這個坑實際踩過：skill_builder 一次灌 24 個 chunk（約 6 萬字元）進去，
    六個任務全部回傳空內容，產出的 SKILL.md 每一節都是「本節未產生內容」，
    但程式沒有任何錯誤訊息 —— 因為對 API 來說那是一次成功的呼叫。

    所以需要長 context 的離線合成走這條路；
    需要「換得掉模型供應商」的線上路徑仍然走 chat()。
    兩條路徑的取捨不同，刻意分開而不是勉強共用。
    """
    name = resolve_model(role, model)
    payload: dict[str, Any] = {
        "model": name,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,   # 理由同 extract_json()
    }
    r = httpx.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
    if r.status_code == 400 and "think" in payload:   # 舊模型不吃 think 參數
        payload.pop("think")
        r = httpx.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = strip_thinking((data.get("message") or {}).get("content", ""))
    if not text.strip():
        raise RuntimeError(
            f"{name} 回傳空內容（prompt_eval={data.get('prompt_eval_count')} tokens，"
            f"eval={data.get('eval_count')} tokens）。"
            f"若 prompt_eval 接近 num_ctx={num_ctx}，代表輸入被截斷，請減少注入的 chunk 數。")
    return text


def ollama_available() -> bool:
    try:
        return httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except Exception:                                # noqa: BLE001
        return False


def installed_models() -> list[str]:
    try:
        r = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:                                # noqa: BLE001
        return []
