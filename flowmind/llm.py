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
from litellm import completion

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
    return _THINK.sub("", text or "").strip()


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
