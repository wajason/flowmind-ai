#!/usr/bin/env python3
"""
eval_models.py — 本地 Ollama 模型選型實測
============================================================================
不用「感覺哪個模型比較強」來決定，而是針對 FlowMind 真正會做的三件事實測：

  T1 抽取遵從度 (Extraction Compliance)
     給一段中文發票 OCR 文字，要求輸出嚴格 JSON。
     金融文件抽取如果 JSON 破格，整條 pipeline 就斷了 —— 這是硬需求，不是加分項。

  T2 拒答紀律 (Abstention Discipline)
     給一段「刻意缺少統一編號」的文字，正確行為是該欄位填 null，而不是編一個。
     這一項直接對應我們的核心賣點：寧可留白，不可臆測。

  T3 中文金融語感 (Domain Fluency)
     用供應鏈金融術語提問，看它是否知道「有追索權/無追索權承購」的差別。

同時量測 first-token 延遲與 tokens/sec —— 決賽 demo 現場等 40 秒是會失分的。

Usage:
    python scripts/eval_models.py
    python scripts/eval_models.py --models qwen3.5:9b gpt-oss:20b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DEFAULT_MODELS = ["qwen3.5:9b", "gemma4:e4b", "gpt-oss:20b"]

# ── T1：抽取遵從度 ──────────────────────────────────────────────────────────
T1_SCHEMA_PROMPT = """你是文件抽取引擎。只輸出 JSON，不要任何說明文字、不要 markdown code fence。
JSON schema：
{"buyer_name": string|null, "buyer_tax_id": string|null, "invoice_number": string|null,
 "invoice_date": string|null, "total_amount": number|null, "currency": string|null}
規則：文件中沒有明確寫出的欄位一律填 null，嚴禁推測或補完。

文件內容：
統一發票
賣方：晶鴻精密工業股份有限公司  統編：84726193
買方：宏昇機械股份有限公司      統編：22099131
發票號碼：AB-45678901
開立日期：中華民國115年6月18日
品名：CNC 主軸組件  數量 40  單價 28,500
銷售額合計：1,140,000
營業稅：57,000
總計：1,197,000
"""
T1_EXPECT = {
    "buyer_name": "宏昇機械股份有限公司",
    "buyer_tax_id": "22099131",
    "invoice_number": "AB-45678901",
    "total_amount": 1197000,
}

# ── T2：拒答紀律（買方統編被刻意拿掉）────────────────────────────────────────
T2_PROMPT = T1_SCHEMA_PROMPT.replace("買方：宏昇機械股份有限公司      統編：22099131",
                                     "買方：宏昇機械股份有限公司")

# ── T3：中文金融語感 ────────────────────────────────────────────────────────
T3_PROMPT = (
    "用三句話說明：中小企業做「無追索權應收帳款承購」相較「有追索權」，"
    "在會計上與銀行授信上最關鍵的差別是什麼？若你不確定，請直接說不確定。"
)
T3_KEYWORDS = ["無追索權", "有追索權", "資產負債表", "除列", "off-balance",
               "買方信用", "呆帳", "風險移轉", "表外"]


def call(model: str, prompt: str, fmt_json: bool = False, timeout: int = 300) -> dict:
    """呼叫 Ollama /api/generate，回傳文字 + 效能數據。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    if fmt_json:
        payload["format"] = "json"   # Ollama 的 grammar-constrained 解碼，強制合法 JSON
    t0 = time.time()
    r = httpx.post(f"{OLLAMA}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    wall = time.time() - t0
    eval_count = d.get("eval_count", 0)
    eval_dur = d.get("eval_duration", 1) / 1e9
    return {
        "text": d.get("response", ""),
        "wall_s": wall,
        "load_s": d.get("load_duration", 0) / 1e9,
        "ttft_s": d.get("prompt_eval_duration", 0) / 1e9,
        "tok_per_s": (eval_count / eval_dur) if eval_dur > 0 else 0.0,
        "out_tokens": eval_count,
    }


def parse_json_loose(text: str):
    """先試嚴格解析；失敗才剝 code fence。能不能嚴格解析本身就是評分項。"""
    try:
        return json.loads(text), True          # strict=True：一次過
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), False
        except Exception:
            return None, False
    return None, False


def norm_amount(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return float(re.sub(r"[^\d.]", "", str(v)) or 0)


def score_model(model: str) -> dict:
    print(f"\n{'═'*72}\n  ⏱  測試模型：{model}\n{'═'*72}")
    res = {"model": model}

    # T1
    r1 = call(model, T1_SCHEMA_PROMPT, fmt_json=True)
    obj, strict = parse_json_loose(r1["text"])
    hit = 0
    if obj:
        for k, want in T1_EXPECT.items():
            got = obj.get(k)
            if k == "total_amount":
                hit += int(norm_amount(got) == float(want))
            else:
                hit += int(str(got or "").strip() == want)
    res["t1_field_acc"] = hit / len(T1_EXPECT)
    res["t1_strict_json"] = strict
    res["t1_tok_per_s"] = r1["tok_per_s"]
    res["t1_wall_s"] = r1["wall_s"]
    print(f"  T1 抽取正確率 {res['t1_field_acc']:.0%}  嚴格JSON={strict}  "
          f"{r1['tok_per_s']:.1f} tok/s  牆鐘 {r1['wall_s']:.1f}s")

    # T2：買方統編應為 null
    r2 = call(model, T2_PROMPT, fmt_json=True)
    obj2, _ = parse_json_loose(r2["text"])
    tax = (obj2 or {}).get("buyer_tax_id")
    res["t2_abstained"] = tax in (None, "", "null", "N/A", "無")
    # 最糟的情況：把賣方統編搬過來當買方 —— 這是真實會發生的幻覺模式
    res["t2_cross_contaminated"] = str(tax or "").strip() == "84726193"
    print(f"  T2 缺欄位是否正確留白：{'✅ 是' if res['t2_abstained'] else '❌ 否 → ' + str(tax)}"
          f"{'  ⚠️ 且是把賣方統編搬過來' if res['t2_cross_contaminated'] else ''}")

    # T3
    r3 = call(model, T3_PROMPT)
    txt = r3["text"]
    kw = sum(1 for k in T3_KEYWORDS if k in txt)
    # 簡體字混入是本地中文模型常見問題，會直接讓提案書/報告失去專業感
    simplified = len(re.findall(r"[账觉务责险产资产标准据]", txt))
    res["t3_keyword_hits"] = kw
    res["t3_simplified_chars"] = simplified
    res["t3_tok_per_s"] = r3["tok_per_s"]
    print(f"  T3 術語命中 {kw}/{len(T3_KEYWORDS)}　簡體字元 {simplified} 個　"
          f"{r3['tok_per_s']:.1f} tok/s")
    print(f"     ↳ {txt.strip()[:160]}...")

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()

    rows = []
    for m in args.models:
        try:
            rows.append(score_model(m))
        except Exception as e:
            print(f"  ❌ {m} 測試失敗：{e}")

    print(f"\n{'═'*92}\n  📊 選型總表\n{'═'*92}")
    print(f"{'模型':<18}{'T1抽取':<10}{'嚴格JSON':<10}{'T2留白':<10}{'T3術語':<9}{'簡體字':<8}{'tok/s':<8}")
    print("─" * 92)
    for r in rows:
        print(f"{r['model']:<18}{r['t1_field_acc']:<10.0%}"
              f"{str(r['t1_strict_json']):<10}{str(r['t2_abstained']):<10}"
              f"{r['t3_keyword_hits']:<9}{r['t3_simplified_chars']:<8}{r['t1_tok_per_s']:<8.1f}")
    print("═" * 92)
    print("\n判讀原則：T2（該留白時留白）優先於 T1（抽得多）。")
    print("在金融場域，一個編造的統一編號造成的損害遠大於一個空欄位。")

    with open("model_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("→ 明細已寫入 model_eval_results.json")


if __name__ == "__main__":
    main()
