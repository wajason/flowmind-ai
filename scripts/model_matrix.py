#!/usr/bin/env python3
"""
model_matrix.py — 多模型完整評測矩陣
=============================================================================
【為什麼不能只用一個指標選模型】

先前只比了 2 個模型、只看 HPES 一個指標就下結論，那不嚴謹。
本程式對每個候選模型量測**五個獨立面向**，並且明確標示
「哪些是硬需求（不達標就淘汰）」與「哪些是取捨」。

| 面向 | 指標 | 性質 |
|---|---|---|
| 結構遵從 | 嚴格 JSON 成功率 | **硬需求**：破格就整條 pipeline 斷掉 |
| 輸出可重現 | 位元級／結構級漂移 | **硬需求**：稽核要求可重現（Basel III / MiFID II） |
| 抽取正確 | 欄位準確率 | 取捨 |
| 拒答紀律 | HPES（答錯 −2） | **主指標**：授信場域的成本不對稱 |
| 語言品質 | 繁中純度（簡體字數） | **硬需求**：對台灣金融客戶交付簡體字不可接受 |
| 效能 | 中位延遲、tok/s | 取捨 |

【關於模型來源的考量】

台灣金融機構對資料主權與供應鏈來源有實質考量。
本評測**同時納入中資與非中資模型**做為對照 ——
不比較就無法說明「選非中資是否付出了效能代價」，
那樣的推薦沒有說服力。但最終部署建議會標明來源國。
這是**如實比較後的取捨說明**，不是預設立場。

Usage:
    python scripts/model_matrix.py                       # 全部候選模型
    python scripts/model_matrix.py --models granite4.1:8b phi4-mini:3.8b
    python scripts/model_matrix.py --drift-runs 10 --extract-n 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, drift, llm, textnorm, verifin           # noqa: E402
from flowmind.verifin import FieldPrediction                         # noqa: E402

BENCH = config.DATA_DIR / "benchmarks" / "sroie.jsonl"

# 候選模型與來源國。來源國如實標註，供部署決策參考。
CANDIDATES = [
    # ── 非中資（部署首選）────────────────────────────────────────────
    ("gemma4:e4b",         "Google · 美國",    "9.6GB，先前的抽取模型"),
    ("gemma4:26b",         "Google · 美國",    "17GB，測「大模型是否更穩定」"),
    ("gpt-oss:20b",        "OpenAI · 美國",    "13GB。★論文中一致性僅 12.5% 的模型，"
                                              "本評測嘗試複現該發現"),
    ("llama3.2:latest",    "Meta · 美國",      "2GB，小模型下限"),
    # ── 中資（納入對照，非部署選項）──────────────────────────────────
    ("qwen3.5:9b",         "Alibaba · 中國",   "6.6GB，先前選用；納入以如實比較效能代價"),
    ("qwen3.6:35b",        "Alibaba · 中國",   "23GB，大模型對照"),

    # ── 以下為原訂候選，因網路問題無法下載，如實記錄未評測 ──────────────
    # granite4.1:8b / granite3.3:8b（IBM）—— 論文中一致性 100% 的系列，最想測
    # phi4-mini:3.8b（Microsoft）· mistral-nemo:12b（法國）· olmo2:7b（AI2）
    # 全部因 Cloudflare R2 連線失敗而未能取得。**不列入結果、不做推測。**
]

# 原訂但未能下載的候選，如實揭露在報告中
UNAVAILABLE = [
    ("granite4.1:8b / granite3.3:8b", "IBM · 美國",
     "★論文中輸出一致性 100% 的系列，最想測的對照組"),
    ("phi4-mini:3.8b", "Microsoft · 美國", "小模型下限"),
    ("mistral-nemo:12b", "Mistral · 法國", "歐洲來源，資料主權考量下的選項"),
    ("olmo2:7b", "AI2 · 美國", "完全開放（含訓練資料）"),
]

# ── 漂移測試用的固定 prompt（中文 B2B 發票，貼近真實任務）───────────
DRIFT_DOC = """統一發票
賣方：晶鴻精密工業股份有限公司  統編：20433212
買方：宏昇機械股份有限公司      統編：04595257
發票號碼：AB-45678901
開立日期：中華民國115年6月18日
品名：CNC 主軸組件  數量 40  單價 28,500
銷售額合計：1,140,000
營業稅：57,000
總計：1,197,000
付款條件：發票日後 60 天"""

DRIFT_SCHEMA = {
    "type": "object",
    "properties": {
        "buyer_name": {"type": ["string", "null"]},
        "buyer_tax_id": {"type": ["string", "null"]},
        "seller_tax_id": {"type": ["string", "null"]},
        "invoice_number": {"type": ["string", "null"]},
        "invoice_date": {"type": ["string", "null"]},
        "sales_amount": {"type": ["number", "null"]},
        "tax_amount": {"type": ["number", "null"]},
        "total_amount": {"type": ["number", "null"]},
        "payment_terms_days": {"type": ["number", "null"]},
    },
    "required": ["buyer_name", "buyer_tax_id", "seller_tax_id", "invoice_number",
                 "invoice_date", "sales_amount", "tax_amount", "total_amount",
                 "payment_terms_days"],
}

DRIFT_PROMPT = (
    "你是文件抽取引擎。只輸出 JSON，不要任何說明文字。\n"
    "文件中沒有明確寫出的欄位一律填 null，嚴禁推測。\n\n"
    f"【文件】\n{DRIFT_DOC}")

DRIFT_EXPECT = {
    "buyer_tax_id": "04595257", "seller_tax_id": "20433212",
    "invoice_number": "AB-45678901", "total_amount": 1197000,
    "sales_amount": 1140000, "tax_amount": 57000, "payment_terms_days": 60,
}

# 中文品質：僅列「繁體中不會出現」的簡體字（見 textnorm 的說明）
SIMPLIFIED_PROBE = textnorm._SIMPLIFIED_PROBE

# 繁中純度：**多題**而非單題。
# 用 n=1 的樣本就把一個模型判定為「硬需求不達標」是不嚴謹的 ——
# 簡體字是否出現與主題用詞高度相關（「貸款／帳期／發票」這類金融詞
# 恰好落在簡繁差異密集區），單一題目可能剛好避開、也可能剛好命中。
# 故取 4 個涵蓋不同金融子領域的題目合併計算，並保留逐題結果供檢視。
ZH_PROMPTS = [
    "用三到四句話說明：中小企業辦理「無追索權應收帳款承購」，"
    "相較「有追索權」，在會計處理與銀行授信上最關鍵的差別為何？若不確定請直接說不確定。",

    "用三到四句話說明：銀行審查中小企業的應收帳款融資時，"
    "為什麼會特別在意買方（債務人）的信用而非賣方？若不確定請直接說不確定。",

    "用三到四句話說明：什麼是「帳期」？供應商拉長帳期會對自身現金流造成什麼影響？"
    "若不確定請直接說不確定。",

    "用三到四句話說明：信用保證機制如何降低銀行承作中小企業貸款的風險？"
    "若不確定請直接說不確定。",
]


def _ollama_text(model: str, prompt: str, *, schema=None, num_ctx: int = 8192,
                 seed: int | None = 42, timeout: float = 900) -> str:
    """
    送一個 prompt 給 Ollama，取回純文字輸出（已去除 <think>）。

    為什麼需要 /api/chat 的 fallback：
      部分模型（實測：gpt-oss:20b）**只註冊了 chat template，不支援 /api/generate**，
      直接回 400 "does not support generate"。
      若不處理，一個完全可用的模型會被記成「評測失敗」——
      那是評測工具的缺陷，不是模型的缺陷，不該汙染選型結論。
    """
    opts: dict = {"temperature": 0.0, "num_ctx": num_ctx}
    if seed is not None:
        opts["seed"] = seed
    base: dict = {"model": model, "stream": False, "options": opts, "think": False}
    if schema is not None:
        base["format"] = schema

    def _post(path: str, payload: dict):
        p = dict(payload)
        r = httpx.post(f"{config.OLLAMA_BASE_URL}{path}", json=p, timeout=timeout)
        if r.status_code == 400 and "think" in p:          # 舊版 Ollama 不認得 think
            p.pop("think")
            r = httpx.post(f"{config.OLLAMA_BASE_URL}{path}", json=p, timeout=timeout)
        return r

    r = _post("/api/generate", {**base, "prompt": prompt})
    if r.status_code == 400 and "does not support generate" in r.text:
        r = _post("/api/chat", {**base, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return llm.strip_thinking((r.json().get("message") or {}).get("content", ""))
    r.raise_for_status()
    return llm.strip_thinking(r.json().get("response", ""))


def _zh_purity_probe(model: str) -> tuple[int, int, list[dict]]:
    """
    繁中純度：跑 ZH_PROMPTS 全部題目並合併計數。

    回傳 (簡體字總數, 中文字總數, 逐題明細)。
    逐題明細保留「實際出現了哪些簡體字」，讓「純度不足因此淘汰」這個
    判定可以被逐字檢驗，而不是丟一個數字要人相信。
    """
    simp = cjk = 0
    per_prompt: list[dict] = []
    for qi, q in enumerate(ZH_PROMPTS, 1):
        txt = _ollama_text(model, q, schema=None, num_ctx=4096, seed=None)
        found = sorted({ch for ch in txt if ch in SIMPLIFIED_PROBE})
        s = sum(1 for ch in txt if ch in SIMPLIFIED_PROBE)
        c = sum(1 for ch in txt if "一" <= ch <= "鿿")
        simp += s
        cjk += c
        per_prompt.append({"prompt_index": qi, "simplified": s, "cjk": c,
                           "simplified_chars": found, "sample": txt[:180]})
    return simp, cjk, per_prompt


def _extract_once(model: str, prompt: str, schema=None):
    t0 = time.time()
    raw = _ollama_text(model, prompt, schema=schema or "json")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None
    return raw, obj, time.time() - t0


# ══════════════════════════════════════════════════════════════════════════
def eval_model(model: str, origin: str, note: str,
               drift_runs: int, extract_n: int) -> dict:
    out = {"model": model, "origin": origin, "note": note}
    print(f"\n{'─'*84}\n  {model}　（{origin}）\n{'─'*84}")

    # ── ① 輸出漂移 ────────────────────────────────────────────────
    print(f"    [1/4] 輸出漂移（{drift_runs} 次相同 prompt，T=0、固定 seed）…",
          end=" ", flush=True)
    d = drift.measure(model, lambda: _extract_once(model, DRIFT_PROMPT, DRIFT_SCHEMA),
                      runs=drift_runs)
    out["drift"] = {"runs": d.runs, "bitwise": d.bitwise_rate,
                    "structural": d.structural_rate, "semantic": d.semantic_rate,
                    "distinct_outputs": d.distinct_outputs, "tier": d.tier,
                    "tier_label": d.tier_label, "error": d.error}
    print("❌ " + (d.error or "") if d.error else
          f"位元 {d.bitwise_rate:.0%} / 結構 {d.structural_rate:.0%} → {d.tier_label}")
    if d.error:
        return out

    # ── ② 固定文件的抽取正確性 ────────────────────────────────────
    print(f"    [2/4] 固定文件抽取…", end=" ", flush=True)
    raw, obj, lat = _extract_once(model, DRIFT_PROMPT, DRIFT_SCHEMA)
    hit = 0
    if isinstance(obj, dict):
        for k, want in DRIFT_EXPECT.items():
            got = obj.get(k)
            if isinstance(want, (int, float)):
                try:
                    hit += int(float(str(got).replace(",", "")) == float(want))
                except (TypeError, ValueError):
                    pass
            else:
                hit += int(str(got or "").strip().replace("‑", "-") == want)
    out["fixed_doc_accuracy"] = round(hit / len(DRIFT_EXPECT), 4)
    out["fixed_doc_latency_s"] = round(lat, 2)
    print(f"{out['fixed_doc_accuracy']:.0%}（{lat:.1f}s）")

    # ── ③ SROIE 上的 HPES 與嚴格 JSON ─────────────────────────────
    print(f"    [3/4] SROIE {extract_n} 份文件（HPES / 嚴格 JSON）…", flush=True)
    rows = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines()
            if l.strip()][:extract_n]
    results, strict_fail, t0 = [], 0, time.time()
    for i, row in enumerate(rows, 1):
        schema = {"type": "object", "properties": {
            "fields": {"type": "array", "items": {"type": "object", "properties": {
                "field": {"type": "string"}, "value": {"type": ["string", "null"]},
                "confidence": {"type": "number"}}, "required": ["field", "value", "confidence"]}}},
            "required": ["fields"]}
        p = (f"你是文件抽取引擎。只輸出 JSON。文件沒寫的欄位填 null，嚴禁推測。\n"
             f"計分：答對 +1、留白 0、答錯 −2。把握度低於 67% 時留白對你有利。\n\n"
             f"【文件】\n{row['text'][:5000]}\n\n"
             f"【欄位】\n" + "\n".join(f"  - {f}" for f in row["ask_fields"]))
        try:
            raw, obj, _ = _extract_once(model, p, schema)
        except Exception:                              # noqa: BLE001
            strict_fail += 1
            continue
        if obj is None:
            strict_fail += 1
            continue
        got = {str(x.get("field")): x for x in (obj.get("fields") or [])
               if isinstance(x, dict)}
        preds = []
        for f in row["ask_fields"]:
            it = got.get(f, {})
            try:
                c = float(it.get("confidence") or 0)
            except (TypeError, ValueError):
                c = 0.0
            preds.append(FieldPrediction(field=f, value=it.get("value"),
                                         confidence=max(0.0, min(1.0, c))))
        results.append(verifin.score_document(row["doc_id"], "SROIE",
                                              row["text"], preds, row["gold"]))
        if i % 10 == 0:
            print(f"          {i}/{len(rows)}　{time.time()-t0:.0f}s", flush=True)

    if results:
        h = verifin.hpes(results)
        out["hpes_raw"] = h["hpes_raw"]
        out["naive_accuracy"] = h["naive_accuracy"]
        out["abstain_count"] = h["abstained"]
        out["wrong_count"] = h["wrong"]
        out["correct_count"] = h["correct"]
    out["strict_json_failure_rate"] = round(strict_fail / max(1, len(rows)), 4)
    out["sroie_elapsed_s"] = round(time.time() - t0, 1)
    print(f"      HPES {out.get('hpes_raw', 0):+.4f}　"
          f"準確率 {out.get('naive_accuracy', 0):.1%}　"
          f"嚴格JSON失敗 {out['strict_json_failure_rate']:.0%}")

    # ── ④ 繁體中文品質 ────────────────────────────────────────────
    print(f"    [4/4] 繁體中文品質…", end=" ", flush=True)
    try:
        t0 = time.time()
        simp, cjk, per_prompt = _zh_purity_probe(model)
        out["zh_simplified_chars"] = simp
        out["zh_cjk_chars"] = cjk
        out["zh_purity"] = round(1 - simp / cjk, 4) if cjk else None
        out["zh_prompts_n"] = len(ZH_PROMPTS)
        out["zh_per_prompt"] = per_prompt
        out["zh_latency_s"] = round((time.time() - t0) / len(ZH_PROMPTS), 2)
        out["zh_sample"] = per_prompt[0]["sample"] if per_prompt else ""
        print(f"簡體字 {simp} 個 / 中文 {cjk} 字"
              f"（純度 {out['zh_purity']:.2%}，{len(ZH_PROMPTS)} 題合計）"
              if cjk else "無中文輸出")
    except Exception as e:                             # noqa: BLE001
        out["zh_error"] = str(e)[:120]
        print(f"❌ {e}")
    return out


def render(rows: list[dict]) -> str:
    ok = [r for r in rows if not r.get("drift", {}).get("error")]
    L = [
        "", "═" * 118,
        "  多模型評測矩陣",
        "═" * 118, "",
        f"  {'模型':<20}{'來源':<16}{'漂移':>7}{'結構':>7}{'JSON失敗':>9}"
        f"{'準確率':>8}{'HPES':>9}{'留白':>6}{'繁中純度':>9}{'延遲':>8}  分級",
        "  " + "─" * 114,
    ]
    for r in ok:
        d = r["drift"]
        L.append(
            f"  {r['model']:<20}{r['origin']:<16}"
            f"{d['bitwise']:>7.0%}{d['structural']:>7.0%}"
            f"{r.get('strict_json_failure_rate', 0):>9.0%}"
            f"{r.get('naive_accuracy', 0):>8.1%}"
            f"{r.get('hpes_raw', 0):>+9.3f}"
            f"{r.get('abstain_count', 0):>6}"
            f"{(r.get('zh_purity') or 0):>9.2%}"
            f"{r.get('fixed_doc_latency_s', 0):>7.1f}s  T{d['tier']}")
    bad = [r for r in rows if r.get("drift", {}).get("error")]
    for r in bad:
        L.append(f"  {r['model']:<20}{r['origin']:<16}  ❌ {r['drift']['error'][:56]}")

    L += [
        "", "─" * 118,
        "  【硬需求 vs 取捨】",
        "    硬需求（不達標即淘汰）：嚴格 JSON 失敗率 = 0%、位元級漂移 = 100%、繁中純度 = 100%",
        "      · JSON 破格 → 整條 pipeline 斷掉",
        "      · 輸出漂移 → 稽核無法回答「當初這個建議是根據什麼給的」（Basel III / MiFID II）",
        "      · 簡體字   → 對台灣金融客戶交付不可接受",
        "    取捨：準確率、HPES、延遲 —— 在硬需求都滿足的候選之間才比較",
        "",
        "  【HPES 為何優先於準確率】",
        "    答對 +1、留白 0、答錯 −2。授信場域一個編造的統編比一個空欄位貴太多，",
        "    所以「留白次數」欄位要一起看：留白 0 次代表這個模型幾乎不承認不知道。",
        "",
        "  【來源國】如實標註供部署決策參考。同時納入中資模型比較，",
        "    是為了說明選擇非中資是否付出效能代價 —— 不比較的推薦沒有說服力。",
        "",
        "  【未能評測的候選（如實揭露，不做推測）】",
    ]
    for name, origin, note in UNAVAILABLE:
        L.append(f"    ✗ {name:<32}{origin:<16}{note}")
    L += [
        "      以上因 Ollama registry 的 Cloudflare R2 連線持續失敗而未能下載。",
        "      **不列入結果、不以推測填補。** 網路恢復後應補測，",
        "      尤其 IBM Granite —— 那是論文中一致性 100% 的系列，是最重要的對照組。",
        "═" * 118,
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--drift-runs", type=int, default=8)
    ap.add_argument("--extract-n", type=int, default=15)
    ap.add_argument("--out", default="docs/MODEL_MATRIX.json")
    args = ap.parse_args()

    installed = set(llm.installed_models())
    cands = [c for c in CANDIDATES
             if (not args.models or c[0] in args.models)]
    cands = [c for c in cands
             if c[0] in installed or c[0] + ":latest" in installed
             or any(m.startswith(c[0].split(":")[0]) for m in installed)]

    print("═" * 84)
    print(f"  多模型評測　{len(cands)} 個模型")
    print(f"  漂移 {args.drift_runs} 次／模型　SROIE {args.extract_n} 份文件／模型")
    print("═" * 84)

    rows = []
    for m, origin, note in cands:
        try:
            rows.append(eval_model(m, origin, note, args.drift_runs, args.extract_n))
        except Exception as e:                         # noqa: BLE001
            print(f"  ❌ {m} 評測失敗：{e}")
            rows.append({"model": m, "origin": origin,
                         "drift": {"error": str(e)[:160]}})

    print(render(rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"),
         "drift_runs": args.drift_runs, "extract_n": args.extract_n,
         "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
