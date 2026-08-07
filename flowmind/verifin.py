"""
flowmind.verifin — VeriFin：不可 gameable 的文件抽取驗證指標
=============================================================================
【問題：為什麼現有的 RAG／抽取評測分數不值得相信】

目前業界最常見的兩種評測方式各有一個致命弱點：

  (a) LLM-as-judge：請另一個模型評分。
      弱點是它獎勵「寫得像對的」。一段語氣篤定、格式漂亮但內容錯誤的輸出，
      分數往往高於一段誠實說「這份文件沒寫」的輸出。
      而在授信場域，後者才是正確答案。

  (b) 純欄位準確率（accuracy / F1）：算抽對幾格。
      弱點是「猜」永遠不虧。填一個看起來合理的統一編號，
      猜中就加分，猜錯的代價跟留白一樣都是不加分。
      理性的最佳策略因此是「全部都猜」—— 這正是我們最不想要的行為。

VeriFin 用四個彼此獨立的指標把這兩個漏洞堵起來。設計原則只有一條：
**任何一個能提高分數的作法，都必須恰好等同於「真的把文件讀對」。**

─────────────────────────────────────────────────────────────────────────────
1. HPES — Hallucination-Penalized Extraction Score（主指標）

   每個欄位三種結果：答對 +1、留白 0、答錯 −λ（預設 λ=2）。

   為什麼這樣就不可 gameable，是可以算出來的：
       猜一個欄位的期望分數 = p·1 + (1−p)·(−λ)
       留白的期望分數       = 0
       兩者相等時 p = λ/(1+λ)

   λ=2 時 break-even 落在 p = 0.667。也就是說：
   除非模型對某個欄位的把握真的超過 2/3，否則「猜」的期望分數低於「留白」。
   這不是靠規則禁止模型亂猜，而是讓亂猜在數學上就不划算 ——
   這類設計在決策理論裡稱為 proper scoring rule，
   它的性質是「說實話」本身就是最佳策略。

   λ 為什麼是 2 而不是 1 或 10：λ 應該反映真實世界的錯誤成本比。
   在授信送件場域，一個編造的買方統編會讓整份申請被退件、
   甚至讓客戶被列為警示戶；一個空欄位只是要求補件。
   代價大約是「一次退件 vs 一次補件」的量級，2~3 倍是合理的估計。
   λ 是公開參數，任何人都可以帶入自己的成本結構重算。

─────────────────────────────────────────────────────────────────────────────
2. CVR — Citation Verifiability Rate（引用可驗證率）

   模型每抽出一個值，必須同時回報它在原文中的字元區間 [start, end)。
   評分程式直接切出 text[start:end]，檢查它是否真的包含那個值。

   這一步是純字串比對，沒有任何模型參與。
   模型無法用「講得更有說服力」通過，只能真的去定位原文。
   一個從記憶或常識編出來的值，永遠指不出正確的位置。

─────────────────────────────────────────────────────────────────────────────
3. CRC — Counterfactual Robustness under Corruption（反事實穩健度）

   把原文中的標準答案改掉（例如把總額 1,197,000 改成 843,500），
   重跑一次，要求模型的答案跟著改成新值。

   這是在區分兩件從輸出上看不出差別的事：
     · 模型真的讀了這份文件
     · 模型靠版面樣式或訓練時見過的資料猜到答案

   關鍵在於「改成什麼值」是評測當下隨機決定的。
   任何模型都不可能事先記住一個還不存在的數字，所以這一項無法靠背題通過。

─────────────────────────────────────────────────────────────────────────────
4. Risk-Coverage 曲線與 AURC（選擇性預測）

   把所有預測依模型自報的信心排序，逐步降低覆蓋率、觀察錯誤率。
   報告「錯誤率壓在 5% 以下時，還能回答多少比例的欄位」。

   自報信心確實可以灌水，但灌水會立刻反映在這條曲線上：
   把所有欄位的信心都填 0.99，曲線就變成一條水平線，AURC 直接爛掉。
   這一項測的不是信心數字本身，而是它有沒有排序能力。

─────────────────────────────────────────────────────────────────────────────
【一併報告，不合成單一分數】

   四個指標刻意不加權合成一個總分。合成分數是 benchmark 被玩壞的主因 ——
   只要知道權重，就能挑最容易刷的那項下手。
   分開報告的代價是不能宣稱「我們拿了 92 分」，
   但換來的是每一個數字都指向一個明確、可獨立查核的能力。
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

# ══════════════════════════════════════════════════════════════════════════
# 正規化：規則寫死在程式碼裡，供任何人檢查與質疑
# ══════════════════════════════════════════════════════════════════════════

_WS = re.compile(r"\s+")
_MONEY = re.compile(r"^[^\d\-]*(-?[\d,]+(?:\.\d+)?)[^\d]*$")
_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
    (re.compile(r"^(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$"), None),   # 21 Jan 2018
]
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

ABSTAIN_TOKENS = {"", "null", "none", "n/a", "na", "unknown", "未知", "無", "查無",
                  "未提供", "not found", "not specified", "-", "--"}


def is_abstain(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in ABSTAIN_TOKENS


def norm_text(s: Any) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = _WS.sub(" ", s).strip().lower()
    return s.rstrip(".,;:")


def norm_money(s: Any) -> Optional[float]:
    m = _MONEY.match(str(s or "").strip())
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def norm_date(s: Any) -> Optional[tuple[int, int, int]]:
    raw = str(s or "").strip()
    for pat, fn in _DATE_PATTERNS:
        m = pat.match(raw)
        if not m:
            continue
        if fn:
            return fn(m)
        mon = _MONTHS.get(m.group(2).lower())
        return (int(m.group(3)), mon, int(m.group(1))) if mon else None
    return None


def values_match(pred: Any, gold: Any, field_name: str = "") -> bool:
    """
    嚴格比對。刻意不做模糊比對 ——
    模糊比對的門檻是評測者自己選的，等於把「多寬鬆算對」的權力交回給被評的人。
    模糊分數會另外獨立報告，永遠不混進主指標。
    """
    if is_abstain(pred) or is_abstain(gold):
        return is_abstain(pred) and is_abstain(gold)

    fname = field_name.lower()
    if any(k in fname for k in ("total", "amount", "price", "金額", "額")):
        p, g = norm_money(pred), norm_money(gold)
        if p is not None and g is not None:
            return abs(p - g) < 0.01
    if any(k in fname for k in ("date", "日期", "到期")):
        p, g = norm_date(pred), norm_date(gold)
        if p and g:
            return p == g
    return norm_text(pred) == norm_text(gold)


# ══════════════════════════════════════════════════════════════════════════
# 單一欄位的預測
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FieldPrediction:
    field: str
    value: Any = None
    confidence: float = 0.0
    evidence_span: Optional[tuple[int, int]] = None   # [start, end) 於原文的字元位置

    @property
    def abstained(self) -> bool:
        return is_abstain(self.value)


@dataclass
class DocResult:
    doc_id: str
    dataset: str
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    abstain_correct: int = 0     # 標準答案本來就是 null，模型也留白 → 正確的留白
    cited: int = 0               # 有給引用區間的欄位數
    cite_verified: int = 0       # 引用區間確實包含該值的欄位數
    details: list[dict] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════
# 1+2. HPES 與 CVR
# ══════════════════════════════════════════════════════════════════════════

LAMBDA_DEFAULT = 2.0


def score_document(
    doc_id: str,
    dataset: str,
    source_text: str,
    predictions: list[FieldPrediction],
    gold: dict[str, Any],
    lam: float = LAMBDA_DEFAULT,
) -> DocResult:
    res = DocResult(doc_id=doc_id, dataset=dataset)
    src_norm = norm_text(source_text)

    for p in predictions:
        g = gold.get(p.field)
        gold_is_null = is_abstain(g)

        if p.abstained:
            res.abstained += 1
            if gold_is_null:
                res.abstain_correct += 1
            outcome = "abstain_correct" if gold_is_null else "abstain_miss"
        elif values_match(p.value, g, p.field):
            res.correct += 1
            outcome = "correct"
        else:
            res.wrong += 1
            outcome = "wrong"

        # ── CVR：引用區間必須真的框到那個值 ──────────────────────────
        verified = False
        if not p.abstained and p.evidence_span:
            res.cited += 1
            s, e = p.evidence_span
            if 0 <= s < e <= len(source_text):
                quoted = norm_text(source_text[s:e])
                val = norm_text(p.value)
                # 兩個方向都接受：區間包含該值（值是區間的一部分），
                # 或該值包含區間（模型框得比較窄）。都代表它確實定位到原文了。
                verified = bool(quoted) and (val in quoted or quoted in val)
            if verified:
                res.cite_verified += 1

        res.details.append({
            "field": p.field, "pred": p.value, "gold": g,
            "outcome": outcome, "confidence": p.confidence,
            "span": list(p.evidence_span) if p.evidence_span else None,
            "citation_verified": verified,
            # 這個旗標抓的是最糟的一種錯：模型給的值在原文裡根本不存在。
            # 那不是抽錯欄位，是憑空生成。
            "value_absent_from_source": (not p.abstained
                                         and norm_text(p.value) not in src_norm),
        })
    return res


def hpes(results: list[DocResult], lam: float = LAMBDA_DEFAULT) -> dict:
    c = sum(r.correct for r in results)
    w = sum(r.wrong for r in results)
    a = sum(r.abstained for r in results)
    n = c + w + a or 1
    raw = (c - lam * w) / n
    return {
        "lambda": lam,
        "break_even_confidence": round(lam / (1 + lam), 4),
        "n_fields": n,
        "correct": c, "wrong": w, "abstained": a,
        "abstain_correct": sum(r.abstain_correct for r in results),
        # 原始分數落在 [−λ, 1]。同時給出線性映射到 [0,1] 的版本方便閱讀，
        # 但兩個都報，避免有人只引用比較好看的那個。
        "hpes_raw": round(raw, 4),
        "hpes_normalized": round((raw + lam) / (1 + lam), 4),
        "naive_accuracy": round(c / n, 4),   # 對照組：傳統指標會給出的數字
    }


def cvr(results: list[DocResult]) -> dict:
    cited = sum(r.cited for r in results)
    ok = sum(r.cite_verified for r in results)
    answered = sum(r.correct + r.wrong for r in results)
    absent = sum(1 for r in results for d in r.details
                 if d.get("value_absent_from_source"))
    return {
        "answered_fields": answered,
        "with_citation": cited,
        "citation_coverage": round(cited / max(1, answered), 4),
        "verified_citations": ok,
        "citation_verifiability_rate": round(ok / max(1, cited), 4),
        "values_absent_from_source": absent,
        "fabrication_rate": round(absent / max(1, answered), 4),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. CRC — 反事實擾動
# ══════════════════════════════════════════════════════════════════════════

def perturb_value(original: str, rng: random.Random) -> str:
    """
    產生一個「型別相同但值不同」的替代品。
    型別要保持一致，否則模型可能只是因為看到奇怪的東西而拒答，
    那樣就測不出它到底有沒有在讀原文。
    """
    s = str(original)
    if re.fullmatch(r"[\d,]+(\.\d+)?", s.strip()):
        digits = re.sub(r"[^\d]", "", s)
        new = "".join(str(rng.randint(0, 9)) for _ in digits)
        if len(digits) > 1 and new[0] == "0":
            new = str(rng.randint(1, 9)) + new[1:]
        # 保留原本的千分位與小數格式，讓擾動後的文件在版面上看起來一模一樣
        out, i = [], 0
        for ch in s:
            if ch.isdigit():
                out.append(new[i]); i += 1
            else:
                out.append(ch)
        return "".join(out)
    if re.search(r"\d", s):
        return re.sub(r"\d", lambda _: str(rng.randint(0, 9)), s)
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") if ch.isalpha() else ch
                   for ch in s)


def make_counterfactual(text: str, gold: dict, rng: random.Random) -> tuple[str, dict, list[str]]:
    """
    在原文中把標準答案換成隨機新值，回傳（擾動後原文, 新標準答案, 被改動的欄位）。
    只改「在原文中找得到、且只出現一次」的值 ——
    出現多次的值改起來容易牽動其他欄位，會污染這一項的判讀。
    """
    new_text, new_gold, changed = text, dict(gold), []
    for fname, gval in gold.items():
        if is_abstain(gval):
            continue
        gs = str(gval)
        if len(gs) < 3 or new_text.count(gs) != 1:
            continue
        nv = perturb_value(gs, rng)
        if nv == gs:
            continue
        new_text = new_text.replace(gs, nv)
        new_gold[fname] = nv
        changed.append(fname)
    return new_text, new_gold, changed


def crc(original: list[DocResult], counterfactual: list[DocResult],
        changed_fields: dict[str, list[str]]) -> dict:
    """
    只計算「原始題目答對、且該欄位有被擾動」的那些欄位。
    原本就答錯的欄位不納入 —— 那測不出任何東西。
    """
    considered = followed = 0
    by_doc_orig = {r.doc_id: {d["field"]: d for d in r.details} for r in original}
    for r in counterfactual:
        base_id = r.doc_id.replace("::cf", "")
        orig = by_doc_orig.get(base_id, {})
        for d in r.details:
            f = d["field"]
            if f not in changed_fields.get(base_id, []):
                continue
            if orig.get(f, {}).get("outcome") != "correct":
                continue
            considered += 1
            if d["outcome"] == "correct":
                followed += 1
    return {
        "fields_considered": considered,
        "followed_perturbation": followed,
        "counterfactual_robustness": round(followed / max(1, considered), 4),
        "interpretation": ("接近 1.0 代表模型確實在讀文件；"
                           "明顯低於 1.0 代表它有一部分答案來自記憶或版面猜測"),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. Risk-Coverage 與 AURC
# ══════════════════════════════════════════════════════════════════════════

def risk_coverage(results: list[DocResult], target_risk: float = 0.05) -> dict:
    """
    依模型自報信心由高到低排序，逐步納入更多預測，觀察錯誤率如何變化。
    留白不計入（它本來就不是預測）。
    """
    items = [(d["confidence"], d["outcome"] == "correct")
             for r in results for d in r.details
             if d["outcome"] in ("correct", "wrong")]
    if not items:
        return {"n": 0, "aurc": None, "coverage_at_target_risk": None}

    items.sort(key=lambda x: -x[0])
    n = len(items)
    errors = 0
    curve, cov_at_target = [], 0.0
    for i, (_, ok) in enumerate(items, 1):
        if not ok:
            errors += 1
        risk, coverage = errors / i, i / n
        curve.append((round(coverage, 4), round(risk, 4)))
        if risk <= target_risk:
            cov_at_target = coverage

    aurc = sum(r for _, r in curve) / len(curve)
    return {
        "n": n,
        "aurc": round(aurc, 4),                       # 越低越好
        "target_risk": target_risk,
        "coverage_at_target_risk": round(cov_at_target, 4),
        "overall_risk": round(errors / n, 4),
        "curve": curve[::max(1, n // 40)],            # 抽樣後存檔，避免報告過長
        "interpretation": (f"在錯誤率壓到 {target_risk:.0%} 以下的前提下，"
                           f"系統仍能自動處理 {cov_at_target:.1%} 的欄位；"
                           f"其餘應轉人工。這個數字直接對應可自動化比例"),
    }


# ══════════════════════════════════════════════════════════════════════════
# 報告
# ══════════════════════════════════════════════════════════════════════════

def build_report(
    dataset: str, model: str, results: list[DocResult],
    cf_results: Optional[list[DocResult]] = None,
    changed_fields: Optional[dict[str, list[str]]] = None,
    lam: float = LAMBDA_DEFAULT,
) -> dict:
    rep = {
        "dataset": dataset,
        "model": model,
        "n_documents": len(results),
        "HPES": hpes(results, lam),
        "CVR": cvr(results),
        "RiskCoverage": risk_coverage(results),
    }
    if cf_results is not None and changed_fields is not None:
        rep["CRC"] = crc(results, cf_results, changed_fields)
    return rep


def render_report(rep: dict) -> str:
    h, c, rc = rep["HPES"], rep["CVR"], rep["RiskCoverage"]
    L = [
        "═" * 78,
        f"  VeriFin 評測報告　{rep['dataset']}　模型 {rep['model']}",
        f"  文件數 {rep['n_documents']}　欄位數 {h['n_fields']}",
        "═" * 78,
        "",
        "① HPES 幻覺懲罰抽取分數（主指標）",
        f"   答對 {h['correct']}　答錯 {h['wrong']}　留白 {h['abstained']}"
        f"（其中正確留白 {h['abstain_correct']}）",
        f"   HPES(raw, λ={h['lambda']}) = {h['hpes_raw']:+.4f}"
        f"　　正規化至[0,1] = {h['hpes_normalized']:.4f}",
        f"   對照：傳統準確率 = {h['naive_accuracy']:.4f}",
        f"   ↳ 亂猜的損益平衡點在把握度 {h['break_even_confidence']:.1%}；"
        f"低於此值時留白的期望分數較高",
        "",
        "② CVR 引用可驗證率",
        f"   已作答欄位 {c['answered_fields']}　附引用 {c['with_citation']}"
        f"（覆蓋率 {c['citation_coverage']:.1%}）",
        f"   引用區間經字串比對確認為真：{c['citation_verifiability_rate']:.1%}",
        f"   ⚠ 值在原文中完全不存在（憑空生成）：{c['values_absent_from_source']} 個"
        f"（{c['fabrication_rate']:.2%}）",
        "",
        "③ Risk-Coverage 選擇性預測",
        f"   AURC = {rc['aurc']}（越低越好）　整體錯誤率 {rc['overall_risk']:.1%}",
        f"   {rc['interpretation']}",
    ]
    if "CRC" in rep:
        k = rep["CRC"]
        L += ["", "④ CRC 反事實穩健度",
              f"   納入評估欄位 {k['fields_considered']}　"
              f"跟隨擾動改變答案 {k['followed_perturbation']}",
              f"   穩健度 = {k['counterfactual_robustness']:.1%}",
              f"   {k['interpretation']}"]
    L += ["", "═" * 78,
          "  四項指標不合成單一總分：合成分數會讓人挑最好刷的那一項下手，",
          "  分開報告才能讓每個數字各自對應一個可獨立查核的能力。",
          "═" * 78]
    return "\n".join(L)
