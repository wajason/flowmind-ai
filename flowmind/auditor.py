"""
flowmind.auditor — 覆核代理人（Auditor Agent）
=============================================================================
【為什麼需要第四個 agent，以及為什麼不是「多開幾個 agent 比較厲害」】

系統目前有三條產出路徑，各自有各自的把關：

    Extractor  文件 → 結構化欄位        失敗時被 crosscheck 抓到
    Verifier   決定性交叉驗證（純程式）   不會失敗
    Advisor    法規/商品問答（RAG）      幻覺被引用驗證擋下

每一條**各自**都有防線，但**沒有人檢查它們彼此說的話是否一致**。

真實會出事的情境長這樣：
  · Extractor 從發票抽出「帳期 60 天」
  · Advisor 從合約段落引用「約定帳期 90 天」——引用完全正確、驗證通過
  · 兩個都「對」，但**放在同一份報告裡是矛盾的**
  · 授信人員看到一份自相矛盾的報告，信任瞬間歸零

這是**跨 agent 的一致性問題**，任何單一 agent 的自我檢查都抓不到，
因為每個 agent 各自都沒有錯。

【設計原則：Auditor 不做判斷，只做比對】

Auditor **不呼叫 LLM**。它做的是：把不同來源對「同一個事實」的說法
擺在一起，用決定性規則比對。理由與整個系統一致 ——
如果讓 LLM 來裁決兩個 LLM 的分歧，我們只是多了一個會出錯的環節，
而且它的錯誤無法被稽核。

Auditor 能做的是：**指出矛盾，並拒絕放行**。
裁決誰對誰錯是人的工作，而且應該是人的工作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from . import crosscheck, metrics
from .textnorm import normalize_tax_id


class Severity(str, Enum):
    CONTRADICTION = "contradiction"   # 兩個來源直接衝突，必須人工裁決
    UNSUPPORTED = "unsupported"       # 某個說法沒有任何來源支持
    STALE = "stale"                   # 引用了已被取代的版本
    OK = "ok"


@dataclass
class AuditFinding:
    check_id: str
    severity: Severity
    title: str
    detail: str
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    checked: int = 0

    @property
    def contradictions(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity is Severity.CONTRADICTION]

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def releasable(self) -> bool:
        """有矛盾就不放行。這是 Auditor 唯一的權力，也是它存在的理由。"""
        return not self.contradictions

    def as_dict(self) -> dict:
        return {"checked": self.checked, "clean": self.clean,
                "releasable": self.releasable,
                "findings": [f.as_dict() for f in self.findings]}


# ══════════════════════════════════════════════════════════════════════════
# 從自由文字中抽出可比對的斷言
# ══════════════════════════════════════════════════════════════════════════
# 只抽「有明確數值、且該數值有權威來源可比對」的三類。
# 抽得越多不代表越好 —— 抽出無法比對的東西只會製造雜訊。

_PCT = re.compile(r"(?:成數|保證成數)[^0-9零一二三四五六七八九十百分]{0,8}"
                  r"(百分之[零一二三四五六七八九十百]+|[\d.]+\s*%|[一二三四五六七八九十]成)")
_DAYS = re.compile(r"(?:帳期|付款(?:條件|期限)?|票期)[^0-9]{0,6}(\d{1,3})\s*(?:天|日)")
_AMOUNT = re.compile(r"(?:NT\$|新台幣|金額)?\s*([\d][\d,]{4,})\s*元?")
_TAX_ID = re.compile(r"(?:統編|統一編號)[^0-9]{0,4}(\d{8})")

_CN_NUM = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _pct_to_float(s: str) -> Optional[float]:
    """「九成」「百分之九十」「90%」→ 0.9。轉不出來回 None，不猜。"""
    s = s.strip()
    m = re.match(r"([一二三四五六七八九十])成$", s)
    if m:
        return _CN_NUM[m.group(1)] / 10
    m = re.match(r"([\d.]+)\s*%$", s)
    if m:
        return float(m.group(1)) / 100
    if s.startswith("百分之"):
        body = s[3:]
        if body.isdigit():
            return int(body) / 100
        # 百分之九十 / 百分之三十七點五
        total, cur = 0, 0
        for ch in body:
            if ch == "十":
                cur = (cur or 1) * 10
                total += cur
                cur = 0
            elif ch in _CN_NUM:
                cur = _CN_NUM[ch]
        total += cur
        return total / 100 if total else None
    return None


@dataclass
class Assertion:
    kind: str          # guarantee_ratio / payment_terms / amount / tax_id
    value: Any
    raw: str
    source: str        # 哪個 agent / 哪份文件說的


def extract_assertions(text: str, source: str) -> list[Assertion]:
    out: list[Assertion] = []
    for m in _PCT.finditer(text or ""):
        v = _pct_to_float(m.group(1))
        if v is not None:
            out.append(Assertion("guarantee_ratio", v, m.group(0), source))
    for m in _DAYS.finditer(text or ""):
        out.append(Assertion("payment_terms", int(m.group(1)), m.group(0), source))
    for m in _TAX_ID.finditer(text or ""):
        out.append(Assertion("tax_id", m.group(1), m.group(0), source))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 覆核
# ══════════════════════════════════════════════════════════════════════════

def audit(advisor_answer: str,
          tenant_id: str,
          claim_sources: Optional[list[str]] = None,
          extracted_invoices: Optional[list[dict]] = None) -> AuditReport:
    """
    比對 Advisor 的自由文字輸出，與 Extractor／Verifier 的結構化事實。

    三類檢查，全部是決定性比對，不呼叫 LLM。
    """
    rep = AuditReport()
    assertions = extract_assertions(advisor_answer, "advisor")
    rep.checked = len(assertions)

    invoices = extracted_invoices
    if invoices is None:
        data = metrics.load_engagement_files(tenant_id)
        invoices = data["invoices"]
        contracts = data["contracts"]
    else:
        contracts = metrics.load_engagement_files(tenant_id)["contracts"]

    # ── ① 帳期：Advisor 說的 vs 憑證與合約的實際值 ────────────────────
    said_terms = [a for a in assertions if a.kind == "payment_terms"]
    if said_terms and invoices:
        actual = sorted({int(i.get("payment_terms_days") or 0)
                         for i in invoices if i.get("payment_terms_days")})
        contract_terms = sorted({int(c.get("payment_terms_days") or 0)
                                 for c in contracts if c.get("payment_terms_days")})
        for a in said_terms:
            if a.value in actual or a.value in contract_terms:
                continue
            rep.findings.append(AuditFinding(
                "AUD-01", Severity.CONTRADICTION, "帳期與本案憑證不符",
                f"回覆中提到「{a.raw}」，但本案發票的實際帳期為 "
                f"{actual or '（無）'}、合約約定為 {contract_terms or '（無）'}。"
                f"兩者不一致，需人工確認回覆中的數字是引用自法規通則、"
                f"還是誤植為本案的實際條件。",
                sources=["advisor", "receivables.json", "contracts.json"]))

    # ── ② 統一編號：Advisor 提到的統編是否真的存在於本案憑證 ──────────
    said_ids = [a for a in assertions if a.kind == "tax_id"]
    if said_ids and invoices:
        known = {normalize_tax_id(i.get("buyer_ban")) for i in invoices} | \
                {normalize_tax_id(i.get("seller_ban")) for i in invoices}
        known.discard(None)
        for a in said_ids:
            if normalize_tax_id(a.value) in known:
                continue
            rep.findings.append(AuditFinding(
                "AUD-02", Severity.UNSUPPORTED, "回覆中的統一編號不存在於本案憑證",
                f"回覆提到統編 {a.value}，但本案的發票中沒有這個統編。"
                f"這可能是引用自公開文件的範例，也可能是幻覺 —— 需人工確認。",
                sources=["advisor", "receivables.json"]))

    # ── ③ 保證成數：與法規原文的比對留給引用驗證，這裡只查內部一致性 ────
    ratios = {a.value for a in assertions if a.kind == "guarantee_ratio"}
    if len(ratios) > 1:
        rep.findings.append(AuditFinding(
            "AUD-03", Severity.CONTRADICTION, "同一份回覆中出現不同的保證成數",
            f"回覆中同時出現 {sorted(f'{r:.0%}' for r in ratios)} 兩種以上的成數。"
            f"不同方案的成數本來就不同，但同一份回覆必須說清楚哪個數字對應哪個方案，"
            f"否則讀的人會誤用。",
            sources=["advisor"]))

    # ── ④ 與決定性驗證結果的一致性 ────────────────────────────────────
    if invoices:
        cross = crosscheck.run_all(invoices, contracts,
                                   metrics.load_engagement_files(tenant_id)["ledger"])
        critical = [f for f in cross["findings"]
                    if not f["passed"] and f["severity"] == "critical"]
        positive = re.search(r"(可以送件|建議送件|沒有問題|皆無異常|全部通過)",
                             advisor_answer or "")
        if critical and positive:
            rep.findings.append(AuditFinding(
                "AUD-04", Severity.CONTRADICTION,
                "回覆稱可送件，但決定性驗證有重大缺失",
                f"回覆中出現「{positive.group(0)}」，但交叉驗證有 "
                f"{len(critical)} 項重大缺失："
                + "、".join(f["check_id"] for f in critical) +
                "。決定性驗證的結果優先於語言模型的敘述。",
                sources=["advisor", "crosscheck"]))

    return rep


def render(rep: AuditReport) -> str:
    if rep.clean:
        return (f"✅ 覆核代理人：檢視 {rep.checked} 項可比對斷言，"
                f"未發現跨來源矛盾。")
    icon = {Severity.CONTRADICTION: "🔴", Severity.UNSUPPORTED: "🟠",
            Severity.STALE: "🟡", Severity.OK: "✅"}
    L = ["", "─" * 80,
         f"🔍 覆核代理人（Auditor）　檢視 {rep.checked} 項斷言，"
         f"發現 {len(rep.findings)} 項問題", "─" * 80]
    for f in rep.findings:
        L.append(f"{icon[f.severity]} [{f.check_id}] {f.title}")
        L.append(f"     {f.detail}")
        L.append(f"     涉及來源：{'、'.join(f.sources)}")
    if not rep.releasable:
        L += ["", "⛔ **存在跨來源矛盾，本回覆不予放行。**",
              "   Auditor 不裁決誰對誰錯 —— 那是人的工作，而且應該是人的工作。",
              "   它能做的是指出矛盾並擋下輸出。"]
    L.append("─" * 80)
    return "\n".join(L)
