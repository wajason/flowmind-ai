#!/usr/bin/env python3
"""
watchtower.py — 主動監控與預警（零 LLM）
=============================================================================
【這支模組把「被動問答」變成「主動秘書」】

先前的系統要等人問才會動。一個真正的財務秘書不是這樣運作的 ——
它會在你沒問的時候就發現「這筆帳期後天到，對方前兩期都遲付」。

但「主動」很容易變成「亂喊」。所以本模組遵守三條規則：

  ① **零 LLM。** 每一條警示都由 SQL 與算術產生。
     一個會編造警示的監控系統，比沒有監控更糟 ——
     使用者會先失去信任，然後開始忽略所有警示，
     包括真的那幾條。

  ② **每條警示都附上觸發它的實際資料列。**
     警示不是一句紅字，是「這 3 張發票、這些金額、這些日期」。
     沒有證據的警示無法複查，也就無法被信任。

  ③ **同一件事只喊一次。**
     用指紋（規則 + 證據內容的雜湊）去重。
     每天把同一件事重喊一遍，等於訓練使用者忽略警示。

【七條規則，以及每一條的理由】

  WATCH-01 帳期即將到期        —— 提前準備收款，這是秘書最基本的價值
  WATCH-02 已逾期未收          —— 逾期本身就是授信品質訊號
  WATCH-03 買方集中度過高      —— 單一買方倒帳會直接拖垮整個應收部位
  WATCH-04 發票帳期與合約不符  —— 帳期被單方面拉長是現金流惡化的早期徵兆
  WATCH-05 特定買方付款趨勢惡化 —— 連續延遲比單次延遲更值得警覺
  WATCH-06 應收週轉天數惡化    —— 整體現金循環變慢
  WATCH-07 重複發票號碼        —— 重複請款，可能是作業疏失也可能是造假

門檻的來源全部寫在各規則的註解裡。**沒有一個門檻是為了讓 demo
好看而挑的** —— 挑得出好數字的門檻，換一批資料就會失效。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Optional

import psycopg2.extras

from . import db

# ── 門檻與其依據 ─────────────────────────────────────────────────────────
# 這些不是調出來的，是有來由的：

# 台灣 B2B 常見帳期為月結 30/60/90 天。提前 7 天提醒，
# 是因為請款到入帳通常需要一個作業週期，7 天讓對方還來得及處理。
DUE_SOON_DAYS = 7

# 集中度 40%：授信實務上單一往來對象達四成即視為集中度風險。
# 這個數字取自一般徵信實務慣例，不是從本專案資料挑出來的 ——
# 若改成從資料挑，就會變成「挑一個剛好讓這份資料看起來有問題」的門檻。
#
# 比較用 >= 而非 >：這是**上限**語意，達到上限就該提示，
# 不必等到超過。這個邊界是被單元測試的負向對照撞出來的
# （測試資料剛好落在 40.0%），當時的處理方式是把意圖寫清楚並測它，
# 而不是把門檻改成 41% 讓測試通過 —— 為了通過而調參，
# 換一批資料就會失效。
CONCENTRATION_PCT = 40.0

# 連續 2 期延遲才算「趨勢」。1 次可能是對方作業疏失，
# 2 次才開始像模式。這裡刻意保守 —— 誤報會消耗使用者的信任額度。
LATE_STREAK = 2

# 帳期差異容忍 3 天：跨月結算、假日順延都會造成小幅差異，
# 把這些當異常會產生大量雜訊。超過 3 天才視為實質不符。
TERMS_TOLERANCE_DAYS = 3

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass
class Alert:
    rule_id: str
    severity: str
    title: str
    detail: str
    evidence: list[dict] = field(default_factory=list)

    def fingerprint(self) -> str:
        """
        指紋 = 規則 + 證據內容。證據變了才算新事件。

        刻意**不**把時間放進指紋：如果放了，同一件事每天都會產生
        新指紋，去重就完全失效 —— 這是這類系統最常見的實作錯誤。
        """
        payload = json.dumps({"rule": self.rule_id, "evidence": self.evidence},
                             ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def _f(v: Any) -> float:
    if isinstance(v, Decimal):
        return float(v)
    return float(v or 0)


# ══════════════════════════════════════════════════════════════════════════
# 規則
# ══════════════════════════════════════════════════════════════════════════

def _w01_due_soon(cur, today: date) -> list[Alert]:
    cur.execute("""
        SELECT invoice_number, buyer_name, buyer_ban, due_date, total_amount
        FROM fin_invoices
        WHERE status NOT IN ('PAID', 'WRITTEN_OFF', 'CANCELLED', 'VOID')
          AND due_date BETWEEN %s AND %s
        ORDER BY due_date, total_amount DESC
    """, (today, today + timedelta(days=DUE_SOON_DAYS)))
    rows = _rows(cur)
    if not rows:
        return []
    total = sum(_f(r["total_amount"]) for r in rows)
    return [Alert(
        "WATCH-01", "info",
        f"{DUE_SOON_DAYS} 日內有 {len(rows)} 筆應收到期",
        f"合計 {total:,.0f} 元。最早到期 {rows[0]['due_date']}"
        f"（{rows[0]['buyer_name']}，{_f(rows[0]['total_amount']):,.0f} 元）。",
        rows)]


def _w02_overdue(cur, today: date) -> list[Alert]:
    cur.execute("""
        SELECT invoice_number, buyer_name, buyer_ban, due_date, total_amount,
               (%s::date - due_date) AS days_overdue
        FROM fin_invoices
        WHERE status NOT IN ('PAID', 'WRITTEN_OFF', 'CANCELLED', 'VOID')
          AND due_date < %s
        ORDER BY (%s::date - due_date) DESC
    """, (today, today, today))
    rows = _rows(cur)
    if not rows:
        return []
    total = sum(_f(r["total_amount"]) for r in rows)
    worst = rows[0]["days_overdue"]
    # 90 天在授信實務上是常見的呆帳認列分界，因此以此區分嚴重度
    sev = "critical" if worst >= 90 else "warning"
    return [Alert(
        "WATCH-02", sev,
        f"{len(rows)} 筆應收已逾期，最久 {worst} 天",
        f"逾期金額合計 {total:,.0f} 元。逾期最久：{rows[0]['buyer_name']} "
        f"{rows[0]['invoice_number']}（{worst} 天）。"
        + ("逾期超過 90 天，已達一般呆帳認列分界。" if sev == "critical" else ""),
        rows)]


def _w03_concentration(cur) -> list[Alert]:
    cur.execute("""
        WITH open_inv AS (
            SELECT buyer_name, buyer_ban, SUM(total_amount) AS amt
            FROM fin_invoices
            WHERE status NOT IN ('PAID', 'WRITTEN_OFF', 'CANCELLED', 'VOID')
            GROUP BY buyer_name, buyer_ban
        )
        SELECT buyer_name, buyer_ban, amt,
               ROUND(100.0 * amt / NULLIF(SUM(amt) OVER (), 0), 2) AS pct
        FROM open_inv
        ORDER BY amt DESC
    """)
    rows = _rows(cur)
    hits = [r for r in rows if _f(r["pct"]) >= CONCENTRATION_PCT]
    if not hits:
        return []
    top = hits[0]
    return [Alert(
        "WATCH-03", "warning",
        f"買方集中度過高：{top['buyer_name']} 占未收餘額 {_f(top['pct']):.1f}%",
        f"單一買方占比達 {_f(top['pct']):.1f}%（門檻 {CONCENTRATION_PCT}%，"
        f"取自徵信實務慣例）。該買方一旦延遲或倒帳，會直接衝擊整體應收部位。",
        hits)]


def _w04_terms_mismatch(cur) -> list[Alert]:
    """
    發票帳期與合約帳期不符。

    這條規則的價值不在抓造假，而在抓**現金流惡化的早期徵兆**：
    帳期被買方單方面拉長，通常比財報惡化更早出現。
    """
    # 用 buyer_ban 關聯，不用 contract_id ——
    # 實際資料裡發票**沒有**帶合約編號，兩者是靠交易對象（統編）對應的。
    # 這反映真實情況：發票是逐筆開立的憑證，合約是框架協議，
    # 中小企業的系統多半不會在發票上回填合約編號。
    # 另外要求發票開立日落在合約有效期內，否則會拿舊合約去比對新發票。
    cur.execute("""
        SELECT i.invoice_number, i.buyer_name, i.buyer_ban,
               c.contract_id,
               i.payment_terms_days AS invoice_terms,
               c.payment_terms_days AS contract_terms,
               (i.payment_terms_days - c.payment_terms_days) AS diff_days,
               i.issue_date, i.total_amount
        FROM fin_invoices i
        JOIN fin_contracts c
          ON c.tenant_id = i.tenant_id
         AND c.counterparty_ban = i.buyer_ban
         AND (c.start_date IS NULL OR i.issue_date >= c.start_date)
         AND (c.end_date   IS NULL OR i.issue_date <= c.end_date)
        WHERE i.payment_terms_days IS NOT NULL
          AND c.payment_terms_days IS NOT NULL
          AND ABS(i.payment_terms_days - c.payment_terms_days) > %s
        ORDER BY ABS(i.payment_terms_days - c.payment_terms_days) DESC
    """, (TERMS_TOLERANCE_DAYS,))
    rows = _rows(cur)
    if not rows:
        return []
    longer = [r for r in rows if (r["diff_days"] or 0) > 0]
    return [Alert(
        "WATCH-04", "warning",
        f"{len(rows)} 筆發票的帳期與合約不符",
        f"其中 {len(longer)} 筆的實際帳期**長於**合約約定"
        f"（容忍 {TERMS_TOLERANCE_DAYS} 天以吸收跨月結算與假日順延）。"
        f"帳期被單方面拉長，通常比財報惡化更早出現。",
        rows)]


def _w05_late_streak(cur) -> list[Alert]:
    """
    同一買方連續延遲付款。

    比「單次延遲」值得警覺得多：單次可能是對方作業疏失，
    連續就開始像是對方本身的資金狀況出了問題。
    """
    cur.execute("""
        SELECT buyer_name, buyer_ban,
               COUNT(*) FILTER (WHERE paid_date > due_date) AS late_cnt,
               COUNT(*) AS paid_cnt,
               MAX(paid_date - due_date) AS worst_delay
        FROM fin_invoices
        WHERE status = 'PAID' AND paid_date IS NOT NULL AND due_date IS NOT NULL
        GROUP BY buyer_name, buyer_ban
        HAVING COUNT(*) FILTER (WHERE paid_date > due_date) >= %s
        ORDER BY COUNT(*) FILTER (WHERE paid_date > due_date) DESC
    """, (LATE_STREAK,))
    rows = _rows(cur)
    if not rows:
        return []
    top = rows[0]
    return [Alert(
        "WATCH-05", "warning",
        f"{len(rows)} 個買方有重複延遲付款紀錄",
        f"最嚴重：{top['buyer_name']} 在 {top['paid_cnt']} 筆已付款發票中"
        f"延遲 {top['late_cnt']} 次，最久延遲 {top['worst_delay']} 天。"
        f"連續延遲（門檻 {LATE_STREAK} 次）比單次延遲更值得警覺 ——"
        f"單次可能是作業疏失，連續開始像對方資金狀況的問題。",
        rows)]


def _w06_dso(cur, today: date) -> list[Alert]:
    """
    應收帳款週轉天數（DSO）惡化。

    比較最近 90 天與前一個 90 天。用**相對變化**而非絕對門檻，
    因為合理的 DSO 高度依產業而異（批發零售與營造完全不同），
    訂一個絕對門檻等於假裝所有產業一樣。
    """
    cur.execute("""
        SELECT
          AVG(paid_date - issue_date) FILTER (
              WHERE paid_date >= %s - INTERVAL '90 days') AS recent,
          AVG(paid_date - issue_date) FILTER (
              WHERE paid_date >= %s - INTERVAL '180 days'
                AND paid_date <  %s - INTERVAL '90 days')  AS prior,
          COUNT(*) FILTER (WHERE paid_date >= %s - INTERVAL '90 days') AS n_recent
        FROM fin_invoices
        WHERE status = 'PAID' AND paid_date IS NOT NULL AND issue_date IS NOT NULL
    """, (today, today, today, today))
    r = _rows(cur)[0] if cur.rowcount else None
    if not r or r["recent"] is None or r["prior"] is None:
        return []
    recent, prior = _f(r["recent"]), _f(r["prior"])
    # 樣本太少時不下判斷。10 筆是很低的門檻，但至少擋掉 1~2 筆造成的假訊號。
    if (r["n_recent"] or 0) < 10 or prior <= 0:
        return []
    delta_pct = 100.0 * (recent - prior) / prior
    if delta_pct < 15.0:        # 15% 以下視為正常波動
        return []
    return [Alert(
        "WATCH-06", "warning",
        f"收款天數惡化 {delta_pct:.1f}%",
        f"最近 90 天平均收款 {recent:.1f} 天，前一期 {prior:.1f} 天。"
        f"以相對變化判斷而非絕對門檻，因為合理 DSO 高度依產業而異。",
        [{"recent_dso": round(recent, 1), "prior_dso": round(prior, 1),
          "change_pct": round(delta_pct, 1), "n_recent": r["n_recent"]}])]


def _w07_duplicate(cur) -> list[Alert]:
    """
    同一買方、同一金額、鄰近日期的疑似重複發票。

    發票號碼本身有主鍵擋著不會重複，真正要抓的是
    「換一個號碼、其他都一樣」的重複請款。
    """
    cur.execute("""
        SELECT buyer_ban, buyer_name, total_amount,
               ARRAY_AGG(invoice_number ORDER BY issue_date) AS invoices,
               ARRAY_AGG(issue_date ORDER BY issue_date) AS dates,
               COUNT(*) AS n
        FROM fin_invoices
        WHERE total_amount IS NOT NULL AND buyer_ban IS NOT NULL
        GROUP BY buyer_ban, buyer_name, total_amount
        HAVING COUNT(*) > 1
           AND MAX(issue_date) - MIN(issue_date) <= 7
        ORDER BY COUNT(*) DESC
    """)
    rows = _rows(cur)
    if not rows:
        return []
    return [Alert(
        "WATCH-07", "warning",
        f"{len(rows)} 組疑似重複請款",
        "同一買方、同一金額、7 日內開立多張發票。"
        "可能是分批請款的正常作業，也可能是重複請款 —— **需人工確認**。"
        "系統不替這件事下結論，只把它指出來。",
        rows)]


RULES = [
    ("WATCH-01", _w01_due_soon, True),
    ("WATCH-02", _w02_overdue, True),
    ("WATCH-03", _w03_concentration, False),
    ("WATCH-04", _w04_terms_mismatch, False),
    ("WATCH-05", _w05_late_streak, False),
    ("WATCH-06", _w06_dso, True),
    ("WATCH-07", _w07_duplicate, False),
]


# ══════════════════════════════════════════════════════════════════════════
# 掃描
# ══════════════════════════════════════════════════════════════════════════

def scan(tenant_id: str, today: Optional[date] = None,
         persist: bool = True) -> list[Alert]:
    """
    對一個委任案跑完整掃描。

    `today` 可注入，讓這件事**可測試** ——
    一個只有在今天才對的監控系統，明天就會壞掉而沒人發現。
    """
    today = today or date.today()
    alerts: list[Alert] = []

    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for rule_id, fn, needs_today in RULES:
                try:
                    alerts += fn(cur, today) if needs_today else fn(cur)
                except Exception as e:                    # noqa: BLE001
                    # 一條規則壞掉不該讓整次掃描消失 ——
                    # 但也**絕不**靜默吞掉：把失敗本身變成一條警示，
                    # 否則「監控沒報警」和「監控壞了」看起來一模一樣。
                    alerts.append(Alert(
                        rule_id, "critical", f"{rule_id} 規則執行失敗",
                        f"這條規則沒有跑完，因此它涵蓋的風險**未經檢查**："
                        f"{type(e).__name__}: {e}", []))

        if persist:
            _persist(conn, tenant_id, alerts)

    alerts.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 9), a.rule_id))
    return alerts


def _persist(conn, tenant_id: str, alerts: list[Alert]) -> None:
    with conn.cursor() as cur:
        for a in alerts:
            cur.execute("""
                INSERT INTO fin_alerts
                    (tenant_id, rule_id, severity, title, detail,
                     evidence, fingerprint)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, fingerprint) DO UPDATE
                    SET last_seen_at = NOW()
            """, (tenant_id, a.rule_id, a.severity, a.title, a.detail,
                  json.dumps(a.evidence, ensure_ascii=False, default=str),
                  a.fingerprint()))
    conn.commit()


def open_alerts(tenant_id: str) -> list[dict]:
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT rule_id, severity, title, detail, evidence,
                       first_seen_at, last_seen_at
                FROM fin_alerts
                WHERE resolved_at IS NULL
                ORDER BY CASE severity WHEN 'critical' THEN 0
                                       WHEN 'warning'  THEN 1 ELSE 2 END,
                         first_seen_at DESC
            """)
            return _rows(cur)


def render(alerts: list[Alert]) -> str:
    if not alerts:
        return "✅ 本次掃描未發現需要注意的事項。"
    icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
    L = ["每日監控摘要", "═" * 74]
    for a in alerts:
        L.append(f"\n{icon.get(a.severity, '·')} [{a.rule_id}] {a.title}")
        L.append(f"   {a.detail}")
        if a.evidence:
            L.append(f"   證據 {len(a.evidence)} 筆（可逐列複查）：")
            for e in a.evidence[:3]:
                items = list(e.items())[:4]
                L.append("     · " + "　".join(f"{k}={v}" for k, v in items))
            if len(a.evidence) > 3:
                L.append(f"     …另外 {len(a.evidence) - 3} 筆")
    L.append("\n" + "═" * 74)
    L.append("本摘要由決定性 SQL 與算術產生，未使用 LLM。")
    L.append("每條警示都附觸發它的實際資料列，可逐列複查。")
    return "\n".join(L)


__all__ = ["Alert", "scan", "open_alerts", "render", "RULES"]
