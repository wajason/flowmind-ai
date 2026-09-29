"""
flowmind.metrics — 決定性指標與問題路由
=============================================================================
【這支檔案來自一次真實的失敗，值得記錄下來】

實測時問系統：「本案最大買方占營收多少？逾期狀況如何？」
系統檢索到 6 個 chunk（90 張發票裡的 3 張），模型據此寫出四句看起來很專業的結論，
四句全部沒有通過逐字驗證，信心掉到 0.31，系統拒答。

拒答是對的 —— 但**問題不在模型，在架構**。
「最大買方占營收多少」需要把 90 張發票全部加總後相除。
RAG 是「取回最相關的幾段文字」，它在設計上就不可能可靠地回答彙總問題。
給模型 6 張發票要它算出 90 張的占比，只有兩種結果：拒答，或編一個數字。

而這個數字其實 `crosscheck.py` 早就算出來了，精確到小數點，零誤差。

所以正確的架構不是把 RAG 調得更好，而是**先判斷這題該不該給 RAG**：

    可以用算的      → 用算的（純 Python，精確、可重算、零幻覺）
    需要理解文義    → 才交給 RAG（法規怎麼規定、商品有什麼差別）

路由本身也刻意用關鍵詞規則而不是 LLM 分類器。
理由是可預測性：使用者問同一句話，永遠走同一條路徑。
一個時好時壞的路由，比沒有路由更難除錯。
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from . import config, crosscheck


# ══════════════════════════════════════════════════════════════════════════
# 讀取委任案的原始憑證
# ══════════════════════════════════════════════════════════════════════════

def load_engagement_files(tenant_id: str) -> dict[str, Any]:
    base = config.RAW_DIR / tenant_id
    out: dict[str, Any] = {"invoices": [], "contracts": [], "payables": [],
                           "ledger": [], "projection": None, "base": base}
    if not base.exists():
        return out

    def jl(name: str) -> list:
        p = base / name
        if not p.exists():
            return []
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else [d]

    out["invoices"] = jl("receivables.json")
    out["contracts"] = jl("contracts.json")
    out["payables"] = jl("payables.json")

    lp = base / "bank_ledger.csv"
    if lp.exists():
        with lp.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    r["amount"] = float(r.get("amount", 0) or 0)
                except ValueError:
                    r["amount"] = 0.0
                out["ledger"].append(r)

    pp = base / "cash_flow_projection.json"
    if pp.exists():
        out["projection"] = json.loads(pp.read_text(encoding="utf-8"))
    return out


def _d(s: Any) -> Optional[date]:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:                                  # noqa: BLE001
        return None


CLOSED = {"PAID", "WRITTEN_OFF", "CANCELLED", "VOID"}

# 原始檔名 → 畫面、答案、報告上給人看的名稱。統一從這裡取，不在各處各寫一份。
SOURCE_NAMES = {
    "receivables.json": "應收帳款發票",
    "contracts.json": "買賣合約",
    "payables.json": "應付帳款",
    "bank_ledger.csv": "銀行流水",
}


def source_label(name: str) -> str:
    return SOURCE_NAMES.get(name, name)


# ══════════════════════════════════════════════════════════════════════════
# 現金部位推估
# 儀表板的現金流時間軸、情境模擬、授信報告都呼叫 cash_projection()，
# 三處的期初餘額、推估餘額與最低點因此必然一致。
# ══════════════════════════════════════════════════════════════════════════

CASH_HORIZON_DAYS = 90


def _num(v: Any) -> Optional[float]:
    """數值欄位。CSV 讀進來是字串、JSON 讀進來是數字，兩種都要吃。"""
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def latest_balance(ledger: list[dict]) -> tuple[float, Optional[date]]:
    """銀行流水最後一筆有餘額的紀錄：(餘額, 日期)。"""
    for row in reversed(ledger):
        b = _num(row.get("balance"))
        if b is not None:
            return b, _d(row.get("date"))
    return 0.0, None


def recurring_outflows(ledger: list[dict], min_months: int = 3) -> list[dict]:
    """
    每月固定支出：摘要相同、至少出現在 min_months 個不同月份、
    且沒有對應任何憑證（reference 空白）的扣款。金額與扣款日取最近一期。

    有對應憑證的扣款（例如支付供應商）已經由應付帳款涵蓋，不重複計入。
    """
    groups: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for row in ledger:
        amt, d = _num(row.get("amount")), _d(row.get("date"))
        if amt is None or amt >= 0 or d is None or str(row.get("reference") or "").strip():
            continue
        groups[str(row.get("description") or "").strip()].append((d, -amt))
    out = []
    for desc, rows in groups.items():
        if len({(d.year, d.month) for d, _ in rows}) >= min_months:
            last_d, last_amt = max(rows)
            out.append({"description": desc, "amount": round(last_amt), "day": last_d.day})
    return out


def cash_projection(data: dict, as_of: Optional[date] = None,
                    horizon_days: int = CASH_HORIZON_DAYS,
                    extra_outflows: Optional[list[dict]] = None) -> dict:
    """
    未來 horizon_days 天的現金部位，逐筆事件累加。

    計入
      期初：銀行流水最新一筆餘額
      流入：尚未收款、到期日落在推估期間內的應收帳款
      流出：尚未付款的應付帳款（已過到期日仍未付者，視為基準日當天支付）、
            每月固定支出（見 recurring_outflows）
    不計入
      已逾期的應收帳款：收不收得回來不確定，假設它會入帳會高估償債能力，
      改列在 overdue_receivables 另行揭露。
    同一天有收有付時先扣再加，取保守的一邊。
    """
    import calendar                                         # noqa: PLC0415
    from datetime import timedelta                          # noqa: PLC0415

    as_of = as_of or date.today()
    end = as_of + timedelta(days=horizon_days)
    ledger = data.get("ledger") or []
    opening, ledger_date = latest_balance(ledger)
    events: list[tuple[date, float, str, str, str]] = []

    overdue_n, overdue_amt = 0, 0.0
    for inv in data.get("invoices") or []:
        if str(inv.get("status", "")).upper() in CLOSED:
            continue
        due, amt = _d(inv.get("due_date")), _num(inv.get("total_amount")) or 0.0
        if due is None:
            continue
        if due < as_of:
            overdue_n += 1
            overdue_amt += amt
        elif due <= end:
            events.append((due, amt, "inflow", inv.get("buyer_name") or "",
                           inv.get("invoice_number") or ""))

    for p in data.get("payables") or []:
        if str(p.get("status", "")).upper() in CLOSED:
            continue
        due, amt = _d(p.get("due_date")), _num(p.get("amount")) or 0.0
        if due is None or due > end:
            continue
        events.append((max(due, as_of), -amt, "outflow", p.get("supplier_name") or "",
                       p.get("bill_number") or ""))

    fixed = recurring_outflows(ledger)
    for f in fixed:
        y, m = as_of.year, as_of.month
        while True:
            d = date(y, m, min(f["day"], calendar.monthrange(y, m)[1]))
            if d > end:
                break
            if d >= as_of:
                events.append((d, -float(f["amount"]), "fixed", f["description"], ""))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    for x in extra_outflows or []:
        events.append((x["date"], -float(x["amount"]), "scenario",
                       x.get("label", "新增應付款"), ""))

    events.sort(key=lambda e: (e[0], e[1] > 0))
    bal, timeline = opening, []
    for d, amt, kind, label, ref in events:
        bal += amt
        timeline.append({"date": d.isoformat(), "amount": round(amt), "type": kind,
                         "counterparty": label, "reference": ref, "balance": round(bal)})

    trough, trough_date = opening, as_of.isoformat()
    for e in timeline:
        if e["balance"] < trough:
            trough, trough_date = e["balance"], e["date"]
    gap = next((e for e in timeline if e["balance"] < 0), None)
    return {
        "as_of": as_of.isoformat(), "horizon_end": end.isoformat(),
        "horizon_days": horizon_days,
        "opening_balance": round(opening),
        "ledger_date": ledger_date.isoformat() if ledger_date else None,
        "fixed_costs": fixed,
        "timeline": timeline,
        "trough_balance": round(trough), "trough_date": trough_date,
        "gap_detected": gap is not None,
        "gap_date": gap["date"] if gap else None,
        "gap_amount": gap["balance"] if gap else None,
        "overdue_receivables": {"count": overdue_n, "total": round(overdue_amt)},
    }


# ══════════════════════════════════════════════════════════════════════════
# 指標計算（每一項都是純算術，可由第三方以相同規則重算）
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Metric:
    key: str
    title: str
    text: str                    # 給人看的完整敘述
    value: Any                   # 機器可讀的數值
    method: str                  # 計算方式，寫給要驗算的人看
    sources: list[str]


def m_concentration(data: dict) -> Optional[Metric]:
    inv = data["invoices"]
    if not inv:
        return None
    by: dict[str, float] = defaultdict(float)
    for i in inv:
        by[i.get("buyer_name") or i.get("buyer_ban") or "未知"] += float(i.get("total_amount", 0))
    total = sum(by.values()) or 1.0
    ranked = sorted(by.items(), key=lambda kv: -kv[1])
    top_n = ranked[:5]
    lines = [f"  {n+1}. {name}：NT${amt:,.0f}（{amt/total:.1%}）"
             for n, (name, amt) in enumerate(top_n)]
    share = ranked[0][1] / total
    judgement = ("集中度在一般可接受範圍（單一買方 < 50%）。"
                 if share < 0.5 else
                 "集中度偏高，銀行通常會要求該買方的信用評等或加保，建議事前準備。")
    return Metric(
        key="concentration",
        title="買方集中度",
        text=(f"本案累計開票 {len(inv)} 張，總額 NT${total:,.0f}，"
              f"買方共 {len(by)} 家。\n最大買方「{ranked[0][0]}」占 {share:.1%}。\n\n"
              f"前五大買方：\n" + "\n".join(lines) + f"\n\n{judgement}"),
        value={"top_buyer": ranked[0][0], "top_share": round(share, 4),
               "total_billed": total, "buyer_count": len(by),
               "top5": [{"name": n, "amount": a, "share": round(a / total, 4)}
                        for n, a in top_n]},
        method="該買方所有發票總額 ÷ 全部發票總額",
        sources=["receivables.json"])


def m_ageing(data: dict) -> Optional[Metric]:
    inv = data["invoices"]
    if not inv:
        return None
    today = date.today()
    buckets = {"未到期": 0.0, "逾期 1-30 天": 0.0, "逾期 31-60 天": 0.0,
               "逾期 61-90 天": 0.0, "逾期 90 天以上": 0.0}
    counts = dict.fromkeys(buckets, 0)
    open_total = 0.0
    for i in inv:
        if str(i.get("status", "")).upper() in CLOSED:
            continue
        amt = float(i.get("total_amount", 0))
        open_total += amt
        due = _d(i.get("due_date"))
        days = (today - due).days if due else 0
        b = ("未到期" if days <= 0 else "逾期 1-30 天" if days <= 30 else
             "逾期 31-60 天" if days <= 60 else "逾期 61-90 天" if days <= 90 else
             "逾期 90 天以上")
        buckets[b] += amt
        counts[b] += 1
    overdue = open_total - buckets["未到期"]
    wo = [i for i in inv if str(i.get("status", "")).upper() == "WRITTEN_OFF"]
    wo_amt = sum(float(i.get("total_amount", 0)) for i in wo)
    billed = sum(float(i.get("total_amount", 0)) for i in inv) or 1.0

    lines = [f"  {b}：{counts[b]} 張　NT${buckets[b]:,.0f}"
             f"（{buckets[b]/(open_total or 1):.1%}）" for b in buckets]
    return Metric(
        key="ageing",
        title="帳齡分析與逾期狀況",
        text=(f"未收帳款 NT${open_total:,.0f}，帳齡分布：\n" + "\n".join(lines) +
              f"\n\n逾期合計 NT${overdue:,.0f}，占未收帳款 "
              f"{overdue/(open_total or 1):.1%}。\n"
              f"另有呆帳沖銷 {len(wo)} 張、NT${wo_amt:,.0f}，"
              f"占累計開票 {wo_amt/billed:.2%}。\n\n"
              f"註：逾期（還沒收到）與呆帳（已認定收不到）分開計算，"
              f"兩者在授信上的意義不同，混在一起會讓正常公司看起來像要倒了。"),
        value={"open_total": open_total, "overdue_total": overdue,
               "overdue_ratio": round(overdue / (open_total or 1), 4),
               "written_off_total": wo_amt,
               "written_off_ratio": round(wo_amt / billed, 4),
               "buckets": {b: {"amount": buckets[b], "count": counts[b]} for b in buckets}},
        method="依到期日與基準日相差的天數分組；已收款與已沖銷的發票不列入未收帳款",
        sources=["receivables.json"])


def m_cashflow(data: dict) -> Optional[Metric]:
    if not (data.get("ledger") or data.get("invoices")):
        return None
    p = cash_projection(data)
    od = p["overdue_receivables"]
    if p["gap_detected"]:
        days = (_d(p["gap_date"]) - _d(p["as_of"])).days
        head = (f"⚠ 預估 {p['gap_date']}（{days} 天後）出現現金缺口，"
                f"金額約 NT${abs(p['gap_amount']):,.0f}。")
        advice = ("建議在缺口日前完成融資動撥。以本案的應收帳款結構，"
                  "應收帳款承購或信保供應商融資是常見的解法；"
                  "適用條件請以法規與各銀行商品說明為準。")
    else:
        head = (f"未來 {p['horizon_days']} 天內未偵測到現金缺口，"
                f"最低推估餘額 NT${p['trough_balance']:,.0f}（{p['trough_date']}）。")
        advice = "目前現金部位可支應已知的應付款與固定支出，無立即融資需求。"
    overdue = (f"逾期應收 {od['count']} 筆、NT${od['total']:,.0f} **未**計入推估："
               f"收不收得回來不確定，假設它會準時入帳會高估償債能力。\n\n"
               if od["count"] else "")
    return Metric(
        key="cashflow",
        title="現金流缺口預測",
        text=(f"目前銀行餘額 NT${p['opening_balance']:,.0f}"
              f"（銀行流水截至 {p['ledger_date']}）。\n{head}\n\n{overdue}{advice}"),
        value={k: p[k] for k in ("opening_balance", "gap_detected", "gap_date",
                                 "gap_amount", "horizon_days", "trough_balance",
                                 "trough_date", "overdue_receivables")},
        method="期初取最新銀行餘額，依未到期應收、未付應付與每月固定支出的日期逐筆累加；"
               "逾期應收不計入",
        sources=["receivables.json", "payables.json", "bank_ledger.csv"])


# 檢查編號前綴 → 使用者會怎麼稱呼它。
# 這張表存在的理由是一個真實的失敗：使用者在儀表板上看到
# 「跨文件勾稽 3/4 通過」，接著問「跨文件勾稽有一個沒通過是什麼地方」，
# 系統卻**跑去知識庫查法規**，當然找不到 —— 答案就在同一頁上。
# 問答層與 crosscheck 引擎是兩條沒接上的線，而接上它們只需要路由認得這些詞。
CHECK_ALIASES: dict[str, list[str]] = {
    "TAXID": ["統編", "統一編號", "檢核碼", "憑證真偽"],
    "FRAUD": ["自我交易", "造假", "憑證真偽"],
    "ARITH": ["金額", "加總", "算術", "稅率", "稅額"],
    "AMT": ["金額", "算術"],
    "DUP": ["重複", "重複請款", "發票號碼"],
    "SEQ": ["連號", "重複請款"],
    "TERM": ["帳期", "到期日", "跨文件", "勾稽", "合約"],
    "CONTRACT": ["合約", "跨文件", "勾稽"],
    "BANK": ["銀行", "流水", "對帳", "勾稽", "跨文件"],
    "LEDGER": ["流水", "勾稽", "跨文件"],
    "RELATED": ["關係人", "跨文件"],
    "DATE": ["日期", "時序", "鑑識"],
    "FORENSIC": ["班佛", "鑑識", "整數", "假日"],
    "RISK": ["集中度", "逾期", "呆帳", "授信風險"],
}


def m_integrity(data: dict, question: str = "") -> Optional[Metric]:
    """
    憑證交叉驗證。

    帶 `question` 是為了回答「**哪一項**沒通過」這種追問 ——
    使用者在儀表板看到某個分類 3/4 通過，下一句必然是問那一項是什麼。
    若只能回「整體有 3 項未通過」，等於要他自己再去畫面上找。
    """
    inv = data["invoices"]
    if not inv:
        return None
    rep = crosscheck.run_all(inv, data["contracts"], data["ledger"])
    failed = [f for f in rep["findings"] if not f["passed"]]

    # 問題若指向特定檢查（編號或分類名），把那幾項提到最前面並展開
    q = re.sub(r"\s+", "", question)
    focus_ids = {cid for cid in CHECK_ALIASES if cid.lower() in q.lower()}
    for cid, words in CHECK_ALIASES.items():
        if any(w in q for w in words):
            focus_ids.add(cid)
    focused = [f for f in rep["findings"]
               if any(f["check_id"].startswith(c) for c in focus_ids)] \
        if focus_ids else []

    def _line(f: dict) -> str:
        icon = "✅" if f["passed"] else ("🔴" if f["severity"] == "critical" else "🟡")
        return f"  {icon} [{f['check_id']}] {f['title']}：{f['detail']}"

    parts = [f"共執行 {len(rep['findings'])} 項交叉查核，"
             f"完整性分數 {rep['integrity_score']:.1%}，"
             f"重大缺失 {rep['critical_failures']} 項。",
             f"送件建議：{'✅ 可送件' if rep['submission_ready'] else '⛔ 建議先補正重大缺失'}"]

    if focused:
        parts.append("\n【您問的項目】")
        parts += [_line(f) for f in focused]

    other = [f for f in failed if f not in focused]
    if other:
        parts.append("\n【其他未通過項目】" if focused else "\n未通過項目：")
        parts += [_line(f) for f in other]
    elif not focused and not failed:
        parts.append("\n所有檢查項目皆通過。")

    return Metric(
        key="integrity",
        title="憑證交叉驗證",
        text="\n".join(parts),
        value=rep,
        method="每一項皆為純算術判定，可由第三方以相同規則重算",
        sources=["receivables.json", "contracts.json", "bank_ledger.csv"])


def m_summary(data: dict) -> Optional[Metric]:
    inv = data["invoices"]
    if not inv:
        return None
    total = sum(float(i.get("total_amount", 0)) for i in inv)
    open_inv = [i for i in inv if str(i.get("status", "")).upper() not in CLOSED]
    open_total = sum(float(i.get("total_amount", 0)) for i in open_inv)
    terms = [int(i.get("payment_terms_days", 0)) for i in inv if i.get("payment_terms_days")]
    avg_term = sum(terms) / len(terms) if terms else 0
    dates = sorted(d for d in (_d(i.get("invoice_date")) for i in inv) if d)
    return Metric(
        key="summary",
        title="應收帳款總覽",
        text=(f"資料期間 {dates[0]} 至 {dates[-1]}（{len(dates)} 張發票）。\n"
              f"累計開票 NT${total:,.0f}，未收 NT${open_total:,.0f}"
              f"（{len(open_inv)} 張）。\n"
              f"加權平均約定帳期 {avg_term:.0f} 天。\n"
              f"合約 {len(data['contracts'])} 份、銀行流水 {len(data['ledger'])} 筆。"),
        value={"billed_total": total, "open_total": open_total,
               "invoice_count": len(inv), "open_count": len(open_inv),
               "avg_terms_days": round(avg_term, 1),
               "period": [str(dates[0]), str(dates[-1])] if dates else None},
        method="彙總全部應收帳款發票",
        sources=["receivables.json"])


def m_statistics(data: dict, question: str = "") -> Optional[Metric]:
    """
    從公開統計表取出精確數字。

    這一項與其他指標不同：它查的是 SHARED 公開統計，不是這個委任案的憑證。
    但同樣的原則適用 —— 有原始檔案可以查的數字，就不該讓語言模型從摘要裡推估。
    """
    from . import tables
    terms = tables.match_question(question)
    if not terms:
        return None
    all_hits, used = [], []
    for t in terms:
        hits = tables.lookup(t, limit=8)
        if hits:
            all_hits.extend(hits)
            used.append(t)
    if not all_hits:
        return None

    # 依查詢詞分組呈現
    parts = []
    for t in used:
        hs = [h for h in all_hits if t in h.row_label]
        if hs:
            parts.append(tables.render_hits(hs, t))
    return Metric(
        key="statistics",
        title="公開統計表精確查詢",
        text="\n\n".join(parts),
        value=[{"source": h.source, "row": h.row_label, "columns": h.columns,
                "period": h.period, "unit": h.unit} for h in all_hits],
        method="直接從官方統計原始檔讀取指定列，未經語言模型處理",
        sources=sorted({h.source for h in all_hits}))


def m_industry(data: dict, question: str = "") -> Optional[Metric]:
    """
    產業側寫：從 29 份真實官方統計推導，零 LLM。

    為什麼這條要走決定性路由而不是走 RAG：
    「製造業的出口依存度是多少」有一個**唯一正確的數字**，
    它躺在一份 CSV 的某一列裡。讓 LLM 從摘要文字裡找這個數字，
    是把一個確定的問題變成一個機率問題 —— 沒有任何好處。
    """
    from . import industry                              # noqa: PLC0415

    try:
        series = industry.load_series()
    except (FileNotFoundError, ValueError) as e:
        return Metric("industry", "產業側寫", f"[產業統計無法載入：{e}]",
                      None, "-", [])

    q = re.sub(r"\s+", "", question)
    # 長名稱優先，否則「製造業」會先吃掉「金屬製品製造業」
    hits = [i for i in sorted(industry.industries(series), key=len, reverse=True)
            if i in q]
    if not hits:
        return None

    parts, srcs = [], set()
    for ind in hits[:3]:
        try:
            p = industry.profile(ind, series=series)
        except KeyError:
            continue
        parts.append(p.render())
        srcs |= {v[0] for v in industry.SOURCES.values()}
    if not parts:
        return None
    if len(hits) > 1:
        parts.append(industry.compare(hits[:4]))

    return Metric(
        key="industry",
        title="產業側寫（推導自官方統計）",
        text="\n\n".join(parts),
        value=[{"industry": h} for h in hits[:3]],
        method="直接讀取中小企業處統計 CSV 並做四則運算；"
               "統計事實與授信判讀分開標示，未經語言模型處理",
        sources=sorted(srcs))


METRIC_FNS = {"concentration": m_concentration, "ageing": m_ageing,
              "cashflow": m_cashflow, "integrity": m_integrity,
              "summary": m_summary, "statistics": m_statistics,
              "industry": m_industry}


# ══════════════════════════════════════════════════════════════════════════
# 路由：關鍵詞規則，刻意不用 LLM 分類器
# ══════════════════════════════════════════════════════════════════════════

ROUTES: list[tuple[str, list[str]]] = [
    ("concentration", ["集中度", "最大買方", "主要客戶", "客戶占比", "占營收",
                       "佔營收", "大客戶", "買方分布", "客戶結構"]),
    ("ageing",        ["帳齡", "逾期", "呆帳", "催收", "未收", "多久沒收",
                       "收款狀況", "壞帳"]),
    ("cashflow",      ["現金流", "現金缺口", "缺口", "夠不夠", "週轉", "資金需求",
                       "會不會缺錢", "何時缺"]),
    # 這一組刻意寫得比其他路由寬。理由是一個真實的失敗：
    # 使用者在儀表板看到「跨文件勾稽 3/4 通過」，接著問
    # 「跨文件勾稽有一個沒通過是什麼地方」，系統卻跑去查法規文件 ——
    # **答案就在同一頁上，只是問答層不認得那些詞。**
    #
    # 寬一點的代價是偶爾把 RAG 題誤判成檢查題；
    # 但那個代價遠小於「畫面上看得到、問卻問不到」給人的印象：
    # 那會讓人覺得整套系統是拼裝的。
    ("integrity",     ["交叉驗證", "驗證", "造假", "可以送件", "能不能送件",
                       "有沒有問題", "憑證", "統編", "重複請款", "自我交易",
                       "送件前", "檢核",
                       # ── 儀表板上看得到的分類名 ──
                       "跨文件", "勾稽", "憑證真偽", "金額算術", "鑑識會計",
                       "授信風險", "檢查結果", "檢查項目",
                       # ── 追問某一項的說法 ──
                       "沒通過", "未通過", "沒過", "失敗的", "哪一項", "哪一條",
                       "哪個地方", "為什麼失敗", "什麼問題", "紅燈", "警示",
                       # ── 直接報檢查編號 ──
                       "TAXID", "ARITH", "FRAUD", "DUP", "TERM", "BANK",
                       "RISK", "CONTRACT", "LEDGER", "FORENSIC", "SEQ"]),
    ("summary",       ["總覽", "應收總額", "開票金額", "多少張發票", "營收多少",
                       "整體狀況", "基本資料"]),
]


# ── 第三方機構名稱：問的是「誰」的數字 ──────────────────────────────────
# 【這是一次真實評測抓到的失敗】105 題評測裡，「信保基金去年的呆帳率是
# 多少？」被「呆帳」關鍵詞路由到 ageing 這個決定性指標——但 ageing
# 算的是**本案自己**帳上的帳齡與呆帳沖銷比率，跟信保基金這個機構自己的
# 呆帳率是兩件事。系統把本案 0.52% 的自家數字，當成信保基金的統計數字
# 端出來，信心還打了 1.00——因為決定性路徑不經過 evidence.py 的信心與
# 拒答閘門，這條路徑原本沒有東西能攔下這種「問錯對象」的情況。
#
# 修法沿用 graph.py 的 PUBLISHER_JURISDICTION（同一份機構名單，不重複維護）：
# 問題裡一旦出現具名的第三方機構，就不走任何「算本案自己帳上數字」的
# 決定性路由——那些指標的計算基礎是本案的 fin_invoices/fin_ledger，
# 從結構上就不可能答得出另一個機構自己的統計。寧可讓它落到 RAG
# （可能拒答、也可能引用到真實統計），也不要讓它套用錯的分母還打滿分信心。
_TENANT_SCOPED_ROUTES = {"concentration", "ageing", "cashflow", "summary"}


def _asks_about_external_entity(q: str) -> bool:
    from . import graph                                    # noqa: PLC0415
    names = set(graph.PUBLISHER_JURISDICTION.keys())
    # 口語簡稱：使用者幾乎不會打全名「玉山商業銀行」，都寫「玉山銀行」——
    # 只比對正式全名會讓這個檢查在真實提問裡形同虛設。
    names |= {"信保基金", "玉山銀行", "中國信託銀行", "中國信託", "永豐銀行"}
    return any(name in q for name in names)


def route(question: str) -> list[str]:
    """回傳這個問題命中的決定性指標清單（可能多個，也可能空）。"""
    q = re.sub(r"\s+", "", question)
    keys = [key for key, kws in ROUTES if any(k in q for k in kws)]

    if _asks_about_external_entity(q):
        keys = [k for k in keys if k not in _TENANT_SCOPED_ROUTES]

    # 統計表查詢：問題裡若出現真實存在於統計表的類別名稱
    # （例如「機械設備製造業」「台北市」「股份有限公司」），
    # 就把精確數字直接從原始檔案讀出來，不要讓 LLM 從摘要裡湊。
    # 這是把入庫摘要那句「完整數據請查原始檔案」真的兌現。
    try:
        from . import tables
        if tables.match_question(question):
            keys.append("statistics")
    except Exception:                                  # noqa: BLE001
        pass

    # 產業側寫：問題提到某個實際存在於統計中的行業別，
    # 且在問這個行業的**特徵**（而不是問本案的某筆交易）。
    # 需要兩個條件同時成立 —— 只憑行業名稱就路由，
    # 會把「我們賣給製造業客戶的那筆帳款」也誤判成產業查詢。
    try:
        from . import industry
        inds = industry.industries()
        if any(i in q for i in inds) and any(
                k in q for k in ["產業", "行業", "同業", "出口依存", "內銷",
                                 "平均規模", "家數", "受僱", "產業特性",
                                 "產業特徵", "產業風險", "比較"]):
            keys.append("industry")
    except Exception:                                  # noqa: BLE001
        pass
    return keys


def _takes_question(fn) -> bool:
    """這個指標函式吃不吃第二個參數（原始問題）。"""
    import inspect                                      # noqa: PLC0415
    try:
        return len(inspect.signature(fn).parameters) >= 2
    except (TypeError, ValueError):
        return False


def compute(tenant_id: str, keys: list[str], question: str = "") -> list[Metric]:
    data = load_engagement_files(tenant_id)
    out = []
    for k in keys:
        fn = METRIC_FNS.get(k)
        if not fn:
            continue
        try:
            # 有些指標需要原始問題才知道要查什麼（statistics 要查哪個類別、
            # industry 要查哪個行業別）。
            #
            # 這裡刻意用**函式簽名**判斷，而不是維護一份「哪些 key 要傳問題」
            # 的清單。原本寫死 `if k == "statistics"`，新增 industry 之後
            # 它就收到空問題、找不到行業、回 None ——
            # 不會拋錯，只會安靜地什麼都不回答。
            # 用簽名判斷的話，新增指標時不會有人忘記更新那份清單。
            m = fn(data, question) if _takes_question(fn) else fn(data)
        except Exception as e:                         # noqa: BLE001
            m = Metric(k, k, f"[計算失敗：{e}]", None, "-", [])
        if m:
            out.append(m)
    return out


def render(metrics: list[Metric]) -> str:
    parts = []
    for m in metrics:
        parts.append(f"### {m.title}\n\n{m.text}\n\n"
                     f"*計算方式：{m.method}*\n"
                     f"*資料來源：{'、'.join(source_label(s) for s in m.sources)}*")
    return "\n\n---\n\n".join(parts)
