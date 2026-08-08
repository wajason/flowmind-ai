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
        method="Σ(該買方所有發票 total_amount) ÷ Σ(全部發票 total_amount)",
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
              f"註：逾期（還沒收到）與呆帳（已認定收不到）刻意分開計算 —— "
              f"兩者在授信上的意義不同，混在一起會讓正常公司看起來像要倒了。"),
        value={"open_total": open_total, "overdue_total": overdue,
               "overdue_ratio": round(overdue / (open_total or 1), 4),
               "written_off_total": wo_amt,
               "written_off_ratio": round(wo_amt / billed, 4),
               "buckets": {b: {"amount": buckets[b], "count": counts[b]} for b in buckets}},
        method="以 due_date 與今日相減分桶；status 為 PAID/WRITTEN_OFF 者排除於未收帳款之外",
        sources=["receivables.json"])


def m_cashflow(data: dict) -> Optional[Metric]:
    p = data["projection"]
    if not p:
        return None
    if p.get("gap_detected"):
        gap_d = _d(p["gap_date"])
        days = (gap_d - date.today()).days if gap_d else None
        head = (f"⚠ 預估 {p['gap_date']}"
                f"{f'（{days} 天後）' if days is not None else ''} 出現現金缺口，"
                f"金額約 NT${abs(p['gap_amount']):,.0f}。")
        advice = ("建議在缺口日前完成融資動撥。以本案的應收帳款結構，"
                  "應收帳款承購或信保供應商融資是常見的解法 —— "
                  "但適用條件請另行查詢法規與商品說明。")
    else:
        head = f"未來 {p.get('horizon_days', 90)} 天內未偵測到現金缺口。"
        advice = "目前現金部位可覆蓋已知的到期應付，無立即融資需求。"
    return Metric(
        key="cashflow",
        title="現金流缺口預測",
        text=(f"目前銀行餘額 NT${p.get('current_balance', 0):,.0f}。\n{head}\n\n"
              f"逾期應收 NT${p.get('overdue_receivables_total', 0):,.0f} "
              f"**未**計入本預測 —— 收不收得回來不確定，"
              f"樂觀假設它會準時入帳會嚴重高估償債能力。\n\n{advice}"),
        value={k: p.get(k) for k in ("current_balance", "gap_detected", "gap_date",
                                     "gap_amount", "horizon_days",
                                     "overdue_receivables_total")},
        method=p.get("computation_method", "deterministic_netting")
               + "：將未來到期的應收（僅 PENDING）與應付攤在時間軸上逐日累加",
        sources=["cash_flow_projection.json"])


def m_integrity(data: dict) -> Optional[Metric]:
    inv = data["invoices"]
    if not inv:
        return None
    rep = crosscheck.run_all(inv, data["contracts"], data["ledger"])
    failed = [f for f in rep["findings"] if not f["passed"]]
    lines = [f"  {'🔴' if f['severity']=='critical' else '🟡'} [{f['check_id']}] "
             f"{f['title']}：{f['detail']}" for f in failed]
    return Metric(
        key="integrity",
        title="憑證交叉驗證",
        text=(f"共執行 {len(rep['findings'])} 項決定性檢查，"
              f"完整性分數 {rep['integrity_score']:.1%}，"
              f"重大缺失 {rep['critical_failures']} 項。\n"
              f"送件建議：{'✅ 可送件' if rep['submission_ready'] else '⛔ 建議先補正重大缺失'}\n\n"
              + ("未通過項目：\n" + "\n".join(lines) if failed else "所有檢查項目皆通過。")),
        value=rep,
        method="見 flowmind/crosscheck.py；每一項皆為純算術判定，可由第三方以相同規則重算",
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
        method="直接彙總 receivables.json 全部紀錄",
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
        method="直接從 data/raw/SHARED 的原始 CSV/XLSX 讀取指定列，未經語言模型處理",
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
    ("integrity",     ["交叉驗證", "驗證", "造假", "可以送件", "能不能送件",
                       "有沒有問題", "憑證", "統編", "重複請款", "自我交易",
                       "送件前", "檢核"]),
    ("summary",       ["總覽", "應收總額", "開票金額", "多少張發票", "營收多少",
                       "整體狀況", "基本資料"]),
]


def route(question: str) -> list[str]:
    """回傳這個問題命中的決定性指標清單（可能多個，也可能空）。"""
    q = re.sub(r"\s+", "", question)
    keys = [key for key, kws in ROUTES if any(k in q for k in kws)]

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
                     f"*資料來源：{'、'.join(m.sources)}*")
    return "\n\n---\n\n".join(parts)
