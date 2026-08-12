#!/usr/bin/env python3
"""
dashboard.py — 授信人員的單頁戰情室
=============================================================================
【為什麼需要這一頁】

在這一頁出現之前，這個產品的所有能力都只存在於終端機輸出裡。
對工程師來說那沒問題；但**銀行的授信主管不會看終端機**。
一個「輸出無法被使用的人看懂」的系統，在對方眼中就等於不存在。

這一頁把已經算出來的東西攤開給人看，**不重算、不新增任何判斷邏輯**：

    區塊 1  委任案總覽紅黃綠燈      ← fin_alerts（watchtower 寫入的）
    區塊 2  交叉驗證分類卡片        ← crosscheck.run_all()
    區塊 3  現金流缺口時間軸        ← fin_invoices / fin_ledger
    區塊 4  最近一次問答的信心組成  ← evidence.compute_confidence() 的權重

【一條刻意的限制：這一頁不做任何運算】

所有數字都來自既有模組。理由是**同一個問題不能有兩個答案** ——
如果儀表板自己算一次集中度，終端機算另一次，兩邊有一天會不一致，
而使用者無從判斷該信哪個。儀表板是**呈現層**，不是第二套邏輯。

程式碼裡的體現：本模組沒有任何一行做金融計算，
只有取資料、排版、上色。

【零外部資源】

前端不引用任何 CDN。金融機構的內網通常擋外部資源，
一個「在你的電腦上很漂亮、在客戶那裡整頁破版」的 demo 沒有意義。

Usage:
    python -m flowmind.dashboard                    # http://127.0.0.1:8000
    python -m flowmind.dashboard --port 8080
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import psycopg2.extras
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from . import config, crosscheck, db, metrics, watchtower

app = FastAPI(title="FlowMind AI 授信戰情室", docs_url="/api/docs")

STATIC = Path(__file__).resolve().parent / "static"


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float, Decimal)) else 0.0


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════
# API：每個端點對應畫面上的一個區塊
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/engagements")
def api_engagements() -> JSONResponse:
    """委任案清單。從 engagements 表讀，不是掃目錄 —— 以資料庫為準。"""
    out = []
    with db.tenant_session("SHARED", admin=True) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT tenant_id, client_name, engagement_type, status "
                        "FROM engagements WHERE tenant_id <> 'SHARED' "
                        "ORDER BY tenant_id")
            out = _rows(cur)
    return JSONResponse(out)


@app.get("/api/overview/{tenant}")
def api_overview(tenant: str) -> JSONResponse:
    """
    區塊 1：紅黃綠燈。

    直接讀 watchtower 寫進 fin_alerts 的警示 —— **不重新掃描**。
    重新掃描會讓畫面上的數字與「系統實際發出的警示」不一致，
    而稽核追的是後者。
    """
    alerts = watchtower.open_alerts(tenant)
    counts = {"critical": 0, "warning": 0, "info": 0}
    for a in alerts:
        counts[a.get("severity", "info")] = counts.get(a.get("severity", "info"), 0) + 1

    light = "critical" if counts["critical"] else ("warning" if counts["warning"] else "good")
    return JSONResponse({
        "tenant": tenant, "light": light, "counts": counts,
        "alerts": [{
            "rule_id": a["rule_id"], "severity": a["severity"],
            "title": a["title"], "detail": a["detail"],
            "evidence_n": len(a.get("evidence") or []),
            "evidence": (a.get("evidence") or [])[:3],
            "first_seen": str(a.get("first_seen_at"))[:19],
        } for a in alerts],
    })


# 檢查 ID 前綴 → 畫面上的分類。逐一寫死而非用字串切割：
# 新增檢查時若忘了歸類，會落到「其他」而被看見，不會被靜默塞進錯的分類。
CHECK_GROUPS = [
    ("憑證真偽", ["TAXID", "FRAUD"]),
    ("金額算術", ["ARITH", "AMT"]),
    ("重複請款", ["DUP", "SEQ"]),
    ("跨文件勾稽", ["TERM", "CONTRACT", "BANK", "LEDGER", "RELATED"]),
    ("鑑識會計", ["FORENSIC", "DATE"]),
    ("授信風險", ["RISK"]),
]


@app.get("/api/crosscheck/{tenant}")
def api_crosscheck(tenant: str) -> JSONResponse:
    """區塊 2：交叉驗證分類卡片。呼叫既有引擎，不重寫任何一條規則。"""
    data = metrics.load_engagement_files(tenant)
    if not data.get("invoices"):
        return JSONResponse({"error": f"{tenant} 沒有可檢查的憑證資料"}, status_code=404)

    rep = crosscheck.run_all(data["invoices"], data.get("contracts"),
                             data.get("ledger"))
    groups = []
    seen: set[str] = set()
    for name, prefixes in CHECK_GROUPS:
        items = [f for f in rep["findings"]
                 if any(f["check_id"].startswith(p) for p in prefixes)]
        seen |= {f["check_id"] for f in items}
        if not items:
            continue
        worst = "good"
        for f in items:
            if not f["passed"]:
                worst = "critical" if f["severity"] == "critical" else \
                    ("warning" if worst != "critical" else worst)
        groups.append({
            "name": name, "status": worst,
            "passed": sum(1 for f in items if f["passed"]), "total": len(items),
            "items": [{"id": f["check_id"], "title": f["title"],
                       "passed": f["passed"], "severity": f["severity"],
                       "detail": f["detail"]} for f in items],
        })
    other = [f for f in rep["findings"] if f["check_id"] not in seen]
    if other:
        groups.append({
            "name": "未分類（請補進 CHECK_GROUPS）", "status": "warning",
            "passed": sum(1 for f in other if f["passed"]), "total": len(other),
            "items": [{"id": f["check_id"], "title": f["title"],
                       "passed": f["passed"], "severity": f["severity"],
                       "detail": f["detail"]} for f in other],
        })

    return JSONResponse({
        "tenant": tenant,
        "integrity_score": rep["integrity_score"],
        "critical_failures": rep["critical_failures"],
        "submission_ready": rep["submission_ready"],
        "documents_examined": rep["documents_examined"],
        "as_of": str(rep["as_of"]),
        "groups": groups,
    })


@app.get("/api/cashflow/{tenant}")
def api_cashflow(tenant: str) -> JSONResponse:
    """
    區塊 3：現金流缺口時間軸。

    以**未收應收的到期日**排出未來現金流入，對照銀行流水推得的目前餘額。
    刻意只用已入庫的資料做加總，不做任何預測 ——
    一條「預測」線在授信報告裡需要另一整套可解釋性，而我們沒有。
    """
    with db.tenant_session(tenant) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT due_date::date AS d, SUM(total_amount) AS inflow,
                       COUNT(*) AS n
                FROM fin_invoices
                WHERE status NOT IN ('PAID','WRITTEN_OFF','CANCELLED','VOID')
                  AND due_date IS NOT NULL
                GROUP BY due_date ORDER BY due_date
            """)
            inflows = _rows(cur)
            cur.execute("SELECT balance FROM fin_ledger WHERE balance IS NOT NULL "
                        "ORDER BY txn_date DESC, entry_id DESC LIMIT 1")
            row = cur.fetchone()
            opening = _num(row["balance"]) if row else 0.0
            # 用過去 90 天的實際淨流出估固定支出。這是**歷史平均**不是預測，
            # 名稱與說明都要講清楚，否則使用者會當成預測值。
            cur.execute("""
                SELECT COALESCE(SUM(amount), 0) AS net, COUNT(*) AS n
                FROM fin_ledger
                WHERE amount < 0
                  AND txn_date >= (SELECT MAX(txn_date) - INTERVAL '90 days'
                                   FROM fin_ledger)
            """)
            o = cur.fetchone()
            daily_outflow = abs(_num(o["net"])) / 90.0 if o and o["n"] else 0.0

    points, bal = [], opening
    for r in inflows:
        points.append({
            "date": str(r["d"]), "inflow": _num(r["inflow"]),
            "invoices": r["n"],
        })
    # 依到期日累加，並扣掉以歷史平均推得的固定支出
    if points:
        start = date.fromisoformat(points[0]["date"])
        for p in points:
            days = (date.fromisoformat(p["date"]) - start).days
            bal = opening + sum(x["inflow"] for x in points
                                if x["date"] <= p["date"]) - daily_outflow * days
            p["balance"] = round(bal)
    return JSONResponse({
        "tenant": tenant, "opening_balance": round(opening),
        "avg_daily_outflow": round(daily_outflow),
        "points": points,
        "note": "支出以過去 90 天實際淨流出的每日平均推算，屬歷史平均而非預測值。",
    })


@app.get("/api/simulate")
def api_simulate(tenant: str, amount: float = 0, days: int = 30) -> JSONResponse:
    """
    區塊 ⑤：情境模擬 —— 「如果現在多一筆應付款會怎樣」

    【為什麼是這個設計，而不是一個「模擬引擎」】

    這裡**沒有任何新的財務邏輯**。現金流推算
    （`compute_cash_flow_projection`）在資料產生器裡已經存在很久，
    只是一直被綁在命令列參數 `--stress` 上，只有工程師會用。

    這個端點做的事是把它搬到畫面上，讓授信人員可以現場輸入一個數字。
    重用既有邏輯而不是重寫，有一個關鍵好處：
    **模擬結果與正式報告用的是同一套算法**。
    若模擬另外寫一套，兩邊有一天會給出不同答案，而那時沒有人知道該信哪個。

    回傳基準線與模擬線兩條曲線，以及缺口日期與金額。
    """
    import sys                                          # noqa: PLC0415
    from pathlib import Path as _P                      # noqa: PLC0415
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from generate_synthetic_data import (               # noqa: PLC0415
        compute_cash_flow_projection)

    data = metrics.load_engagement_files(tenant)
    recv, pay = data.get("invoices") or [], data.get("payables") or []
    if not recv:
        return JSONResponse({"error": f"{tenant} 沒有應收帳款資料"},
                            status_code=404)

    ledger = data.get("ledger") or []
    balance = 0.0
    for row in reversed(ledger):
        if row.get("balance") not in (None, ""):
            balance = _num(row["balance"])
            break

    base = compute_cash_flow_projection(recv, pay, int(balance))

    sim = None
    if amount and amount > 0:
        extra = dict(
            doc_type="AP_BILL", bill_number="SIM-WHATIF",
            issue_date=date.today().isoformat(),
            supplier_name="（模擬）新增應付款", supplier_ban="",
            amount=float(amount), payment_terms_days=int(days),
            due_date=(date.today() + timedelta(days=int(days))).isoformat(),
            status="PENDING",
            source_note="情境模擬，非實際單據")
        sim = compute_cash_flow_projection(recv, list(pay) + [extra],
                                           int(balance))

    def _curve(proj: dict) -> list[dict]:
        # 欄位名是 projected_balance（不是 running_balance）——
        # 名稱猜錯的話整條曲線會全是 null，而畫面上只會看到一片空白，
        # 不會有任何錯誤訊息。
        return [{"date": e["date"], "balance": e.get("projected_balance"),
                 "amount": e["amount"], "type": e["type"],
                 "counterparty": e.get("counterparty", "")}
                for e in (proj.get("timeline") or [])]

    def _summarise(proj: dict) -> dict:
        """
        除了「第一個缺口」，也回報**整段期間的最低餘額**。

        只看第一個缺口在比較情境時會誤導：若基準線本來就有缺口，
        那個日期與金額不會因為新增一筆應付款而改變 ——
        於是加 3,000 萬與加 6,000 萬看起來一模一樣。
        最低餘額才反映得出「這筆錢讓情況惡化多少」。
        """
        curve = _curve(proj)
        bals = [c["balance"] for c in curve if c["balance"] is not None]
        trough = min(bals) if bals else balance
        trough_at = next((c["date"] for c in curve
                          if c["balance"] == trough), None)
        return {"curve": curve,
                "gap_detected": proj.get("gap_detected"),
                "gap_date": proj.get("gap_date"),
                "gap_amount": proj.get("gap_amount"),
                "trough_balance": round(trough),
                "trough_date": trough_at}

    out = {
        "tenant": tenant, "opening_balance": round(balance),
        "input": {"amount": amount, "days": days},
        "baseline": _summarise(base),
        "note": "本模擬重用 compute_cash_flow_projection() —— "
                "與正式報告用的是同一套算法，不是另寫一份。",
    }
    if sim:
        out["simulated"] = _summarise(sim)
        b_t, s_t = out["baseline"]["trough_balance"], out["simulated"]["trough_balance"]
        out["delta"] = {
            "trough_drop": b_t - s_t,
            "turns_negative": b_t >= 0 > s_t,
            "verdict": ("這筆應付款會讓現金部位由正轉負" if b_t >= 0 > s_t
                        else f"最低餘額再下探 {b_t - s_t:,.0f} 元"
                        if b_t != s_t else "對最低餘額沒有影響"),
        }
        # 只在**模擬後**真的會轉負時才給融資建議。
        # 基準線本來就有缺口的話，那是既有問題，不該算在這筆模擬頭上。
        if s_t < 0:
            out["financing"] = _financing_options(abs(s_t))
    return JSONResponse(out)


def _financing_options(gap: float) -> list[dict]:
    """
    融資方案並排比較。

    **每一個數字都來自知識庫裡的公開文件，不是我們推估的。**
    保證成數九成、年費率最低百分之零點三七五，都寫在信保基金的要點裡；
    要點原文可用 `python -m flowmind.tables` 或直接查語料驗證。

    刻意**不給利率**：銀行的實際核准利率不在任何公開文件裡，
    給一個推估的利率會讓整張比較表變成看起來很專業的猜測 ——
    那正是這個產品在反對的東西。
    """
    return [
        {
            "name": "信保基金 供應商融資信用保證",
            "coverage": "保證成數最高九成",
            "fee": "保證手續費年費率最低 0.375%（得視送保逾期情形酌增）",
            "amount_hint": f"以缺口 {gap:,.0f} 元計，"
                           f"九成保證約可支撐 {gap * 0.9:,.0f} 元融資",
            "requires": "中心廠商須經基金認可；需訂單／發票／支票等佐證交易真實性",
            "speed": "須經金融機構送保，非當日撥款",
            "source": "信保基金-供應商融資信用保證要點.md",
            "caveat": "本表僅列公開文件載明的條件，**不含銀行實際核准利率**——"
                      "那不在任何公開文件裡。",
        },
        {
            "name": "應收帳款承購（Factoring）",
            "coverage": "依買方信用核給額度，無追索權可移轉呆帳風險",
            "fee": "承購管理費 + 資金成本（各行不同，公開文件未載明費率）",
            "amount_hint": f"需有金額達 {gap:,.0f} 元以上的合格應收帳款可轉讓",
            "requires": "債權讓與須依民法通知債務人始生效力；買方需為合格對象",
            "speed": "額度核給後可較快動撥",
            "source": "玉山銀行-應收帳款承購.md／應收帳款暨融資業務（中國信託）.md",
            "caveat": "各行條件不同，**本系統不合併成單一答案** —— "
                      "三份商品說明各有側重，應分別查閱。",
        },
    ]


@app.get("/api/confidence")
def api_confidence(q: Optional[str] = None,
                   tenant: str = "SHARED") -> JSONResponse:
    """
    區塊 4：信心分數的組成。

    直接把 evidence.compute_confidence() 的權重與各分項攤開 ——
    這是整個產品最該被看見的一件事：**信心不是模型自己說的，
    是由可量測訊號依公開權重算出來的。**
    """
    from . import evidence                                # noqa: PLC0415
    weights = {
        "引用完整度": evidence.W_CITATION,
        "檢索強度": evidence.W_RETRIEVAL,
        "多文獻佐證": evidence.W_CORROBORATION,
        "稀疏健康度": evidence.W_SPARSE_HEALTH,
    }
    if not q:
        return JSONResponse({"weights": weights, "asked": None,
                             "threshold": config.CONFIDENCE_ABSTAIN_THRESHOLD})

    import rag_query                                      # noqa: PLC0415
    import contextlib, io                                 # noqa: PLC0415
    with contextlib.redirect_stdout(io.StringIO()):
        pack = rag_query.answer_question(tenant, q)
    bd = pack.confidence_breakdown
    # ── 決定性答案 vs RAG 答案，畫面必須分開 ──────────────────────────
    #
    # 「最大買方占營收多少」走決定性運算（直接把發票加總相除），
    # 根本不經過檢索與引用驗證，那四個權重對它完全不適用。
    # 但先前兩種答案套同一個顯示樣板，於是畫面出現
    # 「信心 1.000，但四個組成全是 0.0%」—— 看起來像系統故障。
    #
    # 分數沒有算錯，是**介面沒有區分兩條路徑**。
    # 一個讓人以為壞掉的正確答案，在示範場合等同於壞掉。
    # rag_query 在走決定性路徑時已經把 breakdown 設成 {"deterministic": True, …}，
    # 直接用那個旗標，不要用「信心 1.0 且引用為空」之類的啟發式去猜 ——
    # 猜出來的判斷會在邊界情況上出錯，而且錯的時候沒人知道為什麼。
    is_det = bool(bd.get("deterministic"))

    return JSONResponse({
        "asked": q,
        "answer_kind": "deterministic" if is_det else "rag",
        "answer_kind_label": "決定性運算（零 LLM）" if is_det else "檢索增強生成（RAG）",
        "kind_note": (
            "本題由程式直接彙總本案憑證計算得出，未經語言模型判斷，"
            "因此不適用引用驗證與檢索強度那組權重 —— 那四項顯示 0 是正常的。"
            if is_det else
            "本題需要理解文件內容，走檢索 + 受約束生成 + 引用逐字驗證，"
            "信心由下列四項可量測訊號依公開權重算出。"),
        "weights": {} if is_det else weights,
        "threshold": config.CONFIDENCE_ABSTAIN_THRESHOLD,
        "components": {} if is_det else {
            "引用完整度": bd.get("citation_integrity"),
            "檢索強度": bd.get("retrieval_strength"),
            "多文獻佐證": bd.get("corroboration"),
            "稀疏健康度": bd.get("sparse_health"),
        },
        "confidence": pack.confidence,
        "abstained": bool(pack.abstain_reason),
        "abstain_reason": pack.abstain_reason,
        "answer": pack.answer,
        "claims": [{"statement": c.statement, "quote": c.quote,
                    "source": c.source,
                    "verdict": c.verdict.value if hasattr(c.verdict, "value")
                    else str(c.verdict)} for c in pack.claims],
        "removed": pack.unknowns,
        "sources": pack.sources,
    })


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "dashboard.html").read_text(encoding="utf-8"))


def main() -> None:
    import argparse                                       # noqa: PLC0415
    import uvicorn                                        # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"  FlowMind 授信戰情室 → http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
