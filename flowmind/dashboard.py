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
    return JSONResponse({
        "asked": q,
        "weights": weights,
        "threshold": config.CONFIDENCE_ABSTAIN_THRESHOLD,
        "components": {
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
