#!/usr/bin/env python3
"""
financials.py — 把委任案財務檔案匯入 RLS 保護的資料表
=============================================================================
【為什麼要搬】

先前的隔離故事是「用 Row-Level Security 保護委任案資料」，
但 RLS 實際上只保護了 documents 與 chunks（文件與向量）。
最敏感的那批 —— 發票、應收帳款、合約條款、銀行流水 ——
一直放在磁碟的 JSON/CSV，由 metrics.load_engagement_files() 直接讀。

**資料庫層的隔離保證，管不到最敏感的資料。**
一個把 tenant_id 串進路徑的小 bug，就會讓 A 客戶的發票被 B 客戶讀到，
而且不會有任何錯誤訊息 —— 這是最難發現的那種洩漏。

【匯入不是取代】

原始檔案保留。客戶給的就是檔案，稽核也要能回到原件。
每一列都把**原始整列**存進 raw 欄位：如果我們的欄位對應理解錯了，
還原得回去。丟掉原始資料的匯入是不可逆的錯誤。

【欄位對應刻意不猜】

真實資料的欄位名與我們的表格不完全一致（例如發票用
`invoice_date` 而表格用 `issue_date`）。對應關係全部寫死在
FIELD_MAP 裡而不是用模糊比對 —— 猜錯一個欄位，
後面每一條監控規則都會安靜地算錯。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from . import db, metrics

# 檔案欄位 → 資料表欄位。刻意逐一寫死，不做模糊比對。
INVOICE_MAP = {
    "invoice_number": "invoice_number",
    "buyer_name": "buyer_name",
    "buyer_ban": "buyer_ban",
    "seller_name": "seller_name",
    "seller_ban": "seller_ban",
    "invoice_date": "issue_date",          # 注意：名稱不同
    "due_date": "due_date",
    "sales_amount": "amount",              # 注意：名稱不同
    "tax_amount": "tax_amount",
    "total_amount": "total_amount",
    "payment_terms_days": "payment_terms_days",
    "status": "status",
    "paid_date": "paid_date",
}

CONTRACT_MAP = {
    "contract_number": "contract_id",      # 注意：名稱不同
    "buyer_name": "counterparty",
    "buyer_ban": "counterparty_ban",
    "effective_date": "start_date",
    "expiry_date": "end_date",
    "payment_terms_days": "payment_terms_days",
    "annual_commitment_amount": "contract_amount",
}

LEDGER_MAP = {
    "date": "txn_date",
    "description": "description",
    "counterparty_ban": "counterparty",
    "amount": "amount",
    "balance": "balance",
    "reference": "ref_invoice",
}


def _date(v: Any) -> Optional[str]:
    if not v:
        return None
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _num(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


_DATE_COLS = {"issue_date", "due_date", "paid_date",
              "start_date", "end_date", "signed_date", "txn_date"}
_NUM_COLS = {"amount", "tax_amount", "total_amount", "contract_amount",
             "balance", "payment_terms_days"}


def _project(row: dict, mapping: dict[str, str]) -> dict:
    out: dict = {}
    for src, dst in mapping.items():
        v = row.get(src)
        if dst in _DATE_COLS:
            v = _date(v)
        elif dst in _NUM_COLS:
            v = _num(v)
        out[dst] = v
    out["raw"] = json.dumps(row, ensure_ascii=False, default=str)
    return out


def ingest(tenant_id: str, replace: bool = True) -> dict:
    """
    把檔案匯入資料表。回傳每張表的匯入筆數。

    `replace=True` 先清空該委任案的資料再匯入 ——
    因為這是「以檔案為準」的同步，不是累加。
    增量式的匯入會讓「檔案刪掉的那筆」永遠留在資料庫裡，
    而那種殘留在授信場域是危險的（已作廢的發票還在算集中度）。
    """
    data = metrics.load_engagement_files(tenant_id)
    stats = {"invoices": 0, "contracts": 0, "ledger": 0}

    with db.tenant_session(tenant_id) as conn:
        with conn.cursor() as cur:
            if replace:
                cur.execute("DELETE FROM fin_invoices")
                cur.execute("DELETE FROM fin_contracts")
                cur.execute("DELETE FROM fin_ledger")

            for row in data.get("invoices", []):
                r = _project(row, INVOICE_MAP)
                if not r.get("invoice_number"):
                    continue
                cur.execute("""
                    INSERT INTO fin_invoices
                      (tenant_id, invoice_number, buyer_name, buyer_ban,
                       seller_name, seller_ban, issue_date, due_date, amount,
                       tax_amount, total_amount, payment_terms_days, status,
                       paid_date, raw)
                    VALUES (%(t)s, %(invoice_number)s, %(buyer_name)s,
                            %(buyer_ban)s, %(seller_name)s, %(seller_ban)s,
                            %(issue_date)s, %(due_date)s, %(amount)s,
                            %(tax_amount)s, %(total_amount)s,
                            %(payment_terms_days)s, %(status)s, %(paid_date)s,
                            %(raw)s)
                    ON CONFLICT (tenant_id, invoice_number) DO NOTHING
                """, {**r, "t": tenant_id})
                stats["invoices"] += cur.rowcount

            for row in data.get("contracts", []):
                r = _project(row, CONTRACT_MAP)
                if not r.get("contract_id"):
                    continue
                cur.execute("""
                    INSERT INTO fin_contracts
                      (tenant_id, contract_id, counterparty, counterparty_ban,
                       start_date, end_date, payment_terms_days,
                       contract_amount, raw)
                    VALUES (%(t)s, %(contract_id)s, %(counterparty)s,
                            %(counterparty_ban)s, %(start_date)s, %(end_date)s,
                            %(payment_terms_days)s, %(contract_amount)s, %(raw)s)
                    ON CONFLICT (tenant_id, contract_id) DO NOTHING
                """, {**r, "t": tenant_id})
                stats["contracts"] += cur.rowcount

            for row in data.get("ledger", []):
                r = _project(row, LEDGER_MAP)
                cur.execute("""
                    INSERT INTO fin_ledger
                      (tenant_id, txn_date, description, counterparty,
                       amount, balance, ref_invoice, raw)
                    VALUES (%(t)s, %(txn_date)s, %(description)s,
                            %(counterparty)s, %(amount)s, %(balance)s,
                            %(ref_invoice)s, %(raw)s)
                """, {**r, "t": tenant_id})
                stats["ledger"] += cur.rowcount

        conn.commit()
    return stats


def verify_isolation(tenant_a: str, tenant_b: str) -> dict:
    """
    證明財務明細確實受 RLS 保護 ——
    用 tenant_a 的身分連線，看不看得到 tenant_b 的發票。

    注意查詢語句裡**沒有** WHERE tenant_id：
    隔離若靠應用程式加條件，那就只是一個約定，不是保證。
    """
    out: dict = {"tenant_a": tenant_a, "tenant_b": tenant_b}
    with db.tenant_session(tenant_a) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fin_invoices")
            out["visible_as_a"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT tenant_id) FROM fin_invoices")
            out["distinct_tenants_visible"] = cur.fetchone()[0]
    with db.tenant_session(tenant_b) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fin_invoices")
            out["visible_as_b"] = cur.fetchone()[0]
    out["isolated"] = out["distinct_tenants_visible"] <= 1
    return out


__all__ = ["ingest", "verify_isolation", "INVOICE_MAP", "CONTRACT_MAP",
           "LEDGER_MAP"]
