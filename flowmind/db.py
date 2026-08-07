"""
flowmind.db — RLS 感知的資料庫存取層
=============================================================================
本模組唯一對外的入口是 `tenant_session()`。這是刻意的設計：
系統裡任何一段程式想碰資料庫，都必須先講清楚「我現在是在替哪一個委任案工作」。

一般 SaaS 的做法是在 SQL 裡加 `WHERE tenant_id = ?`。那是「開發者自律」，
一次 code review 疏漏、一個忘記帶條件的 JOIN，就是跨客戶資料外洩。
在金融場域這不只是 bug，是可能觸及營業秘密與個資法的事故。

這裡改成：連線一建立就 `set_config('app.tenant_id', ...)`，
之後所有查詢由 PostgreSQL 的 Row-Level Security policy 強制過濾。
即使某支程式的 SQL 忘了加條件、甚至被 SQL injection 打穿，
資料庫層仍然只會回傳這個 engagement 看得到的列。

`verify_isolation()` 把這件事變成可以在評審面前跑一次的證明，而不是投影片上的一句話。
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

import psycopg2
import psycopg2.extras

from . import config

SHARED = "SHARED"


@contextlib.contextmanager
def tenant_session(
    tenant_id: str,
    *,
    admin: bool = False,
    role: str = "analyst",
    actor: Optional[str] = None,
) -> Iterator[psycopg2.extensions.connection]:
    """
    開一條綁定 engagement 的資料庫連線。

    admin=True 會用 superuser 連線並「繞過」RLS，只在建表、遷移、
    以及 verify_isolation() 需要對照組時使用。日常一律不要用。
    """
    if not tenant_id:
        raise ValueError("tenant_id 不可為空：系統不接受『不知道在替誰工作』的查詢。")

    conn = psycopg2.connect(config.db_url(admin=admin))
    try:
        with conn.cursor() as cur:
            # 第三個參數 is_local=false → session 層級，這條連線之後所有查詢都適用
            cur.execute(
                "SELECT set_config('app.tenant_id', %s, false),"
                "       set_config('app.role',      %s, false),"
                "       set_config('app.actor',     %s, false)",
                (tenant_id, role, actor or config.ACTOR),
            )
        conn.commit()
        yield conn
    finally:
        conn.close()


def current_tenant(conn) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT current_setting('app.tenant_id', true)")
        return cur.fetchone()[0]


# ── 稽核軌跡 ──────────────────────────────────────────────────────────────

def write_audit(
    conn,
    *,
    tenant_id: str,
    action: str,
    query_text: Optional[str] = None,
    doc_sources: Optional[list[str]] = None,
    confidence: Optional[float] = None,
    abstained: Optional[bool] = None,
    actor: Optional[str] = None,
) -> None:
    """
    寫一筆稽核紀錄。刻意記下「這次實際送進 LLM 的是哪幾份文件」，
    因為內控稽核真正會問的問題是「當初這個建議是根據什麼給的」，
    而不是「系統有沒有被使用過」。
    """
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO audit_log
               (actor, tenant_id, action, query_text, doc_sources, confidence, abstained)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (actor or config.ACTOR, tenant_id, action, query_text,
             doc_sources or [], confidence, abstained),
        )
    conn.commit()


def verify_audit_chain() -> tuple[bool, int, Optional[int]]:
    """
    重算整條雜湊鏈，確認稽核紀錄沒有被事後竄改或刪除。
    回傳 (是否完整, 檢查列數, 第一個斷點的 id)。
    """
    import hashlib

    with tenant_session(SHARED, admin=True) as conn:
        with conn.cursor() as cur:
            # ts 一定要用 ts::text 讓 PostgreSQL 自己格式化。
            # 若改在 Python 端 str(datetime)，會得到 '+00:00' 而 PostgreSQL 給的是 '+00'，
            # 雜湊值就對不起來 —— 驗證程式會誤報「稽核紀錄被竄改」，是很難查的假警報。
            cur.execute(
                "SELECT id, ts::text, actor, tenant_id, action, query_text,"
                "       doc_sources, prev_hash, row_hash FROM audit_log ORDER BY id"
            )
            rows = cur.fetchall()

    prev = "GENESIS"
    for (rid, ts, actor, tenant, action, qtext, sources, prev_hash, row_hash) in rows:
        if prev_hash != prev:
            return False, len(rows), rid
        payload = (prev_hash + str(ts) + actor + tenant + action +
                   (qtext or "") + (",".join(sources) if sources else ""))
        if hashlib.sha256(payload.encode()).hexdigest() != row_hash:
            return False, len(rows), rid
        prev = row_hash
    return True, len(rows), None


# ── engagement 管理 ───────────────────────────────────────────────────────

def upsert_engagement(
    tenant_id: str,
    client_name: str,
    engagement_type: str,
    industry_code: str | None = None,
    retention_years: int = 5,
) -> None:
    """
    建立/更新委任案。retention_years 預設 5 年：
    對齊商業會計法第 38 條會計憑證保存年限，到期後由清理作業刪除，
    而不是無限期留著客戶的發票與銀行流水。
    """
    with tenant_session(tenant_id, admin=True, role="admin") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO engagements
                   (tenant_id, client_name, engagement_type, industry_code, retention_until)
                   VALUES (%s, %s, %s, %s, CURRENT_DATE + (%s || ' years')::interval)
                   ON CONFLICT (tenant_id) DO UPDATE SET
                     client_name = EXCLUDED.client_name,
                     engagement_type = EXCLUDED.engagement_type,
                     industry_code = EXCLUDED.industry_code""",
                (tenant_id, client_name, engagement_type, industry_code, str(retention_years)),
            )
        conn.commit()


def list_engagements() -> list[dict]:
    with tenant_session(SHARED, admin=True, role="admin") as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT e.tenant_id, e.client_name, e.engagement_type, e.status,
                          e.retention_until,
                          (SELECT COUNT(*) FROM documents d WHERE d.tenant_id = e.tenant_id) AS chunks,
                          (SELECT COUNT(DISTINCT source) FROM documents d WHERE d.tenant_id = e.tenant_id) AS docs
                   FROM engagements e ORDER BY e.tenant_id"""
            )
            return [dict(r) for r in cur.fetchall()]


# ── 隔離證明 ──────────────────────────────────────────────────────────────

def verify_isolation(tenant_a: str, tenant_b: str) -> dict:
    """
    可執行的隔離證明。做三件事：
      1. 以 A 的身分連線，故意下一句「沒有 WHERE tenant_id」的 SQL，
         確認仍然看不到 B 的任何一列 —— 這就是 RLS 相對於手寫 WHERE 的價值。
      2. 以 A 的身分嘗試寫入標記為 B 的資料，確認被資料庫拒絕。
      3. 用 admin 身分繞過 RLS 取得真實列數，作為對照組，
         證明 B 的資料「確實存在」，只是 A 看不到（而不是根本沒資料的假通過）。
    """
    result: dict = {"tenant_a": tenant_a, "tenant_b": tenant_b}

    with tenant_session(tenant_a, admin=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents WHERE tenant_id = %s", (tenant_b,))
            result["b_rows_actually_exist"] = cur.fetchone()[0]

    with tenant_session(tenant_a) as conn:
        with conn.cursor() as cur:
            # 刻意不加任何 WHERE：模擬開發者忘記過濾的情境
            cur.execute("SELECT tenant_id, COUNT(*) FROM documents GROUP BY tenant_id")
            visible = {row[0]: row[1] for row in cur.fetchall()}
    result["visible_to_a"] = visible
    result["leak_detected"] = tenant_b in visible

    with tenant_session(tenant_a) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO documents (tenant_id, source, chunk_index, content)
                       VALUES (%s, '__isolation_probe__', 0, 'probe')""",
                    (tenant_b,),
                )
            conn.commit()
            result["cross_tenant_write_blocked"] = False
        except psycopg2.Error:
            conn.rollback()
            result["cross_tenant_write_blocked"] = True

    # 三種結果要分開，不能都叫「失敗」：
    #   inconclusive → 對照組根本沒資料，這個測試什麼也沒證明（不是隔離壞了）
    #   failed       → 真的看到了別人的資料，或寫得進去
    #   passed       → 對照組有資料、看不到、也寫不進去
    # 把 inconclusive 誤報成 failed，在評審面前是會出事的。
    if result["b_rows_actually_exist"] == 0:
        result["verdict"] = "inconclusive"
        result["passed"] = False
        result["note"] = (
            f"{tenant_b} 目前沒有任何資料，這個測試無法證明任何事 —— "
            f"看不到不存在的東西是理所當然的。\n"
            f"  請先建立對照組："
            f"python generate_synthetic_data.py --outdir data/raw/{tenant_b} && "
            f"python data_update_finance.py --tenant {tenant_b} --rebuild")
    elif result["leak_detected"] or not result["cross_tenant_write_blocked"]:
        result["verdict"] = "failed"
        result["passed"] = False
        result["note"] = "偵測到跨委任案存取，隔離失效，不得上線。"
    else:
        result["verdict"] = "passed"
        result["passed"] = True
        result["note"] = (f"{tenant_b} 確實有 {result['b_rows_actually_exist']} 筆資料存在，"
                          f"但 {tenant_a} 既讀不到也寫不進去。")
    return result
