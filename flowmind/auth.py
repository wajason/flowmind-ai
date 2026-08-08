"""
flowmind.auth — 認證層（誰在使用系統）
=============================================================================
【這一層補的是一個被問到才發現的缺口】

被問：「你們的系統怎麼判斷這個人有沒有權限？使用時要登入嗎？掃臉嗎？」
當時答不出來 —— 因為 RLS 做的是**授權**，不是**認證**。

  授權（authorization）：給定「這條連線代表 CASE-0001」，強制它只能看 CASE-0001
  認證（authentication）：證明「你是 alice，而 alice 被授權存取 CASE-0001」

原本的 `db.tenant_session("CASE-0001")` 是應用程式**自己宣告**身分。
只要應用程式被打穿（RCE、SQL injection、內部人員直接連資料庫），
攻擊者就能把自己設成任何 tenant，RLS 形同虛設。

【修法：應用程式不再能宣告身分】

改為出示工作階段權杖，由資料庫端的 `begin_session()`（SECURITY DEFINER）
驗證權杖 → 查主體狀態 → 查授權紀錄 → 才設定 RLS 依賴的上下文。
應用程式角色**沒有直接寫 app.tenant_id 的權限**，也不能改授權表。

完整的鏈：
    權杖 → 使用者 → 授權紀錄（誰授權、何時到期）→ 委任案 → RLS 過濾
每一環都在資料庫裡，稽核可以逐環查證。

【誠實的邊界】

本層**不取代企業 SSO**。它假設權杖沒有被竊取。
正式部署應把權杖發放接上 OIDC/SAML，由 IdP 負責身分證明（含 MFA、裝置信任）；
本層負責的是「這個已驗證的身分，能碰哪些委任案」。

所以問「要掃臉嗎」的答案是：**掃臉屬於 IdP 的職責**，
本系統要求的是一個可驗證的身分斷言，MFA 強度由機構的 IdP 政策決定。
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import psycopg2
import psycopg2.extras

from . import config

DEFAULT_TTL_HOURS = 8          # 一個工作日。過長的工作階段是稽核常見缺失。


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Principal:
    user_id: str
    display_name: str
    role: str
    status: str


@dataclass
class SessionContext:
    token: str
    user_id: str
    role: str
    tenant_id: str
    access_level: str
    expires_at: datetime


class AuthError(RuntimeError):
    """認證或授權失敗。刻意不細分原因給呼叫端，避免變成資訊洩漏管道。"""


# ══════════════════════════════════════════════════════════════════════════
# 1. 登入（開發用本地密碼；正式環境應改由 IdP 斷言）
# ══════════════════════════════════════════════════════════════════════════

def login(user_id: str, password: str, ttl_hours: int = DEFAULT_TTL_HOURS,
          note: str = "") -> str:
    """
    驗證本地密碼並發出工作階段權杖。回傳權杖明文（只在此刻存在一次）。

    資料庫只存權杖的 SHA-256 —— 資料庫被讀走也無法冒用既有工作階段。
    這與密碼儲存的原則相同：**永遠不要存可以直接拿來用的東西**。
    """
    conn = psycopg2.connect(config.db_url(admin=True))
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash, status FROM principals WHERE user_id=%s",
                (user_id,))
            row = cur.fetchone()
            # 帳號不存在與密碼錯誤回傳同一個錯誤，避免帳號列舉
            if not row or row[1] != "active" or row[0] != _sha256(password):
                raise AuthError("帳號或密碼錯誤")

            token = secrets.token_urlsafe(32)
            exp = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
            cur.execute(
                """INSERT INTO auth_sessions (token_hash, user_id, expires_at, client_note)
                   VALUES (%s, %s, %s, %s)""",
                (_sha256(token), user_id, exp, note or None))
        conn.commit()
        return token
    finally:
        conn.close()


def logout(token: str) -> None:
    conn = psycopg2.connect(config.db_url(admin=True))
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE auth_sessions SET revoked_at=NOW() "
                        "WHERE token_hash=%s AND revoked_at IS NULL", (_sha256(token),))
        conn.commit()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
# 2. 授權管理（只有 admin 能做）
# ══════════════════════════════════════════════════════════════════════════

def grant_access(user_id: str, tenant_id: str, granted_by: str,
                 access_level: str = "read", days: int = 90) -> None:
    """
    授權某人存取某委任案，並**預設 90 天到期**。

    到期日不是可選項：人員調動後權限沒收回，是內控稽核最常開的缺失。
    要長期存取就定期重新授權，而重新授權這個動作本身就會留下紀錄。
    """
    conn = psycopg2.connect(config.db_url(admin=True))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO engagement_access
                   (user_id, tenant_id, access_level, granted_by, expires_at)
                   VALUES (%s, %s, %s, %s, NOW() + (%s || ' days')::interval)
                   ON CONFLICT (user_id, tenant_id) DO UPDATE SET
                     access_level=EXCLUDED.access_level,
                     granted_by=EXCLUDED.granted_by,
                     granted_at=NOW(), expires_at=EXCLUDED.expires_at,
                     revoked_at=NULL""",
                (user_id, tenant_id, access_level, granted_by, str(days)))
        conn.commit()
    finally:
        conn.close()


def revoke_access(user_id: str, tenant_id: str) -> None:
    conn = psycopg2.connect(config.db_url(admin=True))
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE engagement_access SET revoked_at=NOW() "
                        "WHERE user_id=%s AND tenant_id=%s", (user_id, tenant_id))
        conn.commit()
    finally:
        conn.close()


def list_access(user_id: Optional[str] = None) -> list[dict]:
    conn = psycopg2.connect(config.db_url(admin=True))
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT a.user_id, p.display_name, p.role, a.tenant_id,
                          a.access_level, a.granted_by, a.granted_at,
                          a.expires_at, a.revoked_at,
                          (a.revoked_at IS NULL
                           AND (a.expires_at IS NULL OR a.expires_at > NOW())) AS effective
                   FROM engagement_access a
                   JOIN principals p ON p.user_id = a.user_id
                   WHERE (%s IS NULL OR a.user_id = %s)
                   ORDER BY a.user_id, a.tenant_id""",
                (user_id, user_id))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
# 3. 以權杖開啟工作階段（取代直接指定 tenant_id）
# ══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def authenticated_session(token: str, tenant_id: str) -> Iterator[tuple]:
    """
    出示權杖進入某個委任案。回傳 (conn, SessionContext)。

    與 `db.tenant_session()` 的關鍵差別：
    這裡**不接受呼叫端宣告身分**。tenant_id 只是「想進入哪個委任案」的請求，
    准不准由資料庫端的 begin_session() 依授權紀錄決定。
    """
    conn = psycopg2.connect(config.db_url())
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT * FROM begin_session(%s, %s)", (token, tenant_id))
                row = cur.fetchone()
            except psycopg2.Error as e:
                conn.rollback()
                # 統一錯誤訊息，不洩漏「委任案是否存在」
                raise AuthError(str(e).split("\n")[0].strip()) from None
            if not row:
                raise AuthError("無法建立工作階段")
            user_id, role, level = row
        conn.commit()
        yield conn, SessionContext(token=token, user_id=user_id, role=role,
                                   tenant_id=tenant_id, access_level=level,
                                   expires_at=datetime.now(timezone.utc))
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════
# 4. 可執行的驗證：認證鏈是否真的擋得住
# ══════════════════════════════════════════════════════════════════════════

def verify_auth_chain(user_a: str, pw_a: str,
                      allowed_tenant: str, denied_tenant: str) -> dict:
    """
    可以在評審面前跑的認證鏈驗證。四個情境：

      1. 正確權杖 + 有授權的委任案      → 應成功
      2. 正確權杖 + **沒有授權**的委任案 → 應被拒（這是資訊隔離牆的核心）
      3. 偽造權杖                       → 應被拒
      4. 撤銷授權後再存取原本可存取的    → 應立即失效（人員調動情境）
    """
    result: dict = {"user": user_a, "allowed": allowed_tenant, "denied": denied_tenant}

    token = login(user_a, pw_a, note="verify_auth_chain")
    result["token_issued"] = bool(token)

    # ① 有授權
    try:
        with authenticated_session(token, allowed_tenant) as (conn, ctx):
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                result["allowed_visible_rows"] = cur.fetchone()[0]
        result["allowed_ok"] = True
    except AuthError as e:
        result["allowed_ok"] = False
        result["allowed_error"] = str(e)[:120]

    # ② 沒授權
    try:
        with authenticated_session(token, denied_tenant) as (conn, _):
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                result["denied_visible_rows"] = cur.fetchone()[0]
        result["denied_blocked"] = False
    except AuthError as e:
        result["denied_blocked"] = True
        result["denied_error"] = str(e)[:120]

    # ③ 偽造權杖
    try:
        with authenticated_session("forged-" + secrets.token_urlsafe(24),
                                   allowed_tenant):
            pass
        result["forged_blocked"] = False
    except AuthError:
        result["forged_blocked"] = True

    # ④ 撤銷後立即失效
    revoke_access(user_a, allowed_tenant)
    try:
        with authenticated_session(token, allowed_tenant):
            pass
        result["revocation_effective"] = False
    except AuthError:
        result["revocation_effective"] = True
    finally:
        grant_access(user_a, allowed_tenant, granted_by="verify_auth_chain")

    logout(token)
    result["passed"] = all([
        result.get("allowed_ok"), result.get("denied_blocked"),
        result.get("forged_blocked"), result.get("revocation_effective")])
    return result


def render_verification(r: dict) -> str:
    def mark(ok): return "✅" if ok else "❌"
    L = [
        "═" * 82,
        f"  認證鏈驗證　使用者 {r['user']}",
        "═" * 82, "",
        f"  {mark(r.get('token_issued'))} 登入成功，取得工作階段權杖"
        f"（資料庫只存 SHA-256，不存權杖本身）",
        f"  {mark(r.get('allowed_ok'))} 存取**有授權**的 {r['allowed']}："
        f"看到 {r.get('allowed_visible_rows', '—')} 列",
        f"  {mark(r.get('denied_blocked'))} 存取**沒有授權**的 {r['denied']}："
        f"{r.get('denied_error', '竟然成功了')}",
        f"  {mark(r.get('forged_blocked'))} 偽造權杖：被拒絕",
        f"  {mark(r.get('revocation_effective'))} 撤銷授權後立即失效"
        f"（人員調動／離職情境）",
        "", "─" * 82,
        f"  結論：{'✅ 認證鏈有效' if r.get('passed') else '❌ 認證鏈有缺口，不得上線'}",
        "",
        "  注意應用程式**沒有**直接設定 app.tenant_id 的權限 ——",
        "  它只能出示權杖請求進入某個委任案，准不准由資料庫端依授權紀錄決定。",
        "  這代表即使應用程式被打穿，攻擊者也無法把自己設成任意委任案。",
        "",
        "  完整的鏈：權杖 → 使用者 → 授權紀錄（誰授權、何時到期）→ 委任案 → RLS",
        "  每一環都在資料庫裡，內控稽核可以逐環查證。",
        "═" * 82,
    ]
    return "\n".join(L)
