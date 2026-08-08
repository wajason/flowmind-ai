#!/usr/bin/env python3
"""
demo_auth.py — 認證層（Authentication）現場演示
=============================================================================
【為什麼需要這支腳本】

我們原本只講「用 PostgreSQL Row-Level Security 做委任案隔離」。
那句話**只涵蓋了授權（Authorization），沒有涵蓋認證（Authentication）**，
而這兩件事是不同的問題：

    Authorization  「這個 tenant 能看哪些資料？」  ← RLS 解決
    Authentication 「你憑什麼說你是這個 tenant？」 ← RLS **不解決**

如果應用程式可以自己執行 `SET app.tenant_id = 'CASE-9999'`，
那 RLS 就只是一道自己對自己上的鎖 —— 任何拿到資料庫連線的人
（或任何一個 SQL injection）都能直接切換身分。

本演示證明四件事，每一件都當場跑給你看：

    ① 正常登入 → 只看得到自己被授權的委任案
    ② 換一個未授權的委任案 → 被擋下（不是回空資料，是明確拒絕）
    ③ 偽造 token → 被擋下
    ④ 撤銷授權 → 立即生效，不必等 session 過期

【關鍵設計：應用程式不能自己設定身分】

`app.tenant_id` 這個 session 變數**只能**由資料庫端的
`begin_session(token, tenant)` 函式設定，該函式是 SECURITY DEFINER，
會先驗證 token 的 SHA-256 雜湊、檢查有效期、再檢查該 principal
對這個 tenant 有沒有授權，全部通過才設定變數。

也就是說：**應用程式手上沒有可以直接冒充身分的路徑。**
這一點不是靠程式紀律維持的，是靠權限結構保證的 ——
`flowmind_app` 這個角色刻意不是 superuser、也不是 table owner，
因為那兩種身分都會繞過 RLS。

Usage:
    python scripts/demo_auth.py               # 完整四情境演示
    python scripts/demo_auth.py --json        # 機器可讀輸出
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import auth                                        # noqa: E402

W = 78


def hr(ch: str = "─") -> None:
    print(ch * W)


def title(s: str) -> None:
    print()
    hr("═")
    print(f"  {s}")
    hr("═")


def step(n: str, s: str) -> None:
    print(f"\n  [{n}] {s}")
    hr()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="輸出機器可讀結果")
    # 預設帳號來自 sql/init/01_auth.sql 的開發種子資料。
    # 密碼僅供離線 demo —— 正式環境走 SSO，不使用本地密碼。
    ap.add_argument("--user-a", default="alice")
    ap.add_argument("--pw-a", default="alice-dev-pw")
    ap.add_argument("--tenant-a", default="CASE-0001",
                    help="這個使用者有授權的委任案")
    ap.add_argument("--tenant-b", default="CASE-9999",
                    help="這個使用者**沒有**授權的委任案")
    args = ap.parse_args()

    if not args.json:
        title("FlowMind 認證層演示　Authentication ≠ Authorization")
        print("""
  RLS 回答的是「這個 tenant 能看哪些資料」，
  但沒有回答「你憑什麼說你是這個 tenant」。

  如果應用程式能自己執行 SET app.tenant_id = 'CASE-9999'，
  RLS 就只是一道自己對自己上的鎖。

  以下四個情境，全部當場實跑資料庫，不是模擬。
""")

    result = auth.verify_auth_chain(
        args.user_a, args.pw_a, args.tenant_a, args.tenant_b)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("passed") else 1

    print(auth.render_verification(result))

    title("這個演示證明了什麼")
    print("""
  ① 身分由**資料庫**驗證，不是由應用程式宣告
     begin_session() 是 SECURITY DEFINER，先驗 token 雜湊、
     再驗有效期、再驗授權，全過才設定 app.tenant_id。

  ② 應用程式沒有冒充身分的路徑
     flowmind_app 刻意不是 superuser、也不是 table owner ——
     那兩種身分都會繞過 RLS。這是權限結構保證，不是程式紀律。

  ③ token 只存 SHA-256 雜湊
     資料庫被拖走，也拿不到可直接使用的 token。

  ④ 撤銷即時生效
     授權檢查發生在每次 begin_session()，不是登入時快取一次。
     所以撤銷不必等 session 過期。
""")
    hr("═")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
