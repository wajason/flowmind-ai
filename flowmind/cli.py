"""
flowmind.cli — 決定性驗證的命令列入口
=============================================================================
    python -m flowmind.cli crosscheck --tenant CASE-0001
    python -m flowmind.cli crosscheck --tenant CASE-9999 --against-answer-key
    python -m flowmind.cli engagements
    python -m flowmind.cli doctor

`--against-answer-key` 是這支工具最重要的模式：
它把「交叉驗證引擎抓到的問題」與「刻意注入的已知瑕疵答案卷」自動比對，
算出漏抓（false negative）與誤報（false positive）。

沒有這一步，一份「全部通過 ✅」的報告什麼也證明不了 ——
可能是資料真的乾淨，也可能是檢查根本沒在運作。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import config, crosscheck, db, llm


# ══════════════════════════════════════════════════════════════════════════
def load_engagement_data(tenant_id: str) -> tuple[list, list, list]:
    """從 data/raw/<tenant>/ 讀取憑證。找不到的檔案就當作沒有，不報錯 ——
    真實客戶送來的文件本來就不會齊全，缺件本身是檢查結果的一部分。"""
    base = config.RAW_DIR / tenant_id
    if not base.exists():
        print(f"❌ 找不到 {base}")
        sys.exit(1)

    def load_json(name: str) -> list:
        p = base / name
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]

    invoices = load_json("receivables.json")
    contracts = load_json("contracts.json")

    ledger = []
    lp = base / "bank_ledger.csv"
    if lp.exists():
        with lp.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    row["amount"] = float(row.get("amount", 0) or 0)
                except ValueError:
                    row["amount"] = 0.0
                ledger.append(row)
    return invoices, contracts, ledger


def cmd_crosscheck(args) -> None:
    invoices, contracts, ledger = load_engagement_data(args.tenant)
    if not invoices:
        print(f"❌ {args.tenant} 沒有 receivables.json，無法執行交叉驗證。")
        sys.exit(1)

    report = crosscheck.run_all(invoices, contracts, ledger)
    print(crosscheck.render_text(report))

    # ── 對照答案卷 ────────────────────────────────────────────────────
    if args.against_answer_key:
        key_path = config.RAW_DIR / args.tenant / "_injected_defects_answer_key.json"
        if not key_path.exists():
            print(f"\n⚠️  找不到答案卷 {key_path.name}。"
                  f"請用 --inject-defects 產生負向對照組資料。")
        else:
            answer_key = json.loads(key_path.read_text(encoding="utf-8"))
            expected = {a["check_id"] for a in answer_key}
            failed = {f["check_id"] for f in report["findings"] if not f["passed"]}

            missed = expected - failed          # 該抓沒抓到
            spurious = failed - expected        # 沒注入卻報了

            print("\n" + "═" * 78)
            print("  負向對照組驗收：引擎抓到的 vs 實際注入的")
            print("═" * 78)
            for a in answer_key:
                hit = a["check_id"] in failed
                print(f"  {'✅ 抓到' if hit else '❌ 漏抓'}  [{a['check_id']}] "
                      f"{a['invoice_number']} — {a['defect']}")
            print("─" * 78)
            print(f"  漏抓 {len(missed)} 項　"
                  f"額外回報 {len(spurious)} 項 {sorted(spurious) or ''}")
            if spurious:
                print("  （額外回報未必是誤報：注入的瑕疵可能連帶觸發其他規則，"
                      "也可能是原始資料本身就有的狀況。應逐項判讀，不要自動當成錯誤。）")
            print("─" * 78)
            print(f"  結論：{'✅ 五項全中，檢查引擎有效' if not missed else '❌ 有漏抓，防線不完整'}")
            print("═" * 78)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📄 JSON 報告 → {out}")

    # 交叉驗證也要留稽核軌跡：這份報告可能被拿去跟銀行談，
    # 事後必須查得到「當時是根據哪批文件、在什麼時間跑出來的」。
    try:
        with db.tenant_session(args.tenant) as conn:
            db.write_audit(conn, tenant_id=args.tenant, action="crosscheck",
                           query_text=f"integrity={report['integrity_score']}",
                           doc_sources=["receivables.json", "contracts.json",
                                        "bank_ledger.csv"],
                           confidence=report["integrity_score"],
                           abstained=not report["submission_ready"])
    except Exception as e:                             # noqa: BLE001
        print(f"\n⚠️  稽核軌跡寫入失敗（資料庫可能未啟動）：{e}")


def cmd_engagements(_args) -> None:
    rows = db.list_engagements()
    print(f"\n{'engagement':<14}{'客戶':<30}{'類型':<24}{'文件':>6}{'chunks':>9}"
          f"{'保存至':>13}")
    print("─" * 96)
    for e in rows:
        print(f"{e['tenant_id']:<14}{(e['client_name'] or '')[:28]:<30}"
              f"{(e['engagement_type'] or '')[:22]:<24}"
              f"{e['docs']:>6}{e['chunks']:>9}{str(e['retention_until'] or '-'):>13}")
    print()


def cmd_doctor(_args) -> None:
    """開跑前的環境自檢。把「為什麼跑不起來」從除錯變成一句話。"""
    print("\n" + "═" * 70)
    print("  FlowMind 環境自檢")
    print("═" * 70)
    ok = True

    # Ollama
    if llm.ollama_available():
        models = llm.installed_models()
        print(f"  ✅ Ollama 連線正常（{len(models)} 個模型）")
        for role, name in (("抽取", config.EXTRACT_MODEL),
                           ("顧問", config.ADVISOR_MODEL),
                           ("合成", config.SYNTH_MODEL),
                           ("向量", config.EMBED_MODEL)):
            # Ollama 的模型名可能帶 :latest，比對時兩邊都容忍
            found = any(m == name or m.split(":")[0] == name.split(":")[0] for m in models)
            print(f"     {'✅' if found else '❌'} {role}模型 {name}"
                  f"{'' if found else '  ← 請執行 ollama pull ' + name}")
            ok &= found
    else:
        print(f"  ❌ 連不上 Ollama（{config.OLLAMA_BASE_URL}）。請先啟動 ollama。")
        ok = False

    # 資料庫
    try:
        with db.tenant_session(db.SHARED) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
                ver = cur.fetchone()
                cur.execute("SELECT COUNT(*) FROM documents")
                n = cur.fetchone()[0]
        print(f"  ✅ PostgreSQL 連線正常（pgvector {ver[0] if ver else '?'}）"
              f"，SHARED 知識庫 {n} 個 chunk")
        if n == 0:
            print("     ⚠️  SHARED 知識庫是空的："
                  "python data_update_finance.py --tenant SHARED --rebuild")
    except Exception as e:                             # noqa: BLE001
        print(f"  ❌ 資料庫連線失敗：{str(e)[:120]}")
        print("     → docker compose up -d")
        ok = False

    # 連線角色是否真的受 RLS 管轄
    try:
        with db.tenant_session(db.SHARED) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user, usesuper FROM pg_user "
                            "WHERE usename = current_user")
                user, is_super = cur.fetchone()
        if is_super:
            print(f"  ⚠️  目前以 superuser（{user}）連線 —— RLS 對 superuser 無效！")
            print("     → 請把 .env 的 PGUSER 改回 flowmind_app")
            ok = False
        else:
            print(f"  ✅ 連線角色 {user} 非 superuser，Row-Level Security 生效")
    except Exception:                                  # noqa: BLE001
        pass

    # 稽核鏈
    try:
        intact, n, bad = db.verify_audit_chain()
        print(f"  {'✅' if intact else '❌'} 稽核軌跡 {n} 筆，"
              f"雜湊鏈{'完整' if intact else f'在 id={bad} 斷裂'}")
    except Exception:                                  # noqa: BLE001
        pass

    # 資料夾
    for p in (config.RAW_DIR, config.BENCH_DIR):
        exists = p.exists()
        print(f"  {'✅' if exists else '⚠️ '} {p}"
              f"{'' if exists else '  ← 尚未建立'}")

    print("═" * 70)
    print(f"  {'✅ 環境就緒' if ok else '❌ 有項目需要處理，見上方提示'}")
    print("═" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser(prog="python -m flowmind.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crosscheck", help="對某個委任案執行決定性交叉驗證")
    c.add_argument("--tenant", "-t", required=True)
    c.add_argument("--against-answer-key", action="store_true",
                   help="與注入瑕疵的答案卷比對，算出漏抓與額外回報")
    c.add_argument("--json", help="同時輸出 JSON 報告到指定路徑")
    c.set_defaults(func=cmd_crosscheck)

    e = sub.add_parser("engagements", help="列出所有委任案")
    e.set_defaults(func=cmd_engagements)

    d = sub.add_parser("doctor", help="環境自檢")
    d.set_defaults(func=cmd_doctor)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
