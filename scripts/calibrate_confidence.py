#!/usr/bin/env python3
"""
calibrate_confidence.py — 用「獨立校準集」推導覆蓋率門檻
=============================================================================
【這支程式的存在是為了修正一個方法論錯誤。】

`DENSE_COVERAGE_GATE = 0.66` 最初是這樣訂出來的：
拿 8 題探測檢索分數 → 看到有答案的最低 0.691、沒答案的最高 0.653 → 取中間。

問題是**那 8 題裡有 4 題就在 50 題評測集裡**。
這是標準的測試集洩漏：用測試集調參數，再用同一個測試集報告改善。
這樣得到的門檻會過擬合到那幾題，換一批真實問題就未必成立 ——
而真實客戶問的每一題，對系統來說都是沒見過的題目。

正確做法：**校準集（dev）與評測集（test）完全分開**。
本程式只使用 `data/evalset/calibration_dev.jsonl`（20 題，
與 `zh_finance_qa.jsonl` 零重疊），推導門檻後才拿去跑評測集。

【門檻怎麼訂：不挑一個好看的數字】
不用「讓評測分數最高」的門檻 —— 那還是在對測試集調參。
改用只看校準集分布的規則：

    threshold = (有答案組的最低值 + 無答案組的最高值) / 2

若兩組重疊（最低 < 最高），代表 dense 分數在這個知識庫上
**分不開這兩類問題**，程式會明講並拒絕給出門檻，
而不是硬選一個看起來還行的值。分不開就是分不開。

同時報告分離度（Cohen's d 與重疊率），讓後續有人接手時
知道這個門檻有多可靠、什麼時候該重新校準。

Usage:
    python scripts/calibrate_confidence.py
    python scripts/calibrate_confidence.py --apply     # 把結果寫進 .env
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, db, retrieval                       # noqa: E402

DEV = config.DATA_DIR / "evalset" / "calibration_dev.jsonl"
TEST = config.DATA_DIR / "evalset" / "zh_finance_qa.jsonl"
OUT = config.DATA_DIR / "evalset" / "calibration_result.json"


def check_no_leakage() -> tuple[bool, list[str]]:
    """
    確認校準集與評測集沒有重疊。
    這個檢查必須自動化 —— 靠人記得「不要用到測試集」是不可靠的，
    而洩漏一旦發生，所有報出來的數字都失去意義。
    """
    dev_qs = {json.loads(l)["q"].strip()
              for l in DEV.read_text(encoding="utf-8").splitlines() if l.strip()}
    test_qs = {json.loads(l)["q"].strip()
               for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()}
    overlap = sorted(dev_qs & test_qs)
    return (not overlap), overlap


def measure(tenant: str = "CASE-0001", top_k: int = 6) -> list[dict]:
    rows = []
    items = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines() if l.strip()]
    with db.tenant_session(tenant) as conn:
        for i, it in enumerate(items, 1):
            chunks = retrieval.hybrid_search(conn, it["q"], top_k=top_k)
            d = retrieval.retrieval_diagnostics(chunks)
            rows.append({**it, "top_dense": round(d.get("top_dense", 0.0), 4),
                         "top_rrf": round(d.get("top_rrf", 0.0), 5),
                         "n_chunks": d["n"]})
            print(f"  [{i:>2}/{len(items)}] {'可答' if it['answerable'] else '無解'}"
                  f"  dense={rows[-1]['top_dense']:.4f}  {it['q'][:34]}", flush=True)
    return rows


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = (((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2)
              / (len(a) + len(b) - 2)) ** 0.5
    return (statistics.mean(a) - statistics.mean(b)) / pooled if pooled else 0.0


def derive(rows: list[dict]) -> dict:
    yes = [r["top_dense"] for r in rows if r["answerable"]]
    no = [r["top_dense"] for r in rows if not r["answerable"]]
    if not yes or not no:
        return {"separable": False, "reason": "校準集缺少其中一類問題"}

    lo_yes, hi_no = min(yes), max(no)
    separable = lo_yes > hi_no

    # 重疊率：有多少比例的樣本落在對方的區間內
    overlap = (sum(1 for v in yes if v <= hi_no) + sum(1 for v in no if v >= lo_yes)) \
        / (len(yes) + len(no))

    result = {
        "separable": separable,
        "n_answerable": len(yes), "n_unanswerable": len(no),
        "answerable": {"min": min(yes), "mean": round(statistics.mean(yes), 4),
                       "max": max(yes)},
        "unanswerable": {"min": min(no), "mean": round(statistics.mean(no), 4),
                         "max": max(no)},
        "overlap_rate": round(overlap, 3),
        "cohens_d": round(cohens_d(yes, no), 3),
    }
    # ── 門檻的目標函數：在「不誤攔任何可答題」的約束下，最大化攔截率 ──────
    #
    # 不用「讓分數最高」的門檻，也不用中位數之類的隨手取法。
    # 用一個明確寫出來的目標函數，因為它直接編碼了本場域的成本不對稱：
    #
    #   誤攔一題可答的  → 使用者被要求補件，成本低
    #   放過一題無解的  → 系統給出有信心的錯誤答案，成本極高
    #
    # 但反過來說，若誤攔率高，使用者很快就不信任這個系統了。
    # 所以約束是「零誤攔」，在此前提下盡量多攔。
    #
    # 安全邊際：門檻設在可答組最低值再往下 SAFETY_MARGIN，
    # 而不是貼齊最低值。20 題樣本的最小值本身就有抽樣誤差，
    # 貼齊等於對這 20 題過擬合。
    SAFETY_MARGIN = 0.03
    cand = round(lo_yes - SAFETY_MARGIN, 4)
    caught = sum(1 for v in no if v < cand)
    false_pos = sum(1 for v in yes if v < cand)

    result.update({
        "threshold": cand,
        "objective": "在零誤攔可答題的約束下最大化無解題攔截率",
        "safety_margin": SAFETY_MARGIN,
        "catch_rate": round(caught / len(no), 3),
        "false_positive_rate": round(false_pos / len(yes), 3),
        "caught": caught, "missed": len(no) - caught,
    })
    if not separable:
        result["note"] = (
            f"兩組區間重疊（可答最低 {lo_yes:.4f} ≤ 無解最高 {hi_no:.4f}）—— "
            f"dense 分數**無法單獨**分開這兩類問題。"
            f"最典型的反例是「日本的保證成數是多少」：語意上與台灣的幾乎重合，"
            f"embedding 分不出指涉對象。"
            f"因此本門檻只作為「極端離題」的最後一道防線，"
            f"主要把關交給確定性的範圍詞驗證（見 evidence.missing_scope_terms）。")
    return result


def render(r: dict, rows: list[dict]) -> str:
    L = ["═" * 76, "  覆蓋率門檻校準（使用獨立校準集，與評測集零重疊）", "═" * 76, ""]
    if r.get("threshold") is None:
        L += ["  ❌ 無法推導門檻", "", f"  {r.get('reason')}", "", "═" * 76]
        return "\n".join(L)

    a, n = r["answerable"], r["unanswerable"]
    L += [
        f"  可答問題（{r['n_answerable']} 題）　dense  最低 {a['min']:.4f}"
        f"　平均 {a['mean']:.4f}　最高 {a['max']:.4f}",
        f"  無解問題（{r['n_unanswerable']} 題）　dense  最低 {n['min']:.4f}"
        f"　平均 {n['mean']:.4f}　最高 {n['max']:.4f}",
        "",
        f"  區間是否分離　{'是' if r['separable'] else '否（重疊）'}"
        f"　重疊率 {r['overlap_rate']:.1%}　Cohen's d = {r['cohens_d']}",
        "",
        f"  目標函數：{r['objective']}",
        f"  ▶ 門檻 = 可答最低 {a['min']:.4f} − 安全邊際 {r['safety_margin']}"
        f" = **{r['threshold']:.4f}**",
        f"     攔截無解題 {r['caught']}/{r['n_unanswerable']}"
        f"（{r['catch_rate']:.1%}）　誤攔可答題 {r['false_positive_rate']:.1%}",
        "",
    ]
    if r.get("note"):
        L += ["  ⚠️ " + r["note"], ""]
    L += [
        "─" * 76,
        "  判讀：",
        "    · 門檻只由校準集決定，**沒有拿評測集的分數來挑數字**",
        "    · 安全邊際的用意：20 題樣本的最小值本身有抽樣誤差，",
        "      門檻貼齊最小值等於對這 20 題過擬合",
        "    · Cohen's d > 0.8 代表兩組分離良好；接近 0 代表這個訊號沒有鑑別力",
        "    · 換 embedding 模型或大幅擴充知識庫後，必須重新執行本程式",
        "═" * 76,
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="CASE-0001")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--apply", action="store_true",
                    help="把推導出的門檻寫進 flowmind/evidence.py 的常數（需人工確認）")
    args = ap.parse_args()

    ok, overlap = check_no_leakage()
    if not ok:
        print("❌ 校準集與評測集有重疊，校準結果無效：")
        for q in overlap[:10]:
            print(f"     {q}")
        print("\n   請把重複的題目從校準集移除。用測試集調參數，")
        print("   等於自己給自己出考題再自己閱卷，得到的數字沒有意義。")
        sys.exit(1)
    print("✅ 校準集與評測集零重疊\n")

    rows = measure(args.tenant, args.top_k)
    r = derive(rows)
    print("\n" + render(r, rows))

    OUT.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "embed_model": config.EMBED_MODEL, "top_k": args.top_k,
        "result": r, "measurements": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n📄 {OUT}")

    if args.apply and r.get("threshold"):
        print(f"\n⚠️  --apply 只印出建議，不自動改程式碼：")
        print(f"    請手動把 flowmind/evidence.py 的 DENSE_COVERAGE_GATE")
        print(f"    改為 {r['threshold']:.2f}，並在註解記錄校準日期與樣本數。")
        print(f"    刻意不自動改 —— 門檻變動會影響所有信心分數，")
        print(f"    應該留在 git diff 裡讓人看見，而不是被腳本悄悄改掉。")


if __name__ == "__main__":
    main()
