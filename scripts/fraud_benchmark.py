#!/usr/bin/env python3
"""
fraud_benchmark.py — 造假偵測的完整評測（precision / recall / F1 / MCC）
=============================================================================
【為什麼不能只報「五項全中」】

「五項注入缺陷全中」聽起來很好，但它沒有分母：
  · 總共測了幾張發票？
  · 乾淨的資料會不會被誤報？（precision）
  · 有多少種造假樣態根本沒測？（recall 的真實分母）

沒有這三個數字，「全中」在統計上不構成證據。
本程式對 22 種造假樣態做完整評測，並報告混淆矩陣與四種指標。

─────────────────────────────────────────────────────────────────────────────
【為什麼四種指標都要報，不能只挑一個】

**Accuracy（準確率）在這個任務上會嚴重誤導。**
造假是稀有事件：100 張發票裡可能只有 3 張有問題。
一個永遠回答「全部乾淨」的系統，accuracy 是 97% —— 但它一張都沒抓到。
所以 accuracy 只作為對照，**不作為主指標**。

| 指標 | 回答什麼問題 | 在本任務的意義 |
|---|---|---|
| **Precision** | 系統說有問題的，真的有問題嗎 | 誤報會浪費授信人員的時間，高誤報會讓人乾脆關掉警示 |
| **Recall** | 真的有問題的，抓到幾成 | 漏抓 = 造假案件通過審查，這是最貴的錯誤 |
| **F1** | precision 與 recall 的調和平均 | 單一比較數字，但**它預設兩種錯誤等價**，在本任務並不成立 |
| **MCC** | 考慮混淆矩陣全部四格的相關係數 | **類別極度不平衡時最可靠的單一指標**，範圍 −1~+1 |
| **Specificity** | 乾淨的資料有多少被正確放行 | 直接對應「系統會不會吵到不能用」 |

**MCC（Matthews Correlation Coefficient）是本任務最適合的單一指標**，
因為它是唯一在四格（TP/TN/FP/FN）都被納入計算、且對類別不平衡穩健的係數。
F1 完全忽略 TN，在「多數樣本是乾淨的」場景會給出過於樂觀的印象。

─────────────────────────────────────────────────────────────────────────────
【還有一個更貼近場域的指標：成本加權分數】

precision 與 recall 的取捨在本場域是**不對稱**的：
  · 漏抓一件造假（FN）→ 銀行放款給虛假交易，可能是數百萬的損失 + 商譽
  · 誤報一件（FP）→ 授信人員多花十分鐘查證

所以另外報告成本加權分數，把這個不對稱寫進公式：

    cost = C_fn × FN + C_fp × FP
    預設 C_fn : C_fp = 20 : 1

20:1 是**假設值**，公開為可調參數。每家機構可以帶入自己的成本結構重算。

Usage:
    python scripts/fraud_benchmark.py
    python scripts/fraud_benchmark.py --trials 30 --seed 2026
    python scripts/fraud_benchmark.py --tier 3        # 只測跨文件樣態
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, crosscheck, fraud_injector                # noqa: E402
from flowmind.fraud_injector import ALL_DEFECTS, STATISTICAL_DEFECTS   # noqa: E402

TIER_NAME = {1: "單張可判定", 2: "跨憑證", 3: "跨文件", 4: "統計性", 5: "已知抓不到"}

# 成本比：漏抓一件造假 vs 誤報一件。公開為可調參數。
COST_FN, COST_FP = 20.0, 1.0


def load_clean(tenant: str = "CASE-0001"):
    base = config.RAW_DIR / tenant
    inv = json.loads((base / "receivables.json").read_text(encoding="utf-8"))
    con = json.loads((base / "contracts.json").read_text(encoding="utf-8"))
    led = []
    with (base / "bank_ledger.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["amount"] = float(r.get("amount") or 0)
            r["balance"] = float(r.get("balance") or 0)
            led.append(r)
    return inv, con, led


def failed_checks(report: dict) -> set[str]:
    return {f["check_id"] for f in report["findings"] if not f["passed"]}


# ══════════════════════════════════════════════════════════════════════════
# 指標
# ══════════════════════════════════════════════════════════════════════════

def metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) else None

    # MCC：類別不平衡時最可靠的單一指標。分母為 0 時無定義（不硬給 0）
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else None

    cost = COST_FN * fn + COST_FP * fp
    # 最壞情況：全部漏抓。用來把成本正規化成 0~1 的「相對表現」
    worst = COST_FN * (tp + fn) if (tp + fn) else 1.0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "specificity": spec,
        "accuracy": acc, "f1": f1, "mcc": mcc,
        "cost": cost, "cost_ratio": f"{COST_FN:.0f}:{COST_FP:.0f}",
        "cost_normalized": round(1 - cost / worst, 4) if worst else None,
    }


def fmt(v, pct=True) -> str:
    if v is None:
        return "  n/a"
    return f"{v:6.1%}" if pct else f"{v:6.3f}"


# ══════════════════════════════════════════════════════════════════════════
def run(trials: int, seed: int, tiers: list[int]) -> dict:
    inv0, con0, led0 = load_clean()

    # ── 基線：乾淨資料應該全數通過（測 false positive）─────────────
    clean_report = crosscheck.run_all(inv0, con0, led0)
    clean_failed = failed_checks(clean_report)
    print(f"▶ 基線（乾淨資料，{len(inv0)} 張發票）")
    print(f"    完整性 {clean_report['integrity_score']:.1%}　"
          f"未通過檢查 {len(clean_failed)} 項"
          + (f"：{sorted(clean_failed)}" if clean_failed else ""))
    print(f"    ↳ 這些是**誤報**（資料是乾淨的），會直接計入 FP\n")

    targets = [(d, t, f) for d, t, f in ALL_DEFECTS if t in tiers]
    per_defect: dict[str, dict] = {}
    tot = defaultdict(int)

    print(f"▶ 逐樣態評測（{len(targets)} 種 × {trials} 次獨立試驗）\n")
    print(f"  {'ID':<5}{'Tier':<6}{'樣態':<20}{'命中':<8}{'預期檢查':<14}結果")
    print("  " + "─" * 88)

    # ── 混淆矩陣的分析單位是 (試驗 × 檢查項) ────────────────────────
    #
    # 第一版把「乾淨資料本來就會報的項目」算成 FP，導致 precision 只有 44.9%。
    # 那在概念上是錯的：CONTRACT-01 在乾淨資料上就會報，
    # 因為那批資料**真的有 9 張重大發票沒有合約** —— 那是對資料的正確描述，
    # 不是誤報。把它算成 FP，等於因為系統說了實話而扣它分。
    #
    # 正確定義：
    #   TP = 注入的瑕疵，其對應檢查有觸發
    #   FN = 注入的瑕疵，其對應檢查沒觸發
    #   FP = **注入後才出現、且乾淨資料不會報**的其他檢查（真正的連帶誤報）
    #   TN = 注入後仍正確保持沉默的檢查
    # 乾淨資料上就會報的項目一律排除在混淆矩陣之外，另外單獨列為「資料既有發現」。
    n_checks = len(clean_report["findings"])
    for did, tier, _fn in targets:
        hits = 0
        spurious = 0            # 真正的連帶誤報
        silent_ok = 0           # 正確保持沉默
        name, expected = did, None

        for k in range(trials):
            # 統計性樣態會大量改動資料，單獨注入以免互相干擾
            res = fraud_injector.inject(inv0, con0, led0, [did], seed=seed + k)
            if not res.defects:
                continue
            name, expected = res.defects[0].name, res.defects[0].expected_check
            rep = crosscheck.run_all(res.invoices, res.contracts, res.ledger)
            failed = failed_checks(rep)

            if expected and expected in failed:
                hits += 1
            # 扣掉「乾淨資料本來就會報的」與「這次注入的目標」
            extra = failed - clean_failed - ({expected} if expected else set())
            spurious += len(extra)
            silent_ok += n_checks - len(failed | clean_failed)

        detectable = expected is not None
        rate = hits / trials if trials else 0.0
        per_defect[did] = {
            "tier": tier, "name": name, "expected_check": expected,
            "detectable_by_design": detectable,
            "trials": trials, "hits": hits, "detection_rate": round(rate, 4),
            "avg_spurious_alerts": round(spurious / max(1, trials), 2),
        }

        if detectable:
            tot["TP"] += hits
            tot["FN"] += trials - hits
        else:
            # Tier 5：設計上就抓不到。仍計入 FN，否則 recall 會虛高
            tot["FN_known_undetectable"] += trials
        tot["FP"] += spurious
        tot["TN"] += silent_ok

        mark = "✅" if (detectable and rate >= 0.95) else (
            "⚠️" if detectable else "🔒")
        print(f"  {did:<5}{TIER_NAME[tier][:4]:<6}{name[:18]:<20}"
              f"{rate:>5.0%}   {str(expected or '（設計上抓不到）'):<14}{mark}")

    detectable_m = metrics(tot["TP"], tot["FP"], tot["TN"], tot["FN"])
    honest_m = metrics(tot["TP"], tot["FP"], tot["TN"],
                       tot["FN"] + tot["FN_known_undetectable"])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trials_per_defect": trials, "seed": seed, "tiers": tiers,
        "invoices_per_trial": len(inv0),
        "checks_per_run": n_checks,
        "baseline_findings": sorted(clean_failed),
        "per_defect": per_defect,
        "metrics_detectable_only": detectable_m,
        "metrics_including_known_undetectable": honest_m,
        "known_undetectable_trials": tot["FN_known_undetectable"],
    }


def render(r: dict) -> str:
    a, b = r["metrics_detectable_only"], r["metrics_including_known_undetectable"]
    L = [
        "", "═" * 92,
        "  造假偵測完整評測結果",
        "═" * 92, "",
        f"  試驗設定：{len(r['per_defect'])} 種樣態 × {r['trials_per_defect']} 次"
        f" = {len(r['per_defect']) * r['trials_per_defect']} 次獨立試驗",
        f"  每次試驗：{r['invoices_per_trial']} 張發票、{r['checks_per_run']} 項檢查",
        f"  資料既有發現：{len(r['baseline_findings'])} 項"
        + (f"（{'、'.join(r['baseline_findings'])}）—— "
           f"乾淨資料上就會報，是對資料的正確描述，已排除於混淆矩陣之外"
           if r['baseline_findings'] else "（乾淨資料零發現）"),
        "", "─" * 92,
        f"  {'指標':<28}{'僅計可偵測樣態':>18}{'含已知抓不到樣態':>20}",
        "─" * 92,
    ]
    rows = [("混淆矩陣 TP / FP / TN / FN", None),
            ("Precision（說有問題的真的有嗎）", "precision"),
            ("Recall（有問題的抓到幾成）", "recall"),
            ("Specificity（乾淨的正確放行）", "specificity"),
            ("F1（調和平均）", "f1"),
            ("★ MCC（不平衡資料最可靠）", "mcc"),
            ("Accuracy（僅供對照，會誤導）", "accuracy")]
    for label, key in rows:
        if key is None:
            L.append(f"  {label:<28}"
                     f"{a['TP']}/{a['FP']}/{a['TN']}/{a['FN']:>10}"
                     f"{b['TP']}/{b['FP']}/{b['TN']}/{b['FN']:>12}")
            continue
        pct = key != "mcc"
        L.append(f"  {label:<28}{fmt(a[key], pct):>18}{fmt(b[key], pct):>20}")

    L += [
        "─" * 92,
        f"  成本加權（漏抓:誤報 = {a['cost_ratio']}）"
        f"{a['cost_normalized']:>18.3f}{b['cost_normalized']:>20.3f}",
        "",
        "  【怎麼讀這兩欄】",
        "    左欄只計算「設計上抓得到」的樣態 —— 這是系統在其能力範圍內的表現。",
        f"    右欄把 {r['known_undetectable_trials']} 次「已知抓不到」的試驗也計入 FN ——",
        "    這才是面對**全部已知造假手法**時的真實 recall。",
        "    兩欄都報，是因為只報左欄會讓 recall 虛高，只報右欄則看不出能力範圍。",
        "",
        "  【為什麼 MCC 是主指標而不是 Accuracy】",
        "    造假是稀有事件。一個永遠回答「全部乾淨」的系統，accuracy 會很高，",
        "    但它一張都沒抓到。MCC 是唯一同時考慮混淆矩陣四格、",
        "    且對類別不平衡穩健的係數（範圍 −1 ~ +1，0 等同亂猜）。",
        "═" * 92,
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20,
                    help="每種樣態的獨立試驗次數（不同隨機種子）")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tier", type=int, action="append",
                    help="只測指定 tier，可重複；預設全部")
    ap.add_argument("--out", default="docs/FRAUD_BENCHMARK.json")
    args = ap.parse_args()

    tiers = args.tier or [1, 2, 3, 4, 5]
    print("═" * 92)
    print("  造假偵測 benchmark")
    print("═" * 92 + "\n")
    r = run(args.trials, args.seed, tiers)
    print(render(r))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
