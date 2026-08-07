#!/usr/bin/env python3
"""
roi_model.py — 導入效益試算（參數化、可被質疑、可被重算）
=============================================================================
【為什麼這支程式存在，而不是在提案書寫一個數字】

「每案人工核對時數」是 ROI 公式的關鍵參數，但它有三個性質：

  1. **查不到官方數字。** 案件處理時間（TAT）屬於銀行內部營運機密，
     不會出現在官網或商品說明書上。
  2. **它本來就不是固定值。** 每家銀行的系統、SOP、人員熟練度都不同，
     甚至同一家銀行的不同分行都不一樣。
  3. **它決定這套系統值不值得買。** 如果只能省 5 分鐘，硬體與維運成本就回不了本。

面對這種參數，有三種做法：

  ❌ 編一個看起來合理的數字，寫成「實測值」
     → 評審問「哪裡來的」答不出來，信任瞬間崩掉。
        而且這正是我們整套系統在防範的事：一個聽起來權威、查無實據的數字。

  ❌ 引用二手網站整理的「業界平均 2-3 小時」
     → 那些數字同樣沒有可查證的一手出處。用它等於把問題往上推一層。

  ✅ **拆成可檢查的組成，讓每個假設都能被單獨質疑與替換**
     → 這支程式的做法。銀行拿到後可以直接帶入自己的數字重算，
        不需要接受我們的任何一個假設。

【誠實聲明】
下方所有預設值都是 **bottom-up 推算的假設值，不是實測值**。
具名來源（IOFM / McKinsey / Deloitte）僅作為「數量級方向性佐證」，
且我們是透過二手整理取得，未直接讀原始報告 —— 這一點也如實標註。

自動化比例是唯一的例外：它不是假設，是從 VeriFin 的 Risk-Coverage
曲線**實際量測**出來的（見 --measured-coverage 參數）。

Usage:
    python scripts/roi_model.py                      # 預設情境
    python scripts/roi_model.py --invoices-per-case 200 --hourly-cost 900
    python scripts/roi_model.py --scenario conservative
    python scripts/roi_model.py --show-assumptions   # 只列假設與出處
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind import config                                    # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 假設與出處：每一個數字都必須說得出從哪來
# ══════════════════════════════════════════════════════════════════════════
ASSUMPTIONS = [
    {
        "param": "min_per_invoice",
        "label": "單張憑證人工核對時間（分鐘）",
        "default": 3.0,
        "basis": "bottom-up 推算",
        "reasoning": (
            "核對一張發票需要：讀出買方統編與名稱、核對金額加總、"
            "確認帳期與到期日、與合約比對、在銀行流水中找對應入帳。"
            "以每項約 30~40 秒估計，合計約 3 分鐘。"),
        "directional_support": (
            "IOFM（Institute of Finance & Management）指出人工處理單張發票"
            "平均需 5~20 分鐘（依複雜度）。我們取 3 分鐘是**保守**的下緣，"
            "因為 IOFM 的數字包含建檔與付款流程，而我們只算核對。"),
        "source_caveat": (
            "⚠️ IOFM 數字係透過二手整理網站取得，未直接讀原始報告。"
            "僅作為數量級方向性佐證，不作為實測值引用。"),
    },
    {
        "param": "min_cross_document",
        "label": "跨文件比對與異常追查（分鐘／案）",
        "default": 45.0,
        "basis": "bottom-up 推算",
        "reasoning": (
            "與單張核對不同，這部分是全案層級的工作：買方集中度計算、"
            "帳齡分析、找出重複請款、確認沒有自我交易、"
            "以及對任何一項異常的追查與電話確認。"
            "以一案 3~5 項需要追查估計，每項約 10 分鐘。"),
        "directional_support": "無直接對應的公開基準。這是本模型不確定性最高的一項。",
        "source_caveat": "⚠️ 純推算，無外部佐證。銀行導入評估時應優先替換此值。",
    },
    {
        "param": "min_report_writing",
        "label": "撰寫核對紀錄與簽核意見（分鐘／案）",
        "default": 30.0,
        "basis": "bottom-up 推算",
        "reasoning": (
            "授信案件需留下可供內稽查核的書面紀錄。"
            "本系統的稽核軌跡與證據包可直接輸出，這部分節省比例最高。"),
        "directional_support": (
            "McKinsey Global Institute 指出導入貿易融資自動化的組織回報"
            "處理時間最高可減少 75%；Deloitte 指出自動化文件處理可將"
            "週期縮短 75~90%。"),
        "source_caveat": "⚠️ 同上，二手取得，僅供方向性參考。",
    },
    {
        "param": "invoices_per_case",
        "label": "每案憑證張數",
        "default": 90,
        "basis": "本專案合成資料集規模",
        "reasoning": (
            "以 24 個月、每月 3~4 張發票的中小企業為模型，"
            "送件時銀行通常要求近 1~2 年的往來憑證。"),
        "directional_support": (
            "已用真實資料部分驗證：政府採購決標資料中，"
            "單一廠商跨年度得標筆數的量級與此相符。"),
        "source_caveat": "此值差異極大，製造業與服務業可差 5 倍以上。",
    },
    {
        "param": "hourly_cost",
        "label": "授信人員全負擔時薪（元）",
        "default": 800.0,
        "basis": "bottom-up 推算",
        "reasoning": (
            "以年薪 100 萬（含勞健保、年終、辦公空間等全負擔成本約 1.4 倍）"
            "、年工時 1,800 小時估算：1,000,000 × 1.4 ÷ 1,800 ≈ 780 元。"),
        "directional_support": "可由銀行以自身薪酬結構直接替換，此值最容易校準。",
        "source_caveat": "非實際薪資調查，僅為公開可得資訊的粗略推算。",
    },
    {
        "param": "cases_per_month",
        "label": "每月受理案件數",
        "default": 100,
        "basis": "情境假設",
        "reasoning": "以一個中型分行企金團隊的量體估計。",
        "directional_support": "此值由導入單位直接提供，不需推算。",
        "source_caveat": "純情境參數。",
    },
    {
        "param": "automation_ratio",
        "label": "可自動化比例",
        "default": 0.60,
        "basis": "★ 實際量測（非假設）",
        "reasoning": (
            "來自 VeriFin 的 Risk-Coverage 曲線："
            "在錯誤率壓到 5% 以下的前提下，系統能自動處理的欄位比例。"
            "其餘一律轉人工。這是本模型中唯一**量測**而非推估的參數，"
            "而且可以寫進導入合約作為驗收標準。"),
        "directional_support": "可用 python scripts/run_verifin.py 重跑驗證。",
        "source_caveat": (
            "目前的量測樣本僅 12 份文件，且為域外 benchmark（SROIE）。"
            "正式導入前必須以客戶自己的文件重新量測。"),
    },
]

DEFAULTS = {a["param"]: a["default"] for a in ASSUMPTIONS}

# 三個情境。刻意讓「保守」情境也要能站得住 ——
# 一套只有在最樂觀假設下才划算的系統，不值得買。
SCENARIOS = {
    "conservative": {"min_per_invoice": 1.5, "min_cross_document": 20.0,
                     "min_report_writing": 15.0, "automation_ratio": 0.40},
    "base": {},
    "optimistic": {"min_per_invoice": 5.0, "min_cross_document": 70.0,
                   "min_report_writing": 45.0, "automation_ratio": 0.75},
}


@dataclass
class RoiResult:
    manual_hours_per_case: float
    saved_hours_per_case: float
    saved_hours_per_month: float
    saved_cost_per_month: float
    saved_cost_per_year: float
    fte_equivalent: float
    params: dict


def compute(p: dict) -> RoiResult:
    """
    完全透明的四則運算。刻意不做任何黑箱加權 ——
    一個銀行看不懂也算不出來的 ROI 數字，在採購會議上沒有任何說服力。
    """
    manual_min = (p["min_per_invoice"] * p["invoices_per_case"]
                  + p["min_cross_document"] + p["min_report_writing"])
    manual_h = manual_min / 60.0
    saved_h = manual_h * p["automation_ratio"]
    saved_month = saved_h * p["cases_per_month"]
    cost_month = saved_month * p["hourly_cost"]
    return RoiResult(
        manual_hours_per_case=round(manual_h, 2),
        saved_hours_per_case=round(saved_h, 2),
        saved_hours_per_month=round(saved_month, 1),
        saved_cost_per_month=round(cost_month),
        saved_cost_per_year=round(cost_month * 12),
        # 一個 FTE 以每月 150 個有效工時計
        fte_equivalent=round(saved_month / 150.0, 2),
        params=p,
    )


def show_assumptions() -> None:
    print("═" * 78)
    print("  ROI 模型的全部假設與出處")
    print("═" * 78)
    print("\n⚠️  除了「可自動化比例」以外，以下全部是 bottom-up 推算的**假設值**，")
    print("    不是實測值。任何一項都可以被質疑、被替換。\n")
    for a in ASSUMPTIONS:
        star = "★ " if a["basis"].startswith("★") else ""
        print("─" * 78)
        print(f"  {star}{a['label']}　預設 {a['default']}")
        print(f"  依據類型：{a['basis']}")
        print(f"  推算邏輯：{a['reasoning']}")
        print(f"  方向性佐證：{a['directional_support']}")
        print(f"  {a['source_caveat']}")
    print("═" * 78)


def render(r: RoiResult, scenario: str) -> str:
    p = r.params
    L = [
        "═" * 78,
        f"  導入效益試算　情境：{scenario}",
        "═" * 78,
        "",
        "【輸入參數】（全部可調，銀行可帶入自己的數字重算）",
        f"  每案憑證張數              {p['invoices_per_case']:>10,} 張",
        f"  單張人工核對              {p['min_per_invoice']:>10.1f} 分鐘",
        f"  跨文件比對與異常追查      {p['min_cross_document']:>10.1f} 分鐘／案",
        f"  撰寫核對紀錄              {p['min_report_writing']:>10.1f} 分鐘／案",
        f"  授信人員全負擔時薪        {p['hourly_cost']:>10,.0f} 元",
        f"  每月受理案件數            {p['cases_per_month']:>10,} 件",
        f"  可自動化比例（★實測）     {p['automation_ratio']:>10.0%}",
        "",
        "【推算結果】",
        f"  單案人工核對工時          {r.manual_hours_per_case:>10.2f} 小時",
        f"  單案可節省                {r.saved_hours_per_case:>10.2f} 小時",
        f"  每月可節省                {r.saved_hours_per_month:>10.1f} 小時"
        f"（約 {r.fte_equivalent} 個 FTE）",
        f"  每月可節省人力成本        {r.saved_cost_per_month:>10,} 元",
        f"  **年化效益**              {r.saved_cost_per_year:>10,} 元",
        "",
        "─" * 78,
        "  ⚠️  除「可自動化比例」外，上列參數皆為 bottom-up 推算的假設值，非實測值。",
        "      完整推算邏輯與出處：python scripts/roi_model.py --show-assumptions",
        "      導入評估時應優先替換「跨文件比對時間」—— 這是本模型不確定性最高的一項。",
        "═" * 78,
    ]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="FlowMind 導入效益試算")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="base")
    for a in ASSUMPTIONS:
        ap.add_argument(f"--{a['param'].replace('_', '-')}", type=float,
                        default=None, help=f"{a['label']}（預設 {a['default']}）")
    ap.add_argument("--show-assumptions", action="store_true")
    ap.add_argument("--all-scenarios", action="store_true", help="三種情境一次比較")
    ap.add_argument("--json", help="輸出 JSON 到指定路徑")
    args = ap.parse_args()

    if args.show_assumptions:
        show_assumptions()
        return

    def build(scn: str) -> dict:
        p = dict(DEFAULTS)
        p.update(SCENARIOS[scn])
        for a in ASSUMPTIONS:
            v = getattr(args, a["param"], None)
            if v is not None:
                p[a["param"]] = v
        p["invoices_per_case"] = int(p["invoices_per_case"])
        p["cases_per_month"] = int(p["cases_per_month"])
        return p

    if args.all_scenarios:
        results = {}
        print("═" * 78)
        print("  三情境敏感度比較")
        print("═" * 78)
        print(f"\n{'情境':<14}{'單案工時':>10}{'月省工時':>12}{'年化效益':>16}{'FTE':>8}")
        print("─" * 78)
        for scn in SCENARIOS:
            r = compute(build(scn))
            results[scn] = asdict(r)
            print(f"{scn:<14}{r.manual_hours_per_case:>10.2f}"
                  f"{r.saved_hours_per_month:>12.1f}"
                  f"{r.saved_cost_per_year:>16,}{r.fte_equivalent:>8.2f}")
        print("─" * 78)
        print("  判讀：即使在保守情境下，年化效益仍應顯著高於系統的建置與維運成本，")
        print("        這套系統才值得買。只有在樂觀情境才划算的系統，不應該賣。")
        print("═" * 78)
        if args.json:
            Path(args.json).write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    r = compute(build(args.scenario))
    print(render(r, args.scenario))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(r), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
