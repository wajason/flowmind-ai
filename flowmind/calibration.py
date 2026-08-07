"""
flowmind.calibration — 用真實統計校準合成資料
=============================================================================
【這支檔案回答一個我們被問過、而且答不出來會很難看的問題】

    「你們的合成資料憑什麼說它像真的？」

在此之前，合成器的產業分布、組織型態比例、金額量級都是開發者拍腦袋設的。
那種資料可以驗證「程式邏輯對不對」，但不能宣稱「像真實世界」。

現在知識庫裡有 21 份**信保基金真實承保統計**
（行業別／地區別／組織型態別，民國 115 年 1–7 月），
這是台灣中小企業融資市場的實際分布。合成器改為從這裡取權重。

【第一次校準就抓到兩個錯誤，值得記錄】

  1. **組織型態比例錯得離譜。**
     原本設定：股份有限公司 55% / 有限公司 35% / 企業社 10%
     真實統計：股份有限公司 48.71% / 有限公司 47.20% / 獨資或合夥 3.74%
     → 有限公司被嚴重低估。台灣中小企業以「有限公司」為大宗，
        這在任何一個熟悉中小企業的評審眼裡是常識，錯了會很扎眼。

  2. **產業覆蓋根本沒對到市場。**
     合成器只做製造業四類（精密機械／電子零組件／食品加工／紡織成衣）。
     真實承保金額中：**批發及零售業 47.60%**、製造業 26.83%。
     → 我們把最大的那一塊市場整個漏掉了。
        批發零售的帳期、周轉率、交易對手數量與製造業差異極大，
        這不只是資料問題，是產品覆蓋範圍的問題。

【設計原則】
校準只調「分布」，不編造「個案」。
真實統計告訴我們市場長什麼樣子，個別企業的交易明細仍然是合成的 ——
因為那本來就拿不到，而且拿到了也不能用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import config

STATS_DIR = config.RAW_DIR / "SHARED"

# 民國年月：11507 = 民國115年7月
_ROC = re.compile(r"(\d{3})(\d{2})\.xlsx$")


@dataclass
class Distribution:
    """一個維度的真實分布。weights 已正規化成總和 1.0。"""
    dimension: str
    period: str
    unit: str
    items: dict[str, float] = field(default_factory=dict)   # 名稱 → 比重(0~1)
    amounts: dict[str, float] = field(default_factory=dict)  # 名稱 → 承保金額(千元)
    total_amount: float = 0.0

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.items.items(), key=lambda kv: -kv[1])[:n]


def _latest(pattern: str) -> Optional[Path]:
    """取最新一期的統計檔。資料每月更新，永遠用最新的那份。"""
    files = sorted(STATS_DIR.glob(pattern))
    return files[-1] if files else None


# 行業標準分類的「大類」。統計表把大類與其下的中類混在同一欄，
# 例如「製造業 26.83%」後面接著「食品製造業 2.57%」「紡織業 0.50%」…
# 若不區分，大類與中類會被重複計算，把每一項的比重都稀釋掉
# （實測：製造業真實 26.83%，不分層時算成 15.38%）。
# 用明確清單而不是靠縮排或空白判斷 —— 政府統計表的縮排格式不保證穩定。
INDUSTRY_TOP_LEVEL = {
    "農、林、漁、牧業", "礦業及土石採取業", "製造業", "電力及燃氣供應業",
    "用水供應及污染整治業", "營造業", "批發及零售業", "運輸及倉儲業",
    "住宿及餐飲業", "出版影音及資通訊業", "金融及保險業", "不動產業",
    "專業、科學及技術服務業", "支援服務業", "公共行政及國防",
    "教育業", "醫療保健及社會工作服務業", "藝術、娛樂及休閒服務業",
    "其他服務業", "資訊及通訊傳播業",
}


def _is_number(x) -> bool:
    """float('nan') 不會拋錯，也不會被 <=0 擋掉 —— 必須明確排除。
    實測踩過：一個 NaN 混進權重計算，整份分布全部變成 nan%。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and v > 0            # v != v 就是 NaN


def _parse(path: Path, dimension: str) -> Optional[Distribution]:
    """
    解析信保基金統計表。版面固定：
      第 0~2 列標題、第 3 列單位、第 4 列欄位名、第 5 列起資料、最後一列「合計」。

    刻意用規則解析而不是丟給 LLM：版面穩定的政府統計表就該用程式讀，
    交給模型只會引入不必要的不確定性，而且每個月都要重跑一次。
    """
    import pandas as pd

    try:
        df = pd.read_excel(path, header=None)
    except Exception:                                  # noqa: BLE001
        return None

    period = ""
    for v in df[0].head(4):
        if isinstance(v, str) and "年" in v and "月" in v:
            period = v.strip()
    unit = ""
    for v in df[2].head(5):
        if isinstance(v, str) and "單位" in v:
            unit = v.split(":")[-1].strip()

    items, amounts, total = {}, {}, 0.0
    is_industry = dimension == "行業別"

    for _, row in df.iloc[5:].iterrows():
        name = str(row[0]).strip()
        if name in ("nan", "", "None"):
            continue
        if name == "合計":
            if _is_number(row[1]):
                total = float(row[1])
            continue
        if not _is_number(row[1]):
            continue                                   # 含該期無承保（0）與 NaN
        amt = float(row[1])
        amounts[name] = amt
        # 行業別表把「大類」與其下的「中類」混在同一欄。
        # 只取中類（葉節點）當權重，否則大類與中類相加會超過 100%，
        # 把每一項的比重都稀釋掉。大類的數字仍留在 amounts 供對照。
        if is_industry and name in INDUSTRY_TOP_LEVEL:
            continue
        items[name] = amt

    if not items:
        return None
    if total <= 0:
        total = sum(amounts.values())

    s = sum(items.values())
    items = {k: v / s for k, v in items.items()}
    return Distribution(dimension=dimension, period=period, unit=unit,
                        items=items, amounts=amounts, total_amount=total)


@lru_cache(maxsize=1)
def org_type_distribution() -> Optional[Distribution]:
    p = _latest("*組織型態別承保統計*.xlsx")
    return _parse(p, "組織型態別") if p else None


@lru_cache(maxsize=1)
def industry_distribution() -> Optional[Distribution]:
    p = _latest("*行業別承保統計*.xlsx")
    return _parse(p, "行業別") if p else None


@lru_cache(maxsize=1)
def region_distribution() -> Optional[Distribution]:
    p = _latest("*地區別承保統計*.xlsx")
    return _parse(p, "地區別") if p else None


# ══════════════════════════════════════════════════════════════════════════
# 給合成器用的權重
# ══════════════════════════════════════════════════════════════════════════

# 統計表的行業名稱 → 合成器的產業設定檔。
# 只對應得上的才用；對不上的（例如「農林漁牧」）代表我們目前沒有那個產業的
# 模型，會被 coverage_report() 列為未覆蓋，逼我們正視產品的覆蓋缺口。
INDUSTRY_MAP = {
    "機械設備製造業": "精密機械",
    "金屬製品製造業": "精密機械",
    "產業用機械設備維修及安裝業": "精密機械",
    "電子零組件製造業": "電子零組件",
    "電腦、電子產品及光學製品製造業": "電子零組件",
    "食品製造業": "食品加工",
    "飲料製造業": "食品加工",
    "紡織業": "紡織成衣",
    "成衣及服飾品製造業": "紡織成衣",
    "機械器具批發業": "機械器具批發",
    "建材批發業": "建材批發",
    "食品、飲料及菸草製品批發業": "食品批發",
}

# 統計表用「獨資或合夥」，公司登記實務上對應的名稱是「商行／企業社」
ORG_SUFFIX_MAP = {
    "股份有限公司": "股份有限公司",
    "有限公司": "有限公司",
    "獨資或合夥": "企業社",
}


def org_suffix_weights() -> dict[str, float]:
    """
    回傳公司名稱後綴的真實權重。取不到統計檔時回退到原本的猜測值，
    但呼叫端應該要知道自己拿到的是哪一種 —— 見 is_calibrated()。
    """
    d = org_type_distribution()
    if not d:
        return {"股份有限公司": 0.55, "有限公司": 0.35, "企業社": 0.10}
    out: dict[str, float] = {}
    for name, w in d.items.items():
        key = ORG_SUFFIX_MAP.get(name)
        if key:
            out[key] = out.get(key, 0.0) + w
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s else {}


def industry_weights() -> dict[str, float]:
    """回傳合成器產業別的真實權重（只含對應得上的行業）。"""
    d = industry_distribution()
    if not d:
        return {}
    out: dict[str, float] = {}
    for name, w in d.items.items():
        key = INDUSTRY_MAP.get(name)
        if key:
            out[key] = out.get(key, 0.0) + w
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s else {}


def is_calibrated() -> bool:
    return org_type_distribution() is not None and industry_distribution() is not None


# ══════════════════════════════════════════════════════════════════════════
# 覆蓋率報告：讓漏掉的市場無所遁形
# ══════════════════════════════════════════════════════════════════════════

def coverage_report() -> dict:
    """
    我們的合成器涵蓋了真實承保市場的多少比例？

    這個數字刻意做出來給自己難看：目前只做製造業四類，
    但真實承保金額中批發及零售業佔了將近一半。
    一份宣稱「貼近真實中小企業」的合成資料集，如果漏掉最大的那一塊市場，
    那個宣稱就是空的。
    """
    d = industry_distribution()
    if not d:
        return {"calibrated": False}

    covered = {k: v for k, v in d.items.items() if k in INDUSTRY_MAP}
    uncovered = sorted(((k, v) for k, v in d.items.items() if k not in INDUSTRY_MAP),
                       key=lambda kv: -kv[1])
    return {
        "calibrated": True,
        "period": d.period,
        "total_amount_ntd_thousand": d.total_amount,
        "covered_industries": len(covered),
        "covered_share": round(sum(covered.values()), 4),
        "uncovered_share": round(sum(v for _, v in uncovered), 4),
        "top_uncovered": [{"industry": k, "share": round(v, 4)} for k, v in uncovered[:8]],
    }


def render_report() -> str:
    org = org_type_distribution()
    ind = industry_distribution()
    reg = region_distribution()
    cov = coverage_report()

    L = ["═" * 76,
         "  合成資料校準基準：信保基金真實承保統計",
         "═" * 76]
    if not is_calibrated():
        L += ["", "  ❌ 找不到承保統計檔案，合成器將使用未校準的預設權重。",
              f"     請確認 {STATS_DIR} 下有 *承保統計*.xlsx", "═" * 76]
        return "\n".join(L)

    L += ["", f"  資料期別：{ind.period}　單位：{ind.unit}",
          f"  全體承保融資金額：{ind.total_amount:,.0f} 千元"
          f"（約 {ind.total_amount/1e6:,.0f} 億元／月）", ""]

    L += ["─" * 76, "  組織型態別（決定合成公司名稱的後綴分布）", "─" * 76]
    for k, v in org.top(6):
        L.append(f"    {k:<16}{v:>7.2%}")
    L += ["", "    ⚠️ 校準前的猜測值是 股份有限公司 55% / 有限公司 35% / 企業社 10%，",
          "       與真實分布明顯不符 —— 有限公司被嚴重低估。", ""]

    L += ["─" * 76, "  行業別 Top 12（決定產業分布）", "─" * 76]
    for k, v in ind.top(12):
        mark = "✅" if k in INDUSTRY_MAP else "  "
        L.append(f"  {mark}  {k:<26}{v:>7.2%}")

    L += ["", "─" * 76, "  地區別 Top 6", "─" * 76]
    for k, v in reg.top(6):
        L.append(f"    {k:<12}{v:>7.2%}")

    L += ["", "═" * 76,
          f"  合成器目前涵蓋 {cov['covered_industries']} 個行業，"
          f"占真實承保金額 {cov['covered_share']:.1%}",
          f"  ⚠️ 未涵蓋 {cov['uncovered_share']:.1%}，最大的幾塊：", ""]
    for u in cov["top_uncovered"][:5]:
        L.append(f"       {u['industry']:<26}{u['share']:>7.2%}")
    L += ["",
          "  這個數字刻意做出來給自己難看。一份宣稱「貼近真實中小企業」的資料集，",
          "  如果漏掉最大的那一塊市場，那個宣稱就是空的。",
          "  批發零售的帳期、周轉率與交易對手數量都與製造業差異極大 ——",
          "  這不只是資料問題，是產品覆蓋範圍的問題。",
          "═" * 76]
    return "\n".join(L)


if __name__ == "__main__":
    print(render_report())
