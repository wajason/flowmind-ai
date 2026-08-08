#!/usr/bin/env python3
"""
industry.py — 產業知識層（從真實統計推導，不是 LLM 生成）
=============================================================================
【為什麼這一層不能用 LLM 生成】

六層知識庫盤點時，「產業知識」是最薄的一層 ——
各產業的規模、出口依存度、風險特徵原本都是**推估**的。

補這一層有兩條路：

  ❌ 讓 LLM 寫一份「各產業特徵說明」
     生成出來的產業知識，正是我們整個產品在反對的東西。
     它讀起來很專業、每一句都無法查證，而且錯了不會有人發現。

  ✅ 從我們手上已有的**真實官方統計**把它算出來
     29 份中小企業處統計（2013 年起逐年、按行業別），
     每一個推導出來的數字都能回到原始檔案的某一列。

本模組走第二條路。**零 LLM。**

【嚴格區分「事實」與「判讀」】

這一層最容易犯的錯，是把「我們的解讀」混進「資料說的話」。
所以拆成兩個層次，而且在資料結構上就分開：

    IndustryProfile.facts        從統計直接算出來的，附來源檔案與年度
    IndustryProfile.implications 授信意涵，**是判讀**，附推理依據

判讀可以被反駁，事實不行 —— 兩者混在一起，就兩者都不可信了。

【推導出來的五項事實，以及它們為什麼與供應鏈金融有關】

    中小企業家數占比    這個產業是不是以中小企業為主
    平均每家銷售額      規模 → 對大買方的議價能力
    出口依存度          出口佔銷售額比重 → 帳期長度與匯率曝險
    內銷依存度          內需依賴程度
    每家受僱人數        勞力密集度 → 固定現金流出壓力

這五項都不是我挑出來讓數字好看的，
而是「應收帳款融資要看什麼」反推回來需要的欄位。
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

DATA_DIR = config.DATA_DIR / "raw" / "SHARED"

# 檔名 → 這份檔案提供什麼。刻意逐一列出而不是掃目錄：
# 掃目錄會在別人新增一份格式不同的 CSV 時安靜地算出錯的東西。
# 第三個元素是**換算成基本單位的倍率**。
# 這個欄位是被一個實際的 bug 逼出來的：受僱人數的單位是「千人」不是「人」，
# 一開始寫成「人」，算出來的「平均每家受僱人數」全部是 0.0 ——
# 而 0.0 不會拋錯，只會安靜地出現在報告裡。
# 把單位與倍率寫進資料結構，是為了讓下一次的單位錯誤變成看得見的東西。
SOURCES = {
    "firms":     ("企業行業別家數統計.csv",   "家",     1),
    "sales":     ("企業行業別銷售額統計.csv", "百萬元", 1),
    "exports":   ("企業行業別出口額統計.csv", "百萬元", 1),
    "domestic":  ("企業行業別內銷額統計.csv", "百萬元", 1),
    "employees": ("企業行業別受僱人數統計.csv", "千人", 1000),
}

# 「總計」不是一個產業，納入會讓所有比較失真
_NOT_AN_INDUSTRY = {"總計", "合計", "小計", ""}

# 各統計檔對同一個產業的寫法不一致：
#   家數統計    「農林漁牧業」
#   受僱人數    「農、林、漁、牧業」
# 直接用字串當鍵去跨檔比對，會安靜地對不上而回 None，
# 然後那個產業的欄位就會消失 —— 不會有任何錯誤訊息。
# 所以比對前一律正規化，並在載入時檢查跨檔覆蓋率（見 coverage_report）。
_PUNCT = "、，,．・･ 　()（）"


def _norm_industry(s: str) -> str:
    return "".join(ch for ch in (s or "").strip() if ch not in _PUNCT)


@dataclass
class IndustryProfile:
    industry: str
    year: int
    facts: dict = field(default_factory=dict)
    implications: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def render(self) -> str:
        L = [f"{self.industry}（{self.year} 年）", "─" * 62, "  【統計事實】"]
        for k, v in self.facts.items():
            L.append(f"    {k}：{v}")
        if self.implications:
            L.append("  【授信意涵 —— 這是判讀，不是資料本身】")
            for im in self.implications:
                L.append(f"    · {im['point']}")
                L.append(f"      依據：{im['basis']}")
        L.append("  【資料來源】")
        for k, v in self.provenance.items():
            L.append(f"    {k} ← {v}")
        return "\n".join(L)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        # 不靜默回空 —— 一個「查無資料」與「檔案不見了」長得一樣的系統，
        # 會讓人以為產業知識層沒有涵蓋，其實是檔案掉了
        raise FileNotFoundError(f"產業統計檔案不存在：{path}")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if not rows:
        raise ValueError(f"產業統計檔案是空的：{path}")
    return rows


def _num(v) -> Optional[float]:
    if v in (None, "", "-", "…"):
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _sme_col(row: dict) -> Optional[float]:
    """取「中小企業」那一欄。欄名帶單位（家/百萬元/人），所以用前綴比對。"""
    for k, v in row.items():
        if k and k.startswith("中小企業"):
            return _num(v)
    return None


def _all_col(row: dict) -> Optional[float]:
    for k, v in row.items():
        if k and k.startswith("全部企業"):
            return _num(v)
    return None


def load_series() -> dict[str, dict[tuple[str, int], dict]]:
    """
    讀進所有統計，索引成 {指標: {(產業, 年度): {sme, all}}}。
    """
    out: dict[str, dict[tuple[str, int], dict]] = {}
    for key, (fname, _unit, scale) in SOURCES.items():
        table: dict[tuple[str, int], dict] = {}
        for row in _load(DATA_DIR / fname):
            raw_ind = (row.get("行業別") or "").strip()
            ind = _norm_industry(raw_ind)
            if raw_ind in _NOT_AN_INDUSTRY or ind in _NOT_AN_INDUSTRY:
                continue
            try:
                year = int(str(row.get("年度") or "").strip())
            except ValueError:
                continue
            sme, all_ = _sme_col(row), _all_col(row)
            table[(ind, year)] = {
                "sme": None if sme is None else sme * scale,
                "all": None if all_ is None else all_ * scale,
                "display": raw_ind,
            }
        out[key] = table
    return out


def coverage_report(series: Optional[dict] = None) -> dict:
    """
    跨檔覆蓋率自檢：有多少產業在某些統計裡查得到、在另一些裡查不到。

    這個函式存在的理由，是因為跨檔比對失敗**不會拋錯**，
    只會讓某個欄位安靜地消失。一個「產業知識層」如果有一半的產業
    缺欄位而沒有人知道，它就只是看起來存在。
    """
    series = series or load_series()
    keys = list(SOURCES)
    base_year = max({y for (_i, y) in series["firms"]})
    per_source = {k: {i for (i, y) in series[k] if y == base_year} for k in keys}
    union: set[str] = set().union(*per_source.values())
    inter: set[str] = set(union)
    for s in per_source.values():
        inter &= s
    missing = {k: sorted(union - per_source[k]) for k in keys
               if union - per_source[k]}
    return {
        "year": base_year,
        "industries_in_any_source": len(union),
        "industries_in_all_sources": len(inter),
        "full_coverage_rate": round(len(inter) / len(union), 4) if union else 0.0,
        "missing_by_source": missing,
    }


def available_years(series: Optional[dict] = None) -> list[int]:
    series = series or load_series()
    return sorted({y for (_i, y) in series["firms"]})


def industries(series: Optional[dict] = None) -> list[str]:
    series = series or load_series()
    return sorted({i for (i, _y) in series["firms"]})


def profile(industry: str, year: Optional[int] = None,
            series: Optional[dict] = None) -> IndustryProfile:
    """
    算出某產業某年度的側寫。

    year 省略時取**資料中最新的年度**，而不是寫死一個數字 ——
    寫死的年份會在資料更新後安靜地繼續回答舊數字。
    """
    series = series or load_series()
    year = year or max(available_years(series))
    key_ind = _norm_industry(industry)
    display = ((series["firms"].get((key_ind, year)) or {}).get("display")
               or industry)
    p = IndustryProfile(industry=display, year=year)

    def g(key: str, which: str = "sme") -> Optional[float]:
        return (series[key].get((key_ind, year)) or {}).get(which)

    firms, sales = g("firms"), g("sales")
    exports, domestic, emp = g("exports"), g("domestic"), g("employees")
    firms_all = g("firms", "all")

    # ── 統計事實 ──────────────────────────────────────────────────────
    if firms is not None and firms_all:
        p.facts["中小企業家數占比"] = f"{100.0 * firms / firms_all:.2f}%"
    if firms:
        p.facts["中小企業家數"] = f"{firms:,.0f} 家"
    if sales is not None and firms:
        # 統計單位是百萬元 → 換算成萬元比較好讀
        p.facts["平均每家銷售額"] = f"{sales * 100 / firms:,.0f} 萬元"
    if exports is not None and sales:
        p.facts["出口依存度"] = f"{100.0 * exports / sales:.1f}%"
    if domestic is not None and sales:
        p.facts["內銷依存度"] = f"{100.0 * domestic / sales:.1f}%"
    if emp is not None and firms:
        p.facts["平均每家受僱人數"] = f"{emp / firms:.1f} 人"

    p.provenance = {k: f"{v[0]}（{year} 年，原始單位 {v[1]}）"
                    for k, v in SOURCES.items()}

    # ── 授信意涵（判讀，不是資料本身）─────────────────────────────────
    # 每一條都附上「依據」，讓它可以被反駁。
    # 一條無法被反駁的判讀，和一句廢話沒有差別。
    if exports is not None and sales and sales > 0:
        ratio = 100.0 * exports / sales
        # 30% 只是一個敘述性的分界，用來區分「以外銷為主」與否，
        # 不是風險門檻。刻意不拿它去做任何自動決策。
        if ratio >= 30.0:
            p.implications.append({
                "point": "以外銷為主，應收帳期通常長於內銷，"
                         "並存在匯率與跨境求償風險",
                "basis": f"出口額占銷售額 {ratio:.1f}%（≥30% 視為外銷導向）。"
                         f"國際貿易的信用狀與海運週期使帳期自然拉長，"
                         f"這是結構性因素，不是個別企業的信用問題。",
            })
        else:
            p.implications.append({
                "point": "以內需為主，帳期較短但景氣連動性高",
                "basis": f"出口額占銷售額僅 {ratio:.1f}%。"
                         f"內需產業的應收品質與國內景氣高度同步。",
            })
    if sales is not None and firms:
        avg = sales * 100 / firms          # 萬元
        if avg < 2000:
            p.implications.append({
                "point": "平均規模偏小，面對大型買方時議價能力弱，"
                         "帳期被單方面拉長的風險較高",
                "basis": f"平均每家銷售額 {avg:,.0f} 萬元。"
                         f"這正是供應鏈金融要解決的核心問題 —— "
                         f"用買方信用替賣方取得資金。",
            })
    if emp is not None and firms and emp / firms >= 10:
        p.implications.append({
            "point": "勞力密集，人事支出是剛性現金流出，"
                     "對應收延遲的耐受度較低",
            "basis": f"平均每家受僱 {emp / firms:.1f} 人。"
                     f"薪資無法延後支付，因此收款延遲會直接轉成流動性壓力。",
        })

    if not p.facts:
        # 查無此產業時明說，不要回一個空殼讓人以為「這個產業沒有特徵」
        raise KeyError(
            f"統計中沒有「{industry}」{year} 年的資料。"
            f"可用產業：{'、'.join(industries(series)[:8])}…")
    return p


def compare(industries_: list[str], year: Optional[int] = None) -> str:
    """跨產業比較表。供應鏈金融的授信判斷經常需要「相對於同業如何」。"""
    series = load_series()
    year = year or max(available_years(series))
    rows = []
    for ind in industries_:
        try:
            p = profile(ind, year, series)
        except KeyError:
            continue
        rows.append((ind, p.facts.get("平均每家銷售額", "—"),
                     p.facts.get("出口依存度", "—"),
                     p.facts.get("平均每家受僱人數", "—")))
    if not rows:
        return "查無可比較的產業。"
    L = [f"產業比較（{year} 年，資料來源：中小企業處統計）", "─" * 74,
         f"  {'產業':<20}{'平均銷售額':>14}{'出口依存度':>12}{'平均受僱':>10}"]
    for r in rows:
        L.append(f"  {r[0]:<20}{r[1]:>14}{r[2]:>12}{r[3]:>10}")
    return "\n".join(L)


__all__ = ["IndustryProfile", "profile", "compare", "industries",
           "available_years", "load_series", "SOURCES"]
