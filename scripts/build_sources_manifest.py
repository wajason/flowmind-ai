#!/usr/bin/env python3
"""
build_sources_manifest.py — 資料來源清單與新鮮度管理
=============================================================================
【這支腳本解決 RAG 系統一個必然會遇到、但很少被正面處理的問題：資料會過期。】

信保基金的作業手冊每年改版、保證要點會修正、承保統計每月更新。
知識庫裡同時存在 2015 年英文版作業手冊與 2025 年公告版是正常的
（歷史版本有查考價值），但**回答問題時必須優先採用現行版本**。

如果不管這件事，系統會很自然地引用一份 2015 年的舊規定回答 2026 年的問題，
而且引用驗證會顯示「✅ exact 100 分」—— 因為那句話確實在那份文件裡。
**引用是真的，答案是錯的。** 這是引用驗證擋不住的一類錯誤，
必須靠資料層的版本管理來處理。

做三件事：
  1. 掃描 data/raw/SHARED，比對登錄表，產生 data/SOURCES.md（給人看）
  2. 產生 data/sources_registry.json（給 data_update_finance.py 讀，寫進 metadata）
  3. **列出沒有登錄的檔案** —— 讓「有檔案但沒人知道它是什麼」這件事無所遁形

Usage:
    python scripts/build_sources_manifest.py
    python scripts/build_sources_manifest.py --check   # 只檢查，有未登錄檔案就回傳非 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind import config                                    # noqa: E402

SHARED = config.RAW_DIR / "SHARED"
OUT_MD = config.DATA_DIR / "SOURCES.md"
OUT_JSON = config.DATA_DIR / "sources_registry.json"

CURRENT = "current"          # 現行有效
SUPERSEDED = "superseded"    # 已被新版取代，保留供查考
REFERENCE = "reference"      # 背景參考，非作答依據

# ══════════════════════════════════════════════════════════════════════════
# 規則式登錄表
#
# 用規則而不是逐檔列舉，是因為知識庫會持續長大（承保統計每月一批）。
# 逐檔維護的清單三個月後一定會跟現實脫節，那就失去意義了。
# 規則配對不到的檔案會被明確列為「未登錄」，逼人補上。
# ══════════════════════════════════════════════════════════════════════════
RULES: list[dict] = [
    # ── 法規（正式法源，作答時權重最高）────────────────────────────────
    {"match": r"^中小企業發展條例", "category": "法規", "publisher": "全國法規資料庫",
     "status": CURRENT, "authority": 1,
     "desc": "中小企業融資、信保基金撥款的法源依據。第 35 條為專案融資與信保之核心條文。"},
    {"match": r"^中小企業認定標準", "category": "法規", "publisher": "全國法規資料庫",
     "status": CURRENT, "authority": 1,
     "desc": "界定何謂中小企業（實收資本額 1 億元以下或員工未滿 200 人），決定客戶能否適用信保方案。"},
    {"match": r"^民法債編", "category": "法規", "publisher": "全國法規資料庫",
     "status": CURRENT, "authority": 1,
     "desc": "應收帳款承購的法律本質是債權讓與，第 294–299 條為成立與對抗要件的依據。"},
    {"match": r"營業稅法", "category": "法規", "publisher": "全國法規資料庫",
     "status": CURRENT, "authority": 1,
     "desc": "5% 營業稅率與統一發票開立規定，是交叉驗證 ARITH-02 稅率檢查的規則來源。"},
    {"match": r"^商業會計法", "category": "法規", "publisher": "全國法規資料庫",
     "status": CURRENT, "authority": 1,
     "desc": "會計憑證保存年限（第 38 條），對應本系統 engagement 的 retention_until 設計。"},
    {"match": r"^個人資料保護法", "category": "法規", "publisher": "全國法規資料庫",
     "status": CURRENT, "authority": 1,
     "desc": "特定目的消失後應刪除資料，是多委任案隔離與保存期限設計的法遵依據。"},

    # ── 信保基金：保證要點與作業手冊 ──────────────────────────────────
    {"match": r"供應商融資", "category": "融資商品說明", "publisher": "中小企業信用保證基金",
     "status": CURRENT, "authority": 1,
     "desc": "★本專案的直接制度依據。明定供應商得憑訂單、發票（含電子發票）、支票、"
             "預約付款通知及其他得以佐證交易真實性之文件申請撥貸，保證成數最高九成。"},
    {"match": r"直接信用保證要點", "category": "融資商品說明", "publisher": "中小企業信用保證基金",
     "status": CURRENT, "authority": 2,
     "desc": "企業直接向信保基金申請保證的條件，與透過金融機構的間接保證有別。"},
    {"match": r"企業相對保證專案", "category": "融資商品說明", "publisher": "中小企業信用保證基金",
     "status": CURRENT, "authority": 2,
     "desc": "相對保證專案的申請條件與成數，供跨方案比較使用。"},
    {"match": r"2025年作業手冊_公告版", "category": "融資商品說明",
     "publisher": "中小企業信用保證基金", "status": CURRENT, "authority": 1,
     "published": "2025", "supersedes": ["2015 Operation Munnal(Taiwan SMEG).pdf"],
     "desc": "★現行版作業手冊。保證成數、手續費率、送件流程、代位清償的完整作業規範。"
             "回答作業程序類問題時應優先採用本版。"},
    {"match": r"2025年作業手冊修正對照表", "category": "融資商品說明",
     "publisher": "中小企業信用保證基金", "status": CURRENT, "authority": 2,
     "published": "2025",
     "desc": "2025 年版相對於前版的逐條修正對照，可用來判斷某條規定何時改的、改了什麼。"},
    {"match": r"2025年版作業手冊待修訂彙整表", "category": "融資商品說明",
     "publisher": "中小企業信用保證基金", "status": REFERENCE, "authority": 3,
     "published": "2025",
     "desc": "尚未定案的待修訂項目彙整。**不得作為現行規定引用**，僅供了解制度變動方向。"},
    {"match": r"Operation Munnal|Operation Manual", "category": "融資商品說明",
     "publisher": "Taiwan SMEG", "status": SUPERSEDED, "authority": 4,
     "published": "2015", "superseded_by": "信保基金-2025年作業手冊_公告版.pdf",
     "desc": "2015 年英文版作業手冊。**已被 2025 年公告版取代**，保留供國際交流與歷史查考。"},
    {"match": r"認識信保基金|保證對象資格", "category": "融資商品說明",
     "publisher": "中小企業信用保證基金", "status": CURRENT, "authority": 2,
     "desc": "保證對象資格判定與業務規章總覽，適合回答「我這家公司能不能送保」。"},
    {"match": r"保證規劃一字", "category": "融資商品說明",
     "publisher": "中小企業信用保證基金", "status": CURRENT, "authority": 2,
     "desc": "保證業務相關函文附件，內容為特定專案的補充規定。"},

    # ── 信保基金：真實承保統計（合成資料的校準基準）────────────────────
    {"match": r"承保統計(\d{5})", "category": "承保統計",
     "publisher": "中小企業信用保證基金", "status": CURRENT, "authority": 2,
     "desc": "信保基金**真實**承保統計月報。這是合成資料的校準基準 —— "
             "產業別分布、平均保證金額、件數量級都應該對得上這份真實統計。"},

    # ── 治理規範（與融資問答無關，需與商品說明分開避免污染檢索）────────
    {"match": r"誠信經營規範|工作規則|迴避|組織與職掌", "category": "治理規範",
     "publisher": "中小企業信用保證基金", "status": REFERENCE, "authority": 4,
     "desc": "信保基金內部治理與利益衝突迴避規範。與融資商品條件無關，"
             "但可佐證本系統『資訊隔離牆』設計對應到真實機構的法遵要求。"},

    # ── 銀行商品 ────────────────────────────────────────────────────
    {"match": r"（中國信託）|中國信託", "category": "融資商品說明", "publisher": "中國信託商業銀行",
     "status": CURRENT, "authority": 3,
     "desc": "商業銀行實際商品說明：應收帳款承購與供應鏈融資的申請條件與流程。"},
    {"match": r"^玉山銀行", "category": "融資商品說明", "publisher": "玉山商業銀行",
     "status": CURRENT, "authority": 3,
     "desc": "應收帳款承購（factoring）商品說明，含有／無追索權的差異。"},
    {"match": r"^永豐銀行", "category": "融資商品說明", "publisher": "永豐商業銀行",
     "status": CURRENT, "authority": 3,
     "desc": "應收帳款承購流程與所需文件清單，直接對應本系統的『送件前檢核』。"},

    # ── 白皮書與統計 ─────────────────────────────────────────────────
    {"match": r"^(\d{4})年?中小企業白皮書", "category": "白皮書統計",
     "publisher": "經濟部中小及新創企業署", "status": CURRENT, "authority": 3,
     "desc": "年度中小企業營運動向、財務概況與資金融通統計，含融資政策措施彙整。"},
    {"match": r"^(\d{4})新創企業白皮書", "category": "白皮書統計",
     "publisher": "經濟部中小及新創企業署", "status": CURRENT, "authority": 3,
     "desc": "新創企業年度統計與政策措施，涵蓋早期企業的資金取得管道。"},
    {"match": r"統計\.csv$|統計資料.*\.csv$|^產業部門|^女性企業|^新設企業|^各縣市|^企業行業|^製造業|^年度中小企業統計表",
     "category": "白皮書統計", "publisher": "經濟部中小及新創企業署",
     "status": CURRENT, "authority": 3,
     "desc": "中小企業統計表（行業別／縣市別／規模別／女性企業／新設企業），"
             "用於市場規模論證與合成資料的產業分布校準。"},
    {"match": r"電子發票.*建置指引", "category": "法規", "publisher": "財政部",
     "status": CURRENT, "authority": 1,
     "desc": "★電子發票 B2B 訊息規格。本系統合成發票的欄位命名（InvoiceNumber / "
             "SellerBAN / BuyerBAN / SalesAmount / TaxAmount / TotalAmount）即依此規格。"},
    {"match": r"融資輔導|中小及新創企業署", "category": "融資商品說明",
     "publisher": "經濟部中小及新創企業署", "status": CURRENT, "authority": 3,
     "desc": "政府融資輔導資源與政策工具索引。"},
]

# 民國年月（承保統計 11501 = 民國115年1月）
_ROC_YM = re.compile(r"(\d{3})(\d{2})$")


def infer_published(stem: str, rule: dict) -> str | None:
    """從檔名推斷發布日期。推不出來就回 None —— 不猜。"""
    if rule.get("published"):
        return rule["published"]
    m = _ROC_YM.search(stem)
    if m:
        y, mo = int(m.group(1)) + 1911, int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y}-{mo:02d}"
    m = re.search(r"(20\d{2})", stem)
    return m.group(1) if m else None


def match_rule(name: str) -> dict | None:
    for r in RULES:
        if re.search(r["match"], name):
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只檢查；有未登錄檔案時以非 0 結束（供 CI 使用）")
    args = ap.parse_args()

    files = sorted(p for p in SHARED.iterdir()
                   if p.is_file() and not p.name.startswith("_"))

    entries, unregistered = [], []
    for p in files:
        rule = match_rule(p.name)
        if rule is None:
            unregistered.append(p.name)
            continue
        entries.append({
            "filename": p.name,
            "category": rule["category"],
            "publisher": rule["publisher"],
            "published": infer_published(p.stem, rule),
            "status": rule["status"],
            # authority 1=法源/制度正本，2=主管機關規範，3=業者說明/統計，4=背景參考
            # 檢索時同分則優先採用 authority 較高、published 較新者
            "authority": rule["authority"],
            "superseded_by": rule.get("superseded_by"),
            "description": rule["desc"],
            "size_kb": round(p.stat().st_size / 1024),
        })

    # ── JSON（給 data_update_finance.py 讀，寫進 chunk metadata）────────
    OUT_JSON.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"),
         "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Markdown（給人看）──────────────────────────────────────────────
    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        by_cat.setdefault(e["category"], []).append(e)

    order = ["法規", "融資商品說明", "承保統計", "白皮書統計", "治理規範"]
    cats = [c for c in order if c in by_cat] + [c for c in by_cat if c not in order]

    L = [
        "# 資料來源清單（SOURCES）",
        "",
        f"> 由 `scripts/build_sources_manifest.py` 自動產生於 {date.today()}。",
        f"> 共 **{len(entries)}** 份已登錄文件"
        + (f"，**{len(unregistered)}** 份未登錄。" if unregistered else "，全部已登錄。"),
        "",
        "## 為什麼需要這份清單",
        "",
        "RAG 系統有一個引用驗證擋不住的錯誤類型：**引用是真的，但答案過期了。**",
        "",
        "系統可以完全正確地引用一份 2015 年的舊規定來回答 2026 年的問題，",
        "而且引用驗證會顯示「✅ exact 100 分」—— 因為那句話確實在那份文件裡。",
        "這必須靠資料層的版本管理處理，不是靠模型變聰明。",
        "",
        "因此每份文件都標註：",
        "",
        "| 欄位 | 意義 |",
        "|---|---|",
        "| **狀態** | `current` 現行有效／`superseded` 已被取代（保留查考）／`reference` 背景參考，不得作為作答依據 |",
        "| **權威層級** | 1 = 法源與制度正本｜2 = 主管機關規範｜3 = 業者說明與統計｜4 = 背景參考 |",
        "| **發布時間** | 檢索同分時優先採用較新版本 |",
        "",
        "---",
        "",
    ]

    for c in cats:
        rows = sorted(by_cat[c], key=lambda x: (x["authority"], x["filename"]))
        L += [f"## {c}（{len(rows)} 份）", "",
              "| 檔名 | 發布 | 狀態 | 權威 | 發布機關 | 一句話說明 |",
              "|---|---|---|---|---|---|"]
        for e in rows:
            icon = {"current": "✅", "superseded": "⚠️ 已取代", "reference": "📎 參考"}[e["status"]]
            sup = f"<br>→ 已由 `{e['superseded_by']}` 取代" if e["superseded_by"] else ""
            L.append(f"| `{e['filename']}` | {e['published'] or '—'} | {icon} | "
                     f"{e['authority']} | {e['publisher']} | {e['description']}{sup} |")
        L.append("")

    if unregistered:
        L += ["---", "", f"## ⚠️ 未登錄檔案（{len(unregistered)} 份）", "",
              "以下檔案存在於 `data/raw/SHARED/`，但沒有對應的登錄規則。",
              "**未登錄代表沒有人知道它是什麼、是否為現行版本**，",
              "請在 `scripts/build_sources_manifest.py` 的 `RULES` 補上規則。", ""]
        L += [f"- `{n}`" for n in unregistered]
        L.append("")

    L += ["---", "",
          "## 維護方式", "",
          "```powershell",
          "# 新增文件後重新產生清單",
          "python scripts\\build_sources_manifest.py",
          "",
          "# CI 檢查：有未登錄檔案就失敗",
          "python scripts\\build_sources_manifest.py --check",
          "```", "",
          "新增一類文件時，在 `RULES` 加一條規則即可（用正則配對檔名）。",
          "刻意用規則而非逐檔列舉：知識庫會持續長大（承保統計每月一批），",
          "逐檔維護的清單三個月後一定會跟現實脫節。", ""]

    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"✅ {OUT_MD.relative_to(config.PROJECT_ROOT)}　已登錄 {len(entries)} 份")
    print(f"✅ {OUT_JSON.relative_to(config.PROJECT_ROOT)}")
    for c in cats:
        print(f"   {c:<12}{len(by_cat[c]):>3} 份")
    if unregistered:
        print(f"\n⚠️  {len(unregistered)} 份未登錄：")
        for n in unregistered[:15]:
            print(f"     {n}")
        if args.check:
            sys.exit(1)


if __name__ == "__main__":
    main()
