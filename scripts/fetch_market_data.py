#!/usr/bin/env python3
"""
fetch_market_data.py — 市場資料層（六層知識庫的最後一層）
=============================================================================
【這一層要回答的問題】

授信人員真正會問的是：「**在現在這個利率環境下，這個方案划算嗎？**」

要回答它，系統必須知道當前的資金成本基準。
但這一層有一個特殊風險：**市場資料會過期，而過期的市場資料比沒有更危險。**
用去年的重貼現率去算今年的利差，會得到一個看起來很專業的錯誤答案。

所以這一層的每一筆都強制帶三樣東西：
    · 資料日期（不是抓取日期 —— 兩者常常差很多）
    · 來源機關
    · 過期判定（超過 N 天就標記為 stale，並在回答時揭露）

【為什麼是「最小版」】

完整的市場資料層需要即時行情、新聞情緒分析、景氣領先指標。
那是另一個產品。這裡只做**回答得了那個問題所需的最小集合**：
央行政策利率 + 五大銀行新承作放款利率 + 幾則具時間標記的財經標題。

**刻意不做的**：新聞情緒分析。把「新聞標題」變成「市場情緒分數」
需要另一整套可解釋性，而我們沒有。標題就以標題的形式存在。

Usage:
    python scripts/fetch_market_data.py            # 抓取並寫入語料
    python scripts/fetch_market_data.py --dry-run  # 只顯示不寫檔
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "raw" / "SHARED" / "市場資料-利率與景氣.md"
MANIFEST = ROOT / "data" / "raw" / "SHARED" / "_market_manifest.json"

# 超過這個天數就視為過期。90 天的依據：央行理監事會議每季召開一次，
# 一季沒更新代表至少錯過一次可能的利率決策。
STALE_DAYS = 90

UA = {"User-Agent": "Mozilla/5.0 (compatible; FlowMind/1.0; research)"}


def _fetch_cbc_rates() -> dict:
    """
    中央銀行政策利率。

    央行的網頁結構會變，所以這裡**不做深度解析** ——
    只抓出「重貼現率 X.XXX%」這種明確樣式。
    抓不到就明說抓不到，不要猜一個數字填進去：
    一個猜出來的政策利率會讓所有以它為基礎的試算全部失效。
    """
    url = "https://www.cbc.gov.tw/tw/lp-370-1.html"
    out = {"source": "中央銀行　央行貼放利率", "url": url,
           "items": [], "error": None}
    try:
        r = httpx.get(url, headers=UA, timeout=60, follow_redirects=True)
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text)

        # 這是一張表：欄位名在前，資料列在後，例如
        #   調整日期 重貼現率 擔保放款融通利率 短期融通利率
        #   2024/3/22 2 2.375 4.25
        # 第一版誤以為數值緊接在欄位名後面，結果一筆都抓不到。
        # **抓不到時它拒絕寫入，那個行為是對的** —— 錯的是解析。
        head = text.find("調整日期")
        if head < 0:
            raise ValueError("找不到『調整日期』表頭，頁面結構可能已改變")
        row = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})\s+"
                        r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
                        text[head:head + 600])
        if not row:
            raise ValueError("表頭後找不到『日期 + 三個利率』的資料列")

        out["as_of"] = date(int(row.group(1)), int(row.group(2)),
                            int(row.group(3))).isoformat()
        for i, name in enumerate(("重貼現率", "擔保放款融通利率", "短期融通利率"),
                                 start=4):
            out["items"].append({"name": name, "value_pct": float(row.group(i))})
    except Exception as e:                                 # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:160]
    return out


def _fetch_overnight_rate() -> dict:
    """
    金融業隔夜拆款利率 —— 這一項比政策利率更貼近「當下的資金成本」。

    政策利率可能一年不動（現行是 2024/3/22 調整的），
    但隔夜拆款利率天天在變。回答「現在的利率環境」時，
    只講政策利率會給人一種「什麼都沒變」的錯覺。
    """
    url = "https://www.cbc.gov.tw/tw/lp-372-1.html"
    out = {"source": "中央銀行　金融統計", "url": url, "items": [], "error": None}
    try:
        r = httpx.get(url, headers=UA, timeout=60, follow_redirects=True)
        r.raise_for_status()
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
        m = re.search(r"金融業隔夜拆款利率\s*(\d{4}-\d{2}-\d{2})\s*"
                      r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            out["as_of"] = m.group(1)
            out["items"].append({"name": "金融業隔夜拆款利率",
                                 "value_pct": float(m.group(2))})
        else:
            out["error"] = "未能解析隔夜拆款利率"
    except Exception as e:                                 # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:160]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = date.today()
    rates = _fetch_cbc_rates()

    print("═" * 74)
    print("  市場資料層")
    print("═" * 74)
    print(f"  央行利率：{len(rates['items'])} 筆"
          + (f"　⚠️ {rates['error']}" if rates["error"] else ""))
    for it in rates["items"]:
        print(f"    {it['name']}　{it['value_pct']}%")

    if not rates["items"]:
        print("\n  ❌ 沒有抓到任何利率數字，**不寫入語料**。")
        print("     一個猜出來的政策利率，會讓所有以它為基礎的試算全部失效；")
        print("     缺這一層，比放一個錯的數字進去安全得多。")
        return 1

    as_of = rates.get("as_of", today.isoformat())
    stale = (today - date.fromisoformat(as_of)).days > STALE_DAYS

    body = [
        "# 市場資料：利率與資金成本基準", "",
        f"> 來源：{rates['source']}（{rates['url']}）",
        f"> 資料日期：**{as_of}**　｜　抓取日期：{today}",
        f"> 過期判定：{'⚠️ 已超過 ' + str(STALE_DAYS) + ' 天，使用前必須確認' if stale else '✅ 在有效期內'}",
        "", "---", "",
        "## 央行政策利率", "",
        "| 項目 | 年利率 |", "|---|---|",
    ]
    for it in rates["items"]:
        body.append(f"| {it['name']} | {it['value_pct']}% |")
    body += [
        "", "## 這些數字在授信判斷上的用途", "",
        "重貼現率是銀行向央行融通的成本，也是各項放款利率的定錨。",
        "供應鏈融資的訂價通常表達為「基準利率 + 加碼」，",
        "所以要回答「這個方案划不划算」，必須先知道當期的基準。", "",
        "> ⚠️ **市場資料會過期，而過期的市場資料比沒有更危險。**",
        "> 用去年的重貼現率去算今年的利差，會得到一個看起來很專業的錯誤答案。",
        f"> 本檔案的資料日期是 {as_of}；超過 {STALE_DAYS} 天未更新即應重新抓取。", "",
        "> **本層刻意不做新聞情緒分析。**",
        "> 把新聞標題變成「市場情緒分數」需要另一整套可解釋性 ——",
        "> 那個分數一旦進了授信報告，就必須能被質疑與重算，而我們沒有。",
    ]

    if args.dry_run:
        print("\n" + "\n".join(body))
        return 0

    OUT.write_text("\n".join(body) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps(
        {"fetched_at": datetime.now().isoformat(timespec="seconds"),
         "as_of": as_of, "stale": stale, "stale_days_threshold": STALE_DAYS,
         "rates": rates}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✅ 已寫入 {OUT.name}（資料日期 {as_of}）")
    print("     記得跑 data_update_finance.py --tenant SHARED 讓它進入檢索")
    return 0


if __name__ == "__main__":
    sys.exit(main())
