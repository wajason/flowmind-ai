#!/usr/bin/env python3
"""
capture_dashboard.py — 把儀表板實際畫面截圖，供 README 與簡報使用
=============================================================================
【為什麼要有這支腳本】

「做了但別人看不到」在競賽場合等於沒做。
儀表板是這個產品最能讓非工程背景的人一眼看懂的一項，
但它跑在 localhost —— 評審翻 GitHub 時看不到任何畫面。

截圖要進版控的理由與 DEMO_RESULTS.md 相同：
**放示意圖任何人都做得出來；放真實畫面，代表這套系統當下真的跑得起來。**

【為什麼截圖也要能重跑】

一張手動截的圖，三個改動之後就與實際畫面不符，而且沒有人會發現。
做成腳本，改完 UI 重跑一次即可，截圖永遠對得上程式碼。

Usage:
    python -m flowmind.dashboard --port 8000     # 先另開一個終端機啟動
    python scripts/capture_dashboard.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images"

SHOTS = [
    ("dashboard-overview.png", None, 2200,
     "整體：四區塊一次看完"),
    ("dashboard-crosscheck.png", "#sec-cross", None,
     "交叉驗證分類卡片"),
    ("dashboard-cashflow.png", "#sec-cash", None,
     "現金流時間軸"),
]

# 問答區要先真的問一題才有內容可截 —— 截一張空的問答區沒有意義。
DEMO_Q = "跨文件勾稽有一個沒通過是什麼地方？"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tenant", default="CASE-9999")
    ap.add_argument("--skip-qa", action="store_true",
                    help="略過問答截圖（需要載入模型，約 2 分鐘）")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ 需要 playwright：uv pip install playwright && "
              "python -m playwright install chromium")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("═" * 70)
    print("  儀表板截圖")
    print("═" * 70)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900},
                                device_scale_factor=2)
        try:
            page.goto(args.url, wait_until="networkidle", timeout=60_000)
        except Exception as e:                             # noqa: BLE001
            print(f"❌ 連不上 {args.url}：{e}")
            print("   請先啟動：python -m flowmind.dashboard --port 8000")
            browser.close()
            return 1

        page.select_option("#tenant", args.tenant)
        page.wait_for_timeout(4000)          # 等三個區塊的 API 回來

        for name, sel, height, desc in SHOTS:
            path = OUT_DIR / name
            if sel:
                page.locator(sel).screenshot(path=str(path))
            else:
                page.screenshot(path=str(path), full_page=True)
            print(f"  ✅ {name:<28}{desc}　"
                  f"({path.stat().st_size // 1024} KB)")

        if not args.skip_qa:
            print(f"\n  問答區：實際送出一題（首次會等模型載入）…")
            page.fill("#q", DEMO_Q)
            page.click("#ask")
            try:
                page.wait_for_selector(".turn", timeout=300_000)
                page.wait_for_timeout(1500)
                p = OUT_DIR / "dashboard-qa.png"
                page.locator("#sec-conf").screenshot(path=str(p))
                print(f"  ✅ dashboard-qa.png　問答與信心分數組成"
                      f"　({p.stat().st_size // 1024} KB)")
            except Exception as e:                         # noqa: BLE001
                print(f"  ⚠️ 問答截圖失敗（{type(e).__name__}），其餘截圖仍有效")

        browser.close()

    print(f"\n  輸出目錄：{OUT_DIR}")
    print("  這些圖進版控 —— 放示意圖任何人都做得出來，")
    print("  放真實畫面才代表這套系統當下真的跑得起來。")
    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
