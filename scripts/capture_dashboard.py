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
     "整體：四區塊一次看完（README 用）"),
    ("dashboard-crosscheck.png", "#sec-cross", None,
     "交叉驗證分類卡片"),
    ("dashboard-cashflow.png", "#sec-cash", None,
     "現金流時間軸"),
]

# ── 投影片專用截圖 ────────────────────────────────────────────────────────
#
# 16:9 投影片的可用區域大約是 1280×580（扣掉標題與頁尾）。
# 一張 2880×3294 的整頁截圖放進去，必然壓縮到看不清楚、或者溢出畫面外。
# **實際發生過**：5 分鐘簡報的兩頁 Demo 圖都超出下緣被切掉。
#
# 所以投影片用的是「視窗尺寸的裁切」而不是整頁 —— 一張投影片
# 本來就只該傳達一件事，塞下整頁儀表板既看不清也沒有重點。
# (檔名, 視窗裁切, 區塊選擇器, 區塊最大高度, 說明)
# 區塊截圖也要限高 —— 一個完整區塊往往比 16:9 高，
# 直接整塊截下來放進投影片就是溢出。
SLIDE_SHOTS = [
    ("slide-dashboard-top.png", {"width": 1440, "height": 640}, None, None,
     "戰情室上半：紅黃綠燈 + 交叉驗證卡片"),
    ("slide-dashboard-cash.png", None, "#sec-cash", 600,
     "現金流兩張圖"),
    ("slide-simulate.png", None, "#sec-sim", 620,
     "情境模擬：兩條曲線比較"),
    ("slide-financing.png", None, "#sec-sim", None,
     "融資方案並排比較（單獨一頁）"),
]

# 投影片圖片的長寬比上限。
#
# 算法：16:9 投影片是 1280×720，扣掉 padding 與標題後，
# 圖片實際可用高度約 480px。若圖以 w:1000 放入，
# 高度 = 1000 × ratio，要 ≤ 480 就代表 ratio ≤ 0.48。
# 再留一點給圖下方的說明文字，抓 0.45。
#
# 第一版設 1.05，等於形同虛設 —— 一張 0.50 的圖照樣溢出。
# **一個永遠會通過的檢查，等於沒有檢查。**
MAX_SLIDE_RATIO = 0.45

# 問答區要先真的問一題才有內容可截 —— 截一張空的問答區沒有意義。
DEMO_Q = "跨文件勾稽有一個沒通過是什麼地方？"


def _png_size(path: Path) -> tuple[int, int]:
    """讀 PNG 的寬高（只讀檔頭 24 bytes，不需要 Pillow）。"""
    import struct                                          # noqa: PLC0415
    with path.open("rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    return struct.unpack(">II", head[16:24])


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

        # 模擬區要先真的跑一次才有內容 ——
        # 截一張空的模擬區等於截一張「這個功能沒做」的圖。
        # 第一版就是這樣，產出的圖只有 370px 高（比例 0.13），
        # 一看就知道裡面什麼都沒有。
        try:
            page.fill("#sim-amt", "40000000")
            page.fill("#sim-days", "20")
            page.click("#sim-run")
            page.wait_for_selector(".verdict", timeout=90_000)
            page.wait_for_timeout(1200)
        except Exception as e:                             # noqa: BLE001
            print(f"  ⚠️ 模擬區未能執行（{type(e).__name__}），該張截圖會是空的")

        # ── 投影片專用（裁切過，放得進 16:9）──────────────────────────
        print("\n  投影片專用截圖（16:9 放得下的尺寸）：")
        for name, clip_vp, sel, max_h, desc in SLIDE_SHOTS:
            path = OUT_DIR / name
            if sel and name == "slide-financing.png":
                # 融資卡片在區塊下半，單獨截一頁 —— 它本身就是一個賣點，
                # 擠在同一張圖裡兩邊都看不清。
                cards = page.locator("#sec-sim .cards")
                if cards.count():
                    cards.screenshot(path=str(path))
                else:
                    print(f"  ⚠️ {name} 沒有融資卡片可截（模擬未觸發缺口）")
                    continue
            elif sel:
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(300)
                box = page.locator(sel).bounding_box()
                if box and max_h and box["height"] > max_h:
                    # clip 的座標系要看 full_page：
                    # full_page=False 時是**視窗**座標，區塊在捲動範圍外就會
                    # 報「Clipped area is either empty or outside」。
                    # full_page=True 才是**頁面**座標。
                    page.screenshot(path=str(path), full_page=True, clip={
                        "x": box["x"], "y": box["y"],
                        "width": box["width"], "height": max_h})
                else:
                    page.locator(sel).screenshot(path=str(path))
            else:
                # 一定要先捲回頂端再裁切 ——
                # 前面跑模擬時頁面已經被捲動過，clip 的 y=0 是**視窗座標**，
                # 不捲回去就會截到當下看到的區塊。
                # 第一版就是這樣：想截「紅黃綠燈 + 交叉驗證」，
                # 結果截到現金流區，而且沒有任何錯誤訊息。
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(400)
                page.screenshot(path=str(path), full_page=False,
                                clip={"x": 0, "y": 0, **clip_vp})
            w, h = _png_size(path)
            ratio = h / w if w else 99
            flag = "✅" if ratio <= MAX_SLIDE_RATIO else "⚠️ 太高，放進投影片會溢出"
            print(f"  {flag} {name:<26}{w}×{h}　比例 {ratio:.2f}　{desc}")

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
