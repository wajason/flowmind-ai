#!/usr/bin/env python3
"""
make_moat_figures.py — 產生「看了就懂護城河」的兩張圖
=============================================================================
【為什麼需要這兩張圖】

我們的兩個核心主張都是**抽象的**：
    「引用是可程式驗證的斷言，不是模型生成的標籤」
    「HPES 讓亂猜在數學上不划算」

用講的，評審點頭但記不住；畫出來，五秒就看懂，而且會記得。

這兩張圖刻意用**純 SVG 手繪**而不是繪圖套件：
它們不是資料視覺化（沒有需要探索的資料），是**概念圖**——
概念圖的每一個位置都是刻意安排的，用套件反而綁手綁腳。

【圖一：HPES 損益平衡】
把「猜的期望分數」與「留白的期望分數」畫在同一條把握度軸上，
交點就是 λ/(1+λ) = 66.7%。這條線是**數學事實**，不是我們訂的門檻——
這正是「不可 gameable」的意思。

【圖二：別人的引用 vs 我們的引用】
左右對照：市面上的做法是 LLM 自己寫 `[來源: x.pdf]`（那也是生成的 token），
我們的做法是回原文逐字比對、對不上就刪除。
中間標出「這條路徑上沒有任何一步經過語言模型」。

Usage:
    python scripts/make_moat_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"

# 取自已驗證的調色盤
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
S1, S2 = "#2a78d6", "#eb6834"
GOOD, CRIT = "#0ca30c", "#d03b3b"
TEAL = "#0f766e"
FONT = ("system-ui, -apple-system, 'Segoe UI', "
        "'Noto Sans TC', 'Microsoft JhengHei', sans-serif")

LAMBDA = 2.0        # 與 flowmind/verifin.py 的 HPES 懲罰係數一致


def hpes_figure() -> str:
    """圖一：猜 vs 留白的期望分數，交點即損益平衡點。"""
    W, H = 1000, 520
    P = {"l": 96, "r": 190, "t": 76, "b": 78}
    x0, x1 = P["l"], W - P["r"]
    y0, y1 = H - P["b"], P["t"]

    # y 軸：期望分數 −λ .. +1
    lo, hi = -LAMBDA, 1.0
    X = lambda p: x0 + p * (x1 - x0)                      # p: 0~1 把握度
    Y = lambda v: y1 + (1 - (v - lo) / (hi - lo)) * (y0 - y1)

    be = LAMBDA / (1 + LAMBDA)                            # 0.667
    guess = [(p, p * 1 + (1 - p) * (-LAMBDA)) for p in
             [i / 100 for i in range(101)]]
    gpath = " ".join(f"{'M' if i == 0 else 'L'}{X(p):.1f},{Y(v):.1f}"
                     for i, (p, v) in enumerate(guess))

    ticks = [(-2, "−2"), (-1, "−1"), (0, "0"), (1, "+1")]
    # 刻意不放 75%：它離 66.7% 只有 8 個百分點，兩個標籤會疊在一起。
    # 在概念圖上，**少一個刻度**遠比兩個字疊在一起好。
    xticks = [(0, "0%"), (.25, "25%"), (.5, "50%"), (1, "100%")]

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
     viewBox="0 0 {W} {H}" font-family="{FONT}">
  <rect width="{W}" height="{H}" fill="{SURF}"/>
  <text x="{P['l']}" y="40" font-size="26" font-weight="700" fill="{TEAL}">
    為什麼「亂猜」在數學上不划算</text>
  <text x="{P['l']}" y="63" font-size="14.5" fill="{INK2}">
    HPES：答對 +1　留白 0　答錯 −{LAMBDA:.0f}　—— 損益平衡點是算出來的，不是我們訂的</text>

  {"".join(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
           f'stroke="{GRID}" stroke-width="1"/>'
           f'<text x="{x0-12}" y="{Y(v)+5:.1f}" text-anchor="end" font-size="13" '
           f'fill="{MUTED}">{lab}</text>' for v, lab in ticks)}

  <!-- 留白：期望分數恆為 0 -->
  <line x1="{x0}" x2="{x1}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"
        stroke="{S1}" stroke-width="3"/>
  <!-- 猜：p·1 + (1−p)·(−λ) -->
  <path d="{gpath}" fill="none" stroke="{S2}" stroke-width="3"/>

  <!-- 交點 -->
  <line x1="{X(be):.1f}" x2="{X(be):.1f}" y1="{y1}" y2="{y0}"
        stroke="{CRIT}" stroke-width="1.5" stroke-dasharray="6 4"/>
  <circle cx="{X(be):.1f}" cy="{Y(0):.1f}" r="7" fill="{CRIT}"
          stroke="{SURF}" stroke-width="2.5"/>
  <text x="{X(be):.1f}" y="{y0+26:.1f}" text-anchor="middle" font-size="15"
        font-weight="700" fill="{CRIT}">66.7%</text>
  <!-- 標籤放在繪圖區「內」的頂端，不要放在 y1 之上 ——
       那裡是副標題的位置，兩者會疊在一起（第一版就疊了）。 -->
  <rect x="{X(be)-84:.1f}" y="{y1+8}" width="168" height="26" rx="6"
        fill="{SURF}" opacity="0.92"/>
  <text x="{X(be):.1f}" y="{y1+26:.1f}" text-anchor="middle" font-size="13.5"
        font-weight="600" fill="{CRIT}">損益平衡 λ/(1+λ)</text>

  <!-- 區域標註 -->
  <text x="{X(be)-16:.1f}" y="{Y(-1.15):.1f}" text-anchor="end" font-size="15"
        fill="{S2}" font-weight="600">把握度不足時</text>
  <text x="{X(be)-16:.1f}" y="{Y(-1.45):.1f}" text-anchor="end" font-size="14"
        fill="{INK2}">猜 → 期望分數為負</text>
  <text x="{X(be)+16:.1f}" y="{Y(.62):.1f}" font-size="15" fill="{S2}"
        font-weight="600">把握度夠高時</text>
  <text x="{X(be)+16:.1f}" y="{Y(.40):.1f}" font-size="14" fill="{INK2}">
    猜才開始划算</text>

  {"".join(f'<text x="{X(p):.1f}" y="{y0+26:.1f}" text-anchor="middle" '
           f'font-size="13" fill="{MUTED}">{lab}</text>'
           for p, lab in xticks if lab)}
  <text x="{(x0+x1)/2:.1f}" y="{H-22}" text-anchor="middle" font-size="14"
        fill="{INK2}">模型對這個答案的真實把握度</text>

  <!-- 圖例 -->
  <rect x="{x1+22}" y="{y1+6}" width="14" height="14" rx="3" fill="{S1}"/>
  <text x="{x1+42}" y="{y1+18}" font-size="14" fill="{INK}">留白</text>
  <text x="{x1+42}" y="{y1+37}" font-size="12.5" fill="{MUTED}">恆為 0</text>
  <rect x="{x1+22}" y="{y1+56}" width="14" height="14" rx="3" fill="{S2}"/>
  <text x="{x1+42}" y="{y1+68}" font-size="14" fill="{INK}">猜</text>
  <text x="{x1+42}" y="{y1+87}" font-size="12.5" fill="{MUTED}">隨把握度變化</text>

  <text x="{x1+22}" y="{y0-46}" font-size="13" fill="{INK2}" font-weight="600">
    這條線是數學事實</text>
  <text x="{x1+22}" y="{y0-26}" font-size="12.5" fill="{MUTED}">不是可以調的門檻</text>
  <text x="{x1+22}" y="{y0-8}" font-size="12.5" fill="{MUTED}">→ 指標不可 gameable</text>
</svg>'''


def citation_figure() -> str:
    """圖二：市面上的「引用」vs 可程式驗證的引用。"""
    W, H = 1000, 520
    colw, gap = 430, 60
    lx, rx = 40, 40 + colw + gap

    def box(x, y, w, h, fill, stroke, rx=10):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
     viewBox="0 0 {W} {H}" font-family="{FONT}">
  <rect width="{W}" height="{H}" fill="{SURF}"/>
  <text x="{lx}" y="40" font-size="26" font-weight="700" fill="{TEAL}">
    「有引用」不等於「有根據」</text>

  <!-- 左：市面做法 -->
  {box(lx, 62, colw, 400, "#fdf3f3", "#d03b3b55")}
  <text x="{lx+20}" y="94" font-size="18" font-weight="700" fill="{CRIT}">
    市面上的 RAG 產品</text>
  <text x="{lx+20}" y="126" font-size="14.5" fill="{INK}">請 LLM 在句尾寫上來源檔名</text>

  {box(lx+20, 142, colw-40, 92, "#fff", "#d03b3b33", 8)}
  <text x="{lx+34}" y="170" font-size="14" fill="{INK}">保證成數最高九成。</text>
  <text x="{lx+34}" y="196" font-size="13.5" fill="{CRIT}" font-family="ui-monospace, monospace">
    [來源: 信保要點.pdf]</text>
  <text x="{lx+34}" y="220" font-size="12.5" fill="{MUTED}">↑ 這一串也是模型生成的 token</text>

  <text x="{lx+20}" y="266" font-size="15" font-weight="600" fill="{CRIT}">問題在哪</text>
  <!-- SVG 不吃 markdown，粗體要用 tspan。
       第一版直接寫 **完全沒讀過**，結果星號原樣印在圖上。 -->
  <text x="{lx+20}" y="292" font-size="14" fill="{INK2}">模型可以在<tspan
    font-weight="700" fill="{CRIT}">完全沒讀過</tspan>那份文件</text>
  <text x="{lx+20}" y="314" font-size="14" fill="{INK2}">的情況下，寫出格式完美的引用。</text>
  <text x="{lx+20}" y="348" font-size="14" fill="{INK2}">從畫面上，使用者分辨不出</text>
  <text x="{lx+20}" y="370" font-size="14.5" fill="{CRIT}" font-weight="600">
    「有引用」和「有根據」的差別。</text>

  {box(lx+20, 392, colw-40, 52, "#fff", "#d03b3b33", 8)}
  <text x="{lx+34}" y="414" font-size="13.5" fill="{INK2}">要驗證，只能靠人</text>
  <text x="{lx+34}" y="434" font-size="13.5" fill="{INK2}">一份一份翻回去對。</text>

  <!-- 右：我們的做法 -->
  {box(rx, 62, colw, 400, "#f1f8f4", "#0ca30c55")}
  <text x="{rx+20}" y="94" font-size="18" font-weight="700" fill="{GOOD}">
    FlowMind</text>
  <text x="{rx+20}" y="126" font-size="14.5" fill="{INK}">每個主張必須附一段逐字摘錄</text>

  {box(rx+20, 142, colw-40, 92, "#fff", "#0ca30c33", 8)}
  <text x="{rx+34}" y="170" font-size="14" fill="{INK}">保證成數最高九成。</text>
  <text x="{rx+34}" y="194" font-size="12.5" fill="{INK2}" font-family="ui-monospace, monospace">
    引用「信用保證成數最高九成。」</text>
  <text x="{rx+34}" y="216" font-size="12.5" fill="{GOOD}" font-weight="600">
    ✓ exact　回原文逐字比對通過</text>

  <text x="{rx+20}" y="266" font-size="15" font-weight="600" fill="{GOOD}">關鍵差別</text>
  <text x="{rx+20}" y="292" font-size="14" fill="{INK2}">程式回到實際檢索到的文本</text>
  <text x="{rx+20}" y="314" font-size="14" fill="{INK2}">做字串比對，</text>
  <text x="{rx+20}" y="336" font-size="14.5" fill="{CRIT}" font-weight="600">
    對不上的直接從答案中移除。</text>

  {box(rx+20, 356, colw-40, 88, "#fff", "#0ca30c33", 8)}
  <text x="{rx+34}" y="382" font-size="14" font-weight="700" fill="{TEAL}">
    這條路徑上</text>
  <text x="{rx+34}" y="406" font-size="14" font-weight="700" fill="{TEAL}">
    沒有任何一步經過語言模型</text>
  <text x="{rx+34}" y="430" font-size="13" fill="{INK2}">
    → 不可能被「講得更有說服力」騙過</text>

  <text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="13.5" fill="{MUTED}">
    模型唯一能提高分數的方法，就是真的去讀檢索到的文件</text>
</svg>'''


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figs = [("moat-hpes.svg", hpes_figure(), "HPES 損益平衡：亂猜為何不划算"),
            ("moat-citation.svg", citation_figure(), "有引用 ≠ 有根據")]

    print("═" * 70)
    print("  護城河概念圖")
    print("═" * 70)
    for name, svg, desc in figs:
        (OUT / name).write_text(svg, encoding="utf-8")
        print(f"  ✅ {name:<22}{desc}")

    # marp 需要點陣圖才能穩定嵌入，順便轉 PNG
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1000, "height": 520},
                            device_scale_factor=2)
            for name, _svg, _d in figs:
                png = OUT / name.replace(".svg", ".png")
                pg.goto((OUT / name).as_uri())
                pg.screenshot(path=str(png))
                print(f"  ✅ {png.name:<22}({png.stat().st_size // 1024} KB)")
            b.close()
    except ImportError:
        print("  ⚠️ 未安裝 playwright，略過 PNG 轉檔（SVG 已產生）")

    print("═" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
