#!/usr/bin/env python3
"""
report.py — 產出可以真的遞給銀行的 PDF 授信證據報告
=============================================================================
【為什麼這一支很重要】

產品定位是「把混亂的資料變成銀行看得懂的東西」。
但在這支模組出現之前，銀行看到的仍然是**終端機文字**。

一份可以列印、可以歸檔、可以夾進授信案卷的 PDF，
是把「我們算得出這些東西」變成「你可以拿去用」的最後一哩。
這件事本身就是產品完整性的一部分，不是包裝。

【三條刻意的設計】

  ① **不重算任何東西。** 全部取自 crosscheck / watchtower / metrics。
     報告與畫面、與終端機必須是同一組數字，否則稽核時無法對帳。

  ② **每一項都寫得出「怎麼算的」。** 授信人員被要求覆核時，
     必須能自己重算。一份只有結論沒有方法的報告，在內控上沒有價值。

  ③ **產品邊界寫在報告第一頁。** FlowMind 不做授信決策。
     報告是盡職調查的證據整理，最終要由有權責者簽署 ——
     這句話印在紙上，比寫在 README 裡有意義得多。

【技術選擇：ReportLab 而非 HTML→PDF】

HTML 轉 PDF（wkhtmltopdf / Playwright）需要額外的執行檔或瀏覽器，
在銀行的封閉環境裡是額外的部署阻力。
ReportLab 是純 Python，`pip install` 就結束。

中文字型是這條路的唯一風險：ReportLab 內建字型不含 CJK。
本模組會依序尋找系統中的常見中文字型，找不到時**明確報錯**
而不是產出一份滿是黑方塊的 PDF —— 那種 PDF 看起來像成功了。
"""

from __future__ import annotations

import platform
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from . import crosscheck, metrics, watchtower

# 依序尋找的中文字型。優先用標楷體／宋體這類公文常見字型，
# 讓輸出看起來像一份正式文件而不是網頁截圖。
_FONT_CANDIDATES = [
    ("kaiu.ttf", "TWKai"),          # 標楷體（Windows，公文標準）
    ("msjh.ttc", "MSJhengHei"),     # 微軟正黑體
    ("mingliu.ttc", "MingLiU"),     # 細明體
    ("NotoSansCJKtc-Regular.otf", "NotoTC"),
    ("NotoSansTC-Regular.otf", "NotoTC"),
    ("NotoSansCJK-Regular.ttc", "NotoCJK"),
    ("wqy-zenhei.ttc", "WQY"),
]
_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
    Path.home() / ".fonts", Path("/Library/Fonts"),
]

_STATUS = {
    "critical": (colors.HexColor("#d03b3b"), "重大"),
    "warning": (colors.HexColor("#b8860b"), "注意"),
    "info": (colors.HexColor("#2a78d6"), "提示"),
}
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#c3c2b7")
GOOD = colors.HexColor("#0a7a0a")


def _register_font() -> str:
    """
    找一個能顯示中文的字型並註冊。

    **找不到就拋錯，不退回內建字型。**
    ReportLab 的內建字型畫不出中文，會輸出一份滿是黑方塊的 PDF ——
    那種檔案「看起來像成功了」，是最糟的失敗方式：
    它會一路通過測試、通過流程，直到有人真的打開它。
    """
    for fname, name in _FONT_CANDIDATES:
        for d in _FONT_DIRS:
            if not d.exists():
                continue
            for p in ([d / fname] + list(d.rglob(fname))):
                if p.exists():
                    try:
                        idx = 0 if p.suffix.lower() != ".ttc" else 0
                        pdfmetrics.registerFont(
                            TTFont(name, str(p), subfontIndex=idx)
                            if p.suffix.lower() == ".ttc"
                            else TTFont(name, str(p)))
                        return name
                    except Exception:                      # noqa: BLE001
                        continue
    raise RuntimeError(
        "找不到可用的中文字型，無法產生 PDF。\n"
        f"已搜尋：{[str(d) for d in _FONT_DIRS if d.exists()]}\n"
        "Linux 可安裝：sudo apt install fonts-noto-cjk\n"
        "（刻意不退回英文字型 —— 那會產出一份滿是黑方塊、"
        "看起來卻像成功的 PDF。）")


def _styles(font: str) -> dict:
    base = getSampleStyleSheet()

    def mk(n: str, **kw):
        kw.setdefault("textColor", INK)          # 呼叫端可自行覆寫顏色
        return ParagraphStyle(n, parent=base["Normal"], fontName=font, **kw)

    return {
        "title": mk("t", fontSize=18, leading=24, alignment=TA_CENTER,
                    spaceAfter=2),
        "subtitle": mk("st", fontSize=10.5, leading=15, alignment=TA_CENTER,
                       textColor=MUTED, spaceAfter=14),
        "h1": mk("h1", fontSize=13, leading=19, spaceBefore=12, spaceAfter=6),
        "h2": mk("h2", fontSize=11, leading=16, spaceBefore=8, spaceAfter=4),
        "body": mk("b", fontSize=9.5, leading=15),
        "small": mk("s", fontSize=8.2, leading=12.5, textColor=MUTED),
        "cell": mk("c", fontSize=8.6, leading=12.5),
        "cellm": mk("cm", fontSize=8.2, leading=12, textColor=MUTED),
    }


def _fmt(v: Any) -> str:
    return f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)


def build(tenant_id: str, out_path: str | Path,
          as_of: Optional[date] = None) -> Path:
    """
    產生一份授信證據報告 PDF。

    所有數字取自既有模組，本函式**不做任何金融計算** ——
    報告與畫面、與終端機必須是同一組數字，否則稽核時無法對帳。
    """
    as_of = as_of or date.today()
    font = _register_font()
    S = _styles(font)

    data = metrics.load_engagement_files(tenant_id)
    if not data.get("invoices"):
        raise ValueError(f"{tenant_id} 沒有可供出具報告的憑證資料")
    rep = crosscheck.run_all(data["invoices"], data.get("contracts"),
                             data.get("ledger"), as_of=as_of)
    alerts = watchtower.scan(tenant_id, today=as_of, persist=False)

    client = tenant_id
    try:
        from . import db                                   # noqa: PLC0415
        with db.tenant_session("SHARED", admin=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT client_name, engagement_type FROM engagements "
                            "WHERE tenant_id = %s", (tenant_id,))
                row = cur.fetchone()
                if row:
                    client = row[0] or tenant_id
                    etype = row[1] or "未指定"
    except Exception:                                      # noqa: BLE001
        etype = "未指定"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _chrome(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 12 * mm,
                          f"FlowMind AI 授信證據報告　{client}　{as_of}")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"第 {doc.page} 頁")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title=f"FlowMind 授信證據報告 {client}", author="FlowMind AI")

    F: list = []
    F.append(Paragraph("授信證據報告", S["title"]))
    F.append(Paragraph("Verifiable Credit Evidence Report", S["subtitle"]))

    # ── 封面資訊 ─────────────────────────────────────────────────────
    head = [
        ["委任案編號", tenant_id, "報告基準日", str(as_of)],
        ["受查企業", client, "委任類型", etype],
        ["檢視文件", str(rep["documents_examined"]), "產製時間",
         datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    t = Table([[Paragraph(c, S["cell"]) for c in r] for r in head],
              colWidths=[24 * mm, 60 * mm, 24 * mm, 62 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f1ec")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f2f1ec")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    F.append(t)
    F.append(Spacer(1, 8))

    # ── 產品邊界：印在第一頁，不是藏在附註 ───────────────────────────
    F.append(Paragraph(
        "<b>本報告的性質與界線</b>", S["h2"]))
    F.append(Paragraph(
        "本報告為<b>盡職調查的證據整理</b>，"
        "所有結果均由程式以公開規則決定性計算，未經語言模型判斷，"
        "可由第三方以相同規則重算驗證。<br/>"
        "<b>本報告不構成授信決策、不構成投資或財務建議。</b>"
        "任何據以提出的融資建議與額度判斷，"
        "應由具授信權責之人員覆核並簽署。",
        S["body"]))
    F.append(Spacer(1, 6))

    # ── 一、結論摘要 ─────────────────────────────────────────────────
    ready = rep["submission_ready"]
    F.append(Paragraph("一、結論摘要", S["h1"]))
    summary = [
        ["文件完整性分數", f"{rep['integrity_score']:.1f}%"],
        ["重大缺失項數", str(rep["critical_failures"])],
        ["送件建議", "可送件" if ready else "建議先補正重大缺失"],
        ["主動監控警示",
         f"重大 {sum(1 for a in alerts if a.severity == 'critical')}　"
         f"注意 {sum(1 for a in alerts if a.severity == 'warning')}　"
         f"提示 {sum(1 for a in alerts if a.severity == 'info')}"],
    ]
    t = Table([[Paragraph(a, S["cell"]), Paragraph(f"<b>{b}</b>", S["cell"])]
               for a, b in summary], colWidths=[45 * mm, 125 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f1ec")),
        ("TEXTCOLOR", (1, 2), (1, 2), GOOD if ready else _STATUS["critical"][0]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    F.append(t)

    # ── 二、決定性檢查逐項結果 ───────────────────────────────────────
    F.append(Paragraph("二、決定性檢查逐項結果", S["h1"]))
    F.append(Paragraph(
        f"共 {len(rep['findings'])} 項檢查，全部為程式計算，零語言模型參與。"
        "「結果」欄位的判定規則公開於原始碼，可獨立重算。", S["small"]))
    F.append(Spacer(1, 4))

    rows = [[Paragraph(f"<b>{h}</b>", S["cell"]) for h in
             ("編號", "檢查項目", "結果", "說明")]]
    styles_extra = []
    for i, f in enumerate(rep["findings"], start=1):
        ok = f["passed"]
        mark = "通過" if ok else ("重大" if f["severity"] == "critical" else "注意")
        rows.append([
            Paragraph(f["check_id"], S["cellm"]),
            Paragraph(f["title"], S["cell"]),
            Paragraph(f"<b>{mark}</b>", S["cell"]),
            Paragraph(f["detail"], S["cellm"]),
        ])
        if not ok:
            styles_extra.append(
                ("TEXTCOLOR", (2, i), (2, i),
                 _STATUS["critical"][0] if f["severity"] == "critical"
                 else _STATUS["warning"][0]))
        else:
            styles_extra.append(("TEXTCOLOR", (2, i), (2, i), GOOD))

    t = Table(rows, colWidths=[20 * mm, 36 * mm, 13 * mm, 101 * mm],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f1ec")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ] + styles_extra))
    F.append(t)

    # ── 三、主動監控警示 ─────────────────────────────────────────────
    F.append(PageBreak())
    F.append(Paragraph("三、主動監控警示", S["h1"]))
    if not alerts:
        F.append(Paragraph("本次掃描未發現需要注意的事項。", S["body"]))
    else:
        F.append(Paragraph(
            "以下警示由決定性 SQL 產生，"
            "每一條均附上觸發它的實際資料列，可逐列複查。", S["small"]))
        F.append(Spacer(1, 4))
        for a in alerts:
            color, label = _STATUS.get(a.severity, _STATUS["info"])
            # ReportLab 的 <font color> 需要 # 前綴；
            # hexval() 回傳 '0xRRGGBB'，直接切掉 0x 會變成無效色值。
            hexc = "#" + color.hexval()[2:]
            block = [Paragraph(
                f'<font color="{hexc}"><b>[{label}] {a.title}</b></font>'
                f'　<font size="7.5" color="#888888">{a.rule_id}</font>', S["cell"]),
                Paragraph(a.detail, S["cellm"])]
            if a.evidence:
                head = list(a.evidence[0].keys())[:5]
                ev = [[Paragraph(f"<b>{h}</b>", S["cellm"]) for h in head]]
                for e in a.evidence[:5]:
                    ev.append([Paragraph(_fmt(e.get(h, "")), S["cellm"])
                               for h in head])
                et = Table(ev, colWidths=[170 / len(head) * mm] * len(head))
                et.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.3, RULE),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f7f6f2")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                block.append(Spacer(1, 3))
                block.append(et)
                if len(a.evidence) > 5:
                    block.append(Paragraph(
                        f"（另有 {len(a.evidence) - 5} 筆，完整清單見系統）",
                        S["small"]))
            block.append(Spacer(1, 9))
            F.append(KeepTogether(block))

    # ── 四、方法與可重現性 ───────────────────────────────────────────
    F.append(Paragraph("四、方法與可重現性", S["h1"]))
    F.append(Paragraph(
        "本報告的每一個數字都可由第三方以相同規則重算。重現指令：", S["body"]))
    for cmd, desc in [
        (f"python -m flowmind.cli crosscheck --tenant {tenant_id}",
         "第二節的逐項檢查結果"),
        (f"python -c \"from flowmind import watchtower; "
         f"print(watchtower.render(watchtower.scan('{tenant_id}')))\"",
         "第三節的監控警示"),
        (f"python -m flowmind.report --tenant {tenant_id}",
         "重新產生本報告"),
    ]:
        F.append(Paragraph(f"<font face='{font}' size='8'>{cmd}</font>",
                           S["cellm"]))
        F.append(Paragraph(f"　→ {desc}", S["small"]))
    F.append(Spacer(1, 6))
    F.append(Paragraph(
        "<b>檢查規則不會因為報告而改變。</b> 同一批資料在任何時間、"
        "由任何人執行，都會得到相同結果 —— 這是本系統與"
        "「請語言模型看一遍」最根本的差別。", S["body"]))

    doc.build(F, onFirstPage=_chrome, onLaterPages=_chrome)
    return out_path


def main() -> int:
    import argparse                                        # noqa: PLC0415
    ap = argparse.ArgumentParser(description="產生銀行可用的授信證據報告 PDF")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args()
    out = a.out or f"out/授信證據報告_{a.tenant}_{date.today()}.pdf"
    as_of = date.fromisoformat(a.as_of) if a.as_of else None
    p = build(a.tenant, out, as_of)
    print(f"✅ 已產生：{p}　({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
