#!/usr/bin/env python3
"""
fetch_public_corpus.py — 公開知識庫語料抓取
=============================================================================
把 RAG 共用知識層（tenant_id='SHARED'）需要的公開文件抓下來。

為什麼要寫成腳本而不是手動下載：
  評審與未來的合作企業會問「你的知識庫是哪來的、能不能重現」。
  一個「請手動去官網下載」的專案沒辦法回答這個問題；
  一支能重跑、會列出每個來源成功與否的腳本可以。
  抓取失敗的項目也會如實列出，不會假裝知識庫是完整的。

全部為政府機關與行庫公開發布的資料，僅作學術競賽用途，
不重新散布、不去除原始出處，下載後的檔案保留原始檔名與來源網址記錄。

用法：
  python scripts/fetch_public_corpus.py                 # 抓全部
  python scripts/fetch_public_corpus.py --only 法規      # 只抓某一類
  python scripts/fetch_public_corpus.py --list          # 只列出清單不下載
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind import config                                    # noqa: E402

OUT_DIR = config.RAW_DIR / "SHARED"
MANIFEST = OUT_DIR / "_source_manifest.json"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# ── 語料清單 ──────────────────────────────────────────────────────────────
# kind: pdf → 直接存檔；html → 抽正文存成 .md
SOURCES: list[dict] = [
    # ═══ 法規（正式法源，比白皮書更適合當引用依據）═══
    {"cat": "法規", "kind": "html",
     "name": "中小企業發展條例",
     "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0140001",
     "why": "SME 融資、信保基金撥款的法源依據"},
    {"cat": "法規", "kind": "html",
     "name": "中小企業認定標準",
     "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0140002",
     "why": "界定客戶是否符合中小企業資格，決定能否適用信保方案"},
    {"cat": "法規", "kind": "html",
     "name": "商業會計法",
     "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=J0080009",
     "why": "會計憑證保存年限、應收帳款認列，對應本系統的資料保存政策"},
    {"cat": "法規", "kind": "html",
     "name": "加值型及非加值型營業稅法",
     "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=G0340080",
     "why": "5% 營業稅率與發票開立規定，是交叉驗證 ARITH-02 的規則來源"},
    {"cat": "法規", "kind": "html",
     "name": "民法債編-債權讓與",
     "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=B0000001",
     "why": "應收帳款承購的法律本質是債權讓與，第294-299條為核心條文"},
    {"cat": "法規", "kind": "html",
     "name": "個人資料保護法",
     "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=I0050021",
     "why": "多租戶資料隔離與保存期限設計的法遵依據"},

    # ═══ 信保基金：保證要點與作業手冊 ═══
    # 註：smeg.org.tw 主站有 WAF，程式化存取一律 403。
    #     改用全國法規網站 rootlaw 的同一份要點全文，內容一致且可重現。
    {"cat": "融資商品說明", "kind": "html",
     "name": "信保基金-供應商融資信用保證要點",
     "url": "https://www.rootlaw.com.tw/LawArticle.aspx?LawID=A040390041056700-1090102",
     "why": "★本專案的直接法源：明定供應商得憑「訂單、發票（含電子發票）、"
            "支票、預約付款通知及其他得以佐證交易真實性之文件」申請撥貸，"
            "保證成數最高九成。FlowMind 做的就是自動化產出這些佐證文件的驗證結果"},
    {"cat": "融資商品說明", "kind": "pdf",
     "name": "信保基金-企業相對保證專案信用保證要點",
     "url": "https://www.nasme.org.tw/uploads/ck_file/manage/files/"
            "xc04125988111628831600-1.pdf",
     "why": "另一種保證方案的條件，供 RAG 做方案比較（由全國中小企業總會轉載）"},
    {"cat": "融資商品說明", "kind": "pdf",
     "name": "信保基金-認識信保基金與保證對象資格及業務規章",
     "url": "https://www.smeg.org.tw/archive/file/1-1%E8%AA%8D%E8%AD%98%E4%B8%AD%E5%B0%8F"
            "%E4%BF%A1%E4%BF%9D%E5%9F%BA%E9%87%91%E3%80%81%E4%BF%9D%E8%AD%89%E5%B0%8D%E8%B1%A1"
            "%E8%B3%87%E6%A0%BC%E5%8F%8A%E6%A5%AD%E5%8B%99%E8%A6%8F%E7%AB%A0(%E4%BF%A1%E5%90%88%E7%A4%BE).pdf",
     "why": "保證對象資格判定的第一手依據"},
    {"cat": "融資商品說明", "kind": "html",
     "name": "信保基金-直接信用保證要點",
     "url": "https://www.rootlaw.com.tw/LawContent.aspx?LawID=A040390041055500-1090102",
     "why": "直接保證（企業直接向信保基金申請）的條件，與間接保證的差異"},

    # ═══ 銀行供應鏈金融商品 ═══
    {"cat": "融資商品說明", "kind": "html",
     "name": "玉山銀行-應收帳款承購",
     "url": "https://www.esunbank.com/zh-tw/business/corporate/trade/factoring",
     "why": "商業銀行實際商品條件（有無追索權、通知/不通知）"},
    {"cat": "融資商品說明", "kind": "html",
     "name": "中國信託-應收帳款暨融資業務",
     "url": "https://www.ctbcbank.com/content/twcbo/zh_tw/finance/accountsreceivable/factoring.html",
     "why": "另一家行庫的商品說明，供跨行比較"},
    {"cat": "融資商品說明", "kind": "html",
     "name": "永豐銀行-應收帳款承購說明",
     "url": "https://bank.sinopac.com/sinopacBT/personal/article/intelligence/Factoring.html",
     "why": "承購流程與所需文件清單，直接對應本系統的『送件前檢核』"},

    # ═══ 市場規模與政策佐證 ═══
    {"cat": "白皮書統計", "kind": "html",
     "name": "經濟部中小及新創企業署-融資輔導",
     "url": "https://www.sme.gov.tw/",
     "why": "政策工具與輔導資源，提案書市場章節的引用來源"},
]


def slug(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", name).strip()


def html_to_markdown(html: str) -> str:
    """
    抽出網頁正文。刻意不用重量級的 readability 套件：
    法規網站與銀行商品頁的結構相對單純，去掉 script/style/nav 就夠了，
    多一個相依就多一個在別台機器上跑不起來的理由。
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    # 全國法規資料庫的條文在 .law-reg-content；抓不到就退回整頁
    main = (soup.select_one(".law-reg-content") or soup.select_one("main")
            or soup.select_one("#Content") or soup.body or soup)

    lines: list[str] = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "div"]):
        if el.find(["p", "li", "div", "td"]):
            continue                       # 只取葉節點，避免同一段文字重複三次
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 2:
            continue
        prefix = "#" * int(el.name[1]) + " " if el.name.startswith("h") else ""
        lines.append(prefix + text)

    out, seen = [], set()
    for ln in lines:
        if ln not in seen:                 # 導覽列文字常在頁面重複出現
            seen.add(ln)
            out.append(ln)
    return "\n\n".join(out)


def fetch(client: httpx.Client, src: dict) -> dict:
    result = {**src, "status": "failed", "file": None, "bytes": 0, "error": None}
    try:
        r = client.get(src["url"], follow_redirects=True, timeout=90)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()

        if src["kind"] == "pdf" or "pdf" in ctype:
            if not r.content.startswith(b"%PDF"):
                # 常見情況：連結失效時伺服器回一個 HTML 錯誤頁，副檔名還是 .pdf。
                # 不擋下來的話，後面的解析階段會拿到一堆亂碼卻不知道原因。
                raise ValueError(f"回應不是 PDF（content-type={ctype}，"
                                 f"開頭={r.content[:16]!r}），連結可能已失效")
            path = OUT_DIR / f"{slug(src['name'])}.pdf"
            path.write_bytes(r.content)
        else:
            md = html_to_markdown(r.text)
            if len(md) < 400:
                raise ValueError(f"抽出的正文只有 {len(md)} 字元，"
                                 f"可能是 JavaScript 動態載入的頁面，需要人工下載")
            path = OUT_DIR / f"{slug(src['name'])}.md"
            path.write_text(
                f"# {src['name']}\n\n"
                f"> 來源：{src['url']}\n"
                f"> 用途：{src['why']}\n"
                f"> 抓取時間：{time.strftime('%Y-%m-%d %H:%M')}\n\n---\n\n{md}",
                encoding="utf-8")

        result.update(status="ok", file=path.name, bytes=path.stat().st_size)
    except Exception as e:                              # noqa: BLE001
        result["error"] = str(e)[:200]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只抓某一分類（法規／融資商品說明／白皮書統計）")
    ap.add_argument("--list", action="store_true", help="只列出清單，不下載")
    args = ap.parse_args()

    todo = [s for s in SOURCES if not args.only or s["cat"] == args.only]

    if args.list:
        for s in todo:
            print(f"[{s['cat']}] {s['name']}\n    {s['url']}\n    用途：{s['why']}\n")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(headers={"User-Agent": UA}, verify=False) as client:
        for i, s in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {s['name']} …", end=" ", flush=True)
            r = fetch(client, s)
            results.append(r)
            print(f"✅ {r['file']} ({r['bytes']:,} bytes)" if r["status"] == "ok"
                  else f"❌ {r['error']}")
            time.sleep(1.0)      # 對政府網站禮貌一點，別把人家當壓測目標

    MANIFEST.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{'═'*70}")
    print(f"  成功 {len(ok)}／{len(results)}，共 {sum(r['bytes'] for r in ok):,} bytes")
    print(f"  來源清單已寫入 {MANIFEST}")
    if len(ok) < len(results):
        print("\n  以下來源需要人工處理（多半是動態網頁或連結已更新）：")
        for r in results:
            if r["status"] != "ok":
                print(f"   · {r['name']}\n     {r['url']}\n     原因：{r['error']}")
    print(f"{'═'*70}")
    print("  下一步： python data_update_finance.py --tenant SHARED --rebuild")


if __name__ == "__main__":
    main()
