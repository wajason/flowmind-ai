#!/usr/bin/env python3
"""
integrate_new_data.py — 把手動下載的檔案整併進 data/raw/SHARED
=============================================================================
一次性的整併工具。人工從有 WAF 或需要登入的網站下載的檔案，
統一丟進 new_data/，跑這支腳本把它們併進知識庫。

做四件事：
  1. 略過瀏覽器「另存新檔」產生的 *_files/ 資源目錄（css/js/圖片，不是知識內容）
  2. HTML → Markdown（保留來源與抓取時間的 front matter）
  3. 以內容雜湊偵測重複檔案（例如「直接信用保證要點.pdf」與
     「4_直接信用保證要點.pdf」是同一份，只是檔名多了項次前綴）
  4. 檔名正規化：去掉「4_」「(1)」這類前後綴，統一命名慣例

為什麼要正規化檔名：檔名是引用驗證的比對鍵，也是 LLM 會照抄的字串。
檔名越規律，模型抄錯的機率越低，使用者回頭查證也越容易。

Usage:
    python scripts/integrate_new_data.py --dry-run    # 先看會做什麼
    python scripts/integrate_new_data.py             # 真的搬
    python scripts/integrate_new_data.py --purge     # 搬完刪掉 new_data
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowmind import config                                    # noqa: E402

SRC = config.PROJECT_ROOT / "new_data"
DST = config.RAW_DIR / "SHARED"

KEEP_EXT = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".pptx", ".md", ".txt", ".html", ".htm"}

# 檔名正規化規則：把人工下載時帶上的雜訊拿掉
RENAME_RULES = [
    (re.compile(r"^\d+_"), ""),                       # 「4_直接信用保證要點」→「直接信用保證要點」
    (re.compile(r"\s*\(\d+\)$"), ""),                 # 「xxx (1)」→「xxx」
    (re.compile(r"\s*-\s*中國信託$"), "（中國信託）"),   # 統一機構標示位置
    (re.compile(r"^財團法人中小企業信用保證基金"), "信保基金-"),
]

# 已知需要加上機構前綴的檔案（讓檔名自帶出處，模型引用時不會搞混）
PREFIX_RULES = [
    (["直接信用保證要點", "供應商融資", "作業手冊", "誠信經營規範",
      "工作規則", "迴避", "承保統計", "保證規劃", "組織與職掌"], "信保基金-"),
]

# 明確排除的檔案。雜湊去重抓不到這種：內容 99% 相同、
# 只差一個「項次4」前綴，雜湊完全不同。
# 同一份要點入庫兩次會在檢索時互相稀釋（多樣性過濾把兩份當成不同來源），
# 所以寧可人工列一條規則，也不要留兩份。
EXPLICIT_SKIP = {
    "4_直接信用保證要點.pdf": "與「直接信用保證要點.pdf」為同一份要點，僅多了項次編號前綴",
}


def norm_name(stem: str) -> str:
    for pat, rep in RENAME_RULES:
        stem = pat.sub(rep, stem)
    stem = stem.strip()
    for keys, prefix in PREFIX_RULES:
        if not stem.startswith(prefix) and any(k in stem for k in keys):
            stem = prefix + stem
    return stem


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def html_to_md(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]):
        t.decompose()
    main = soup.select_one("main") or soup.select_one("#Content") or soup.body or soup

    lines, seen = [], set()
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "dd", "dt"]):
        if el.find(["p", "li", "td", "dd", "dt"]):
            continue                                   # 只取葉節點，避免重複
        txt = el.get_text(" ", strip=True)
        if len(txt) < 2 or txt in seen:
            continue
        seen.add(txt)
        prefix = "#" * int(el.name[1]) + " " if el.name.startswith("h") else ""
        lines.append(prefix + txt)
    return "\n\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--purge", action="store_true", help="整併完成後刪除 new_data")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"找不到 {SRC}，無事可做。")
        return

    DST.mkdir(parents=True, exist_ok=True)
    existing = {file_sha(p): p.name for p in DST.iterdir() if p.is_file()}

    moved, skipped_dup, skipped_asset, converted = [], [], 0, []

    for p in sorted(SRC.rglob("*")):
        if p.is_dir():
            continue
        # 瀏覽器另存的資源目錄
        if any(part.endswith("_files") for part in p.relative_to(SRC).parts):
            skipped_asset += 1
            continue
        # 「.下載」是 Chrome 未完成下載的暫存副檔名
        if p.suffix.lower() not in KEEP_EXT:
            skipped_asset += 1
            continue

        if p.name in EXPLICIT_SKIP:
            skipped_dup.append((p.name, EXPLICIT_SKIP[p.name]))
            continue

        sha = file_sha(p)
        if sha in existing:
            skipped_dup.append((p.name, f"與既有的 {existing[sha]} 內容完全相同"))
            continue

        stem = norm_name(p.stem)

        if p.suffix.lower() in (".html", ".htm"):
            md = html_to_md(p)
            if len(md) < 300:
                skipped_asset += 1
                continue
            target = DST / f"{stem}.md"
            body = (f"# {stem}\n\n"
                    f"> 來源：人工下載（該站台阻擋程式化存取）\n"
                    f"> 原始檔名：{p.name}\n\n---\n\n{md}")
            if not args.dry_run:
                target.write_text(body, encoding="utf-8")
            converted.append((p.name, target.name))
        else:
            target = DST / f"{stem}{p.suffix.lower()}"
            n = 1
            while target.exists() and not args.dry_run:
                target = DST / f"{stem}_{n}{p.suffix.lower()}"
                n += 1
            if not args.dry_run:
                shutil.copy2(p, target)
            moved.append((p.name, target.name))

        existing[sha] = target.name

    print("═" * 74)
    print(f"  new_data 整併{'（試跑，未實際搬動）' if args.dry_run else ''}")
    print("═" * 74)
    print(f"\n▶ 複製 {len(moved)} 個檔案")
    for a, b in moved:
        mark = "  " if a == b else "→ "
        print(f"    {mark}{a}" + (f"　改名為 {b}" if a != b else ""))
    print(f"\n▶ HTML 轉 Markdown {len(converted)} 份")
    for a, b in converted:
        print(f"    {a} → {b}")
    print(f"\n▶ 重複略過 {len(skipped_dup)} 個")
    for a, why in skipped_dup:
        print(f"    {a}\n      理由：{why}")
    print(f"\n▶ 略過網頁資源檔 {skipped_asset} 個（css/js/圖片，非知識內容）")

    if args.purge and not args.dry_run:
        shutil.rmtree(SRC)
        print(f"\n🗑️  已刪除 {SRC}")

    print("\n下一步：")
    print("  python scripts/build_sources_manifest.py    # 更新資料來源清單")
    print("  python data_update_finance.py --tenant SHARED   # 增量入庫")


if __name__ == "__main__":
    main()
