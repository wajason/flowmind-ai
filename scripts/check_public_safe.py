#!/usr/bin/env python3
"""
check_public_safe.py — 提交前檢查：不該公開的內容有沒有混進版控
=============================================================================
【為什麼需要這支腳本】

這個 repository 是公開的，但團隊本機有一些不對外的文件。
把它們排除在 .gitignore 只擋得住「不小心 git add」，擋不住另外三種外洩：

  1. `git add -f` 強制加入
  2. **新檔案**用了不同的檔名，但內容一樣不該公開
  3. **commit 訊息**本身寫出了不該說的內容

第 3 種最容易被忽略，而且傷害最大：檔案內容就算清乾淨了，
一句「移除了 XXX 內部文件」的 commit 訊息，等於公告「這裡藏了東西、
它叫什麼名字、為什麼藏」。本專案實際踩過這個坑，所以把檢查自動化。

【為什麼詞彙清單放在版控外】

如果把「禁止出現的詞」直接寫在這支腳本裡，這支腳本本身就是外洩來源——
公開 repo 裡放一份「這些字不能出現」的清單，等於把答案寫在題目旁邊。
所以：**邏輯公開、清單私有**。清單放在 `.private-terms`（已被 .gitignore 排除）。

【找不到清單時故意讓它失敗，而不是靜默跳過】

一個「找不到設定就自動放行」的檢查，在最需要它的時候（新環境、剛 clone）
剛好不會作用，而且不會有人發現。所以清單缺失時直接視為錯誤。

Usage:
    python scripts/check_public_safe.py --staged        # 檢查暫存區（pre-commit 用）
    python scripts/check_public_safe.py --msg <file>    # 檢查 commit 訊息（commit-msg 用）
    python scripts/check_public_safe.py --all           # 檢查整個工作目錄與全部歷史訊息
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMS_FILE = ROOT / ".private-terms"
EXAMPLE_FILE = ROOT / ".private-terms.example"

TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".yml", ".yaml", ".html",
                 ".css", ".js", ".sql", ".sh", ".jsonl", ".csv", ".cfg",
                 ".toml", ".ini", ""}


def load_terms() -> list[str]:
    """讀私有詞彙清單。缺檔就是錯誤，不靜默放行。"""
    if not TERMS_FILE.exists():
        print(f"❌ 找不到 {TERMS_FILE.name}——這份清單不進版控，需另行取得。")
        print(f"   請向團隊索取，或參考 {EXAMPLE_FILE.name} 自行建立。")
        print("   （在清單缺失的情況下放行檢查，等於在最需要它的時候關掉它）")
        sys.exit(2)
    terms = []
    for line in TERMS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    if not terms:
        print(f"❌ {TERMS_FILE.name} 是空的，等於沒有保護。")
        sys.exit(2)
    return terms


def ignored_paths() -> list[str]:
    """從 .gitignore 取出明確列出的私有路徑（非萬用字元那些）。"""
    gi = ROOT / ".gitignore"
    if not gi.exists():
        return []
    out, in_block = [], False
    for line in gi.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("# ── 不納入版控的本機文件"):
            in_block = True
            continue
        if in_block:
            if s.startswith("# ──"):
                break
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8").stdout


def check_staged(terms: list[str]) -> int:
    problems: list[str] = []
    staged = [f for f in _git("diff", "--cached", "--name-only").splitlines() if f]

    # (1) 私有路徑被強制加入
    private = ignored_paths()
    for f in staged:
        for pat in private:
            p = pat.rstrip("/")
            if f == p or f.startswith(p + "/") or (
                    "*" in pat and Path(f).match(pat)):
                problems.append(f"暫存區含私有路徑：{f}（比對到 .gitignore 的 {pat}）")

    # (2) 新檔案內容含私有詞彙（換個檔名一樣會外洩）
    for f in staged:
        if Path(f).suffix.lower() not in TEXT_SUFFIXES:
            continue
        content = _git("show", f":{f}")
        for t in terms:
            if t in content:
                problems.append(f"暫存內容含私有詞彙：{f} → 「{t}」")
                break

    return report(problems, "暫存區")


def check_message(path: Path, terms: list[str]) -> int:
    msg = path.read_text(encoding="utf-8")
    problems = [f"commit 訊息含私有詞彙：「{t}」" for t in terms if t in msg]
    hint = ([] if not problems else
            ["描述『移除了什麼不對外的文件』的訊息，本身就是外洩——",
             "訊息會永久留在歷史裡，而且比檔案更難清除。",
             "改用中性描述，例如「chore: 調整版控範圍」。"])
    return report(problems, "commit 訊息", hint)


def check_all(terms: list[str]) -> int:
    problems: list[str] = []
    for f in _git("ls-files").splitlines():
        if not f or Path(f).suffix.lower() not in TEXT_SUFFIXES:
            continue
        fp = ROOT / f
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for t in terms:
            if t in content:
                problems.append(f"已追蹤檔案含私有詞彙：{f} → 「{t}」")
                break

    history = _git("log", "--all", "--format=%H%n%B")
    for t in terms:
        if t in history:
            problems.append(f"歷史 commit 訊息含私有詞彙：「{t}」"
                            "（需 git filter-repo --replace-message 才能清除）")
    return report(problems, "全庫與歷史")


def report(problems: list[str], scope: str, hint: list[str] | None = None) -> int:
    print("═" * 70)
    print(f"  公開安全檢查：{scope}")
    print("═" * 70)
    if not problems:
        print("  ✅ 未發現不該公開的內容")
        print("═" * 70)
        return 0
    for p in problems:
        print(f"  ❌ {p}")
    if hint:
        print("─" * 70)
        for h in hint:
            print(f"  {h}")
    print("═" * 70)
    print(f"  發現 {len(problems)} 項問題，已阻止本次操作。")
    print("  確認為誤判時可用 --no-verify 略過，但請先真的確認過。")
    print("═" * 70)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--msg", type=Path)
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    terms = load_terms()
    if args.staged:
        return check_staged(terms)
    if args.msg:
        return check_message(args.msg, terms)
    return check_all(terms)


if __name__ == "__main__":
    sys.exit(main())
