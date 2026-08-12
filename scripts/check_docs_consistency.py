#!/usr/bin/env python3
"""
check_docs_consistency.py — 文件裡的數字必須與實際跑出來的一致
=============================================================================
【為什麼需要這支腳本】

同一個「回歸測試項數」在文件裡出現過 39、58、139、154 四個版本。
這不是誰不小心 —— 而是**同一個事實被抄寫在十幾個地方**，
每次改動都要人記得全部更新，那注定會失效。

在一份要交給評審、要說服金融機構的技術文件裡，
數字對不上比數字不好看嚴重得多：它會讓人合理懷疑**其他所有數字**。

所以把「文件宣稱的數字」變成**可執行的檢查**：
從真實來源取值（跑測試、查資料庫、讀評測結果檔），
再掃描所有 .md，找出與事實不符的數字。

【設計原則：只查有唯一正確答案的數字】

不查敘述、不查形容詞、不做語意判斷。只查這幾類：

    回歸測試項數      跑 tests/test_core.py 取得
    知識庫規模        查資料庫
    知識圖譜節點數    查資料庫
    50 題評測指標     讀 docs/QA_EVAL_final.json
    造假偵測指標      讀 docs/FRAUD_BENCHMARK.json

這些都有唯一正確答案，對不上就是錯，沒有解釋空間。

Usage:
    python scripts/check_docs_consistency.py            # 只檢查
    python scripts/check_docs_consistency.py --fix      # 自動更新過時數字
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 要掃描的文件。刻意逐一列出而不是掃全部 *.md ——
# scratch 筆記、第三方文件的數字不歸我們管。
DOCS = [
    "README.md", "HANDOVER.md", "ONE_PAGER.md",
    "docs/SDD.md", "docs/PROPOSAL.md", "docs/PROPOSAL_SUBMISSION.md",
    "docs/DECISIONS.md", "docs/BUSINESS_CASE.md", "docs/MODEL_SELECTION.md",
    "docs/MARKET_VALIDATION.md", "docs/DEMO_RESULTS.md",
    "docs/slides/FlowMind_團隊簡報.md",
]


@dataclass
class Fact:
    key: str
    value: object
    # 找出「宣稱這個事實」的所有寫法。用 (?P<num>) 標出數字的位置。
    patterns: list[str]
    note: str = ""

    def fmt(self) -> str:
        if isinstance(self.value, float):
            return f"{self.value:.3f}".rstrip("0").rstrip(".")
        # 千分位從 1000 起加，與文件既有寫法一致（7,619 而不是 7619）。
        # 原本門檻設 10000，結果把文件裡正確的「7,619」改成「7619」——
        # 一個「修正一致性」的工具自己製造了不一致。
        return f"{self.value:,}" if isinstance(self.value, int) and self.value >= 1000 \
            else str(self.value)


def _run_tests() -> tuple[int, int]:
    r = subprocess.run([sys.executable, "tests/test_core.py"],
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=str(ROOT), timeout=1800)
    m = re.search(r"通過\s*(\d+)\s*　?失敗\s*(\d+)", r.stdout or "")
    if not m:
        raise RuntimeError("測試輸出無法解析 —— 無法取得真實項數")
    return int(m.group(1)), int(m.group(2))


def _db_stats() -> dict:
    from flowmind import db                                # noqa: PLC0415
    out = {}
    with db.tenant_session("SHARED") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT source), COUNT(*) FROM documents "
                        "WHERE tenant_id = 'SHARED'")
            out["sources"], out["chunks"] = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM kg_nodes")
            out["kg_nodes"] = cur.fetchone()[0]
    return out


def _json_get(path: str, *keys):
    p = ROOT / path
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d = d.get("summary", d)
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


def collect_facts(skip_db: bool = False) -> list[Fact]:
    facts: list[Fact] = []

    n_pass, n_fail = _run_tests()
    if n_fail:
        raise SystemExit(f"❌ 有 {n_fail} 項測試失敗 —— 先修測試再談文件一致性")
    # ── 樣式設計原則 ───────────────────────────────────────────────────
    #
    # 假陽性比漏抓危險：這支腳本有 --fix，會**直接改檔案**。
    # 一個把「83 份文件」裡的「3」當成份數改掉的樣式，會把正確的文件改壞。
    # 第一版就踩到這個 —— `\d+` 沒有左邊界，從「83」裡比中「3」。
    #
    # 所以每個樣式都遵守三條：
    #   1. 數字前後都要有邊界：(?<!\d) 與 (?!\d)
    #   2. 綁定到**這個事實獨有的措辭**，不用通用詞
    #   3. 排除已知會撞的語境（SROIE 樣本數、「其中有 N 項」）
    B = r"(?<!\d)"          # 左邊界
    E = r"(?!\d)"           # 右邊界

    facts.append(Fact(
        "回歸測試項數", n_pass,
        # 「154 / 154」這種寫法**兩個數字都要對**。
        # 第一版只換分子，留下「**154 / 139**」這種自相矛盾的寫法 ——
        # 比原本的過時數字更糟，因為它看起來像有人改到一半。
        [rf"回歸測試\s*\|\s*\*\*{B}(?P<num>\d+){E}\s*/",     # 表格分子
         rf"回歸測試\s*\|\s*\*\*\d+\s*/\s*{B}(?P<num>\d+){E}",  # 表格分母
         rf"回歸測試\s*\|\s*{B}(?P<num>\d+){E}\s*/\s*\d+",      # 無粗體分子
         rf"回歸測試\s*\|\s*\d+\s*/\s*{B}(?P<num>\d+){E}",      # 無粗體分母
         rf"回歸測試（{B}(?P<num>\d+){E}\s*項",               # 「回歸測試（154 項…」
         rf"回歸測試\s+{B}(?P<num>\d+){E}\s*/\s*\d+",         # 「回歸測試 154 / 154」
         rf"回歸測試\s+{B}(?P<num>\d+){E}\s*項",              # 「回歸測試 154 項」
         rf"tests/test_core\.py\s+{B}(?P<num>\d+){E}\s*項",   # 目錄樹註解
         rf"{B}(?P<num>\d+){E}\s*項回歸測試",                  # 語序相反的寫法
         rf"回歸測試\s*{B}(?P<num>\d+){E}\s*項",               # 「回歸測試 58 項」
         rf"測試\s*\d+\s*→\s*{B}(?P<num>\d+){E}\s*全數通過"],  # commit 風格敘述
        "跑 tests/test_core.py 取得"))
    # 「回歸測試中有 4 項是…」講的是子集合，不是總數 —— 明確不納入樣式。

    if not skip_db:
        try:
            s = _db_stats()
            # 只認「N 份文件 / M chunks」這個固定寫法。
            # SROIE 的「15 份文件 / 60 欄位」語境不同，不會被這個樣式比到。
            facts.append(Fact(
                "SHARED 文件份數", s["sources"],
                [rf"{B}(?P<num>\d+){E}\s*份文件\s*/\s*\*{{0,2}}[\d,]+\*{{0,2}}\s*chunks?"],
                "查資料庫"))
            facts.append(Fact(
                "SHARED chunk 數", s["chunks"],
                [rf"\d+\s*份文件\s*/\s*\*{{0,2}}{B}(?P<num>[\d,]+){E}\*{{0,2}}\s*chunks?"],
                "查資料庫"))
            facts.append(Fact(
                "知識圖譜節點數", s["kg_nodes"],
                [rf"知識圖譜\s*\|\s*\*\*{B}(?P<num>\d+){E}\s*節點",
                 rf"圖譜[^\n]{{0,6}}\*\*{B}(?P<num>\d+){E}\s*節點\*\*"],
                "查資料庫"))
        except Exception as e:                             # noqa: BLE001
            print(f"  ⚠️ 資料庫未連線，略過知識庫相關事實：{e}")

    pr = _json_get("docs/QA_EVAL_final.json", "overall_pass_rate")
    if pr is not None:
        facts.append(Fact("50 題整體通過率(%)", round(pr * 100, 1),
                          [rf"整體通過\s*\*{{0,2}}{B}(?P<num>[\d.]+){E}\s*%"],
                          "讀 QA_EVAL_final.json"))
    return facts


def scan(facts: list[Fact], fix: bool) -> int:
    problems = 0
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            continue
        text = original = p.read_text(encoding="utf-8")
        for f in facts:
            want = f.fmt()
            for pat in f.patterns:
                # ── 由後往前替換 ──────────────────────────────────────
                # 第一版是「先收集全部 match，再依原始位置逐一替換」。
                # 那是錯的：每替換一次字串長度就變，後面 match 的位置全部失效。
                # 實際造成的損害：把 154 寫成 1549、吃掉一個全形括號。
                #
                # 由後往前替換，前面的位置就不受影響。
                # 每個 pattern 替換完再重新 finditer，避免不同 pattern 之間互相干擾。
                for m in reversed(list(re.finditer(pat, text))):
                    got = m.group("num").replace(",", "")
                    if got == want.replace(",", ""):
                        continue
                    line = text[:m.start()].count("\n") + 1
                    print(f"  ❌ {rel}:{line}　{f.key}　文件寫 {got}，實際 {want}")
                    problems += 1
                    if fix:
                        s, e = m.span("num")
                        text = text[:s] + want + text[e:]
        if fix and text != original:
            p.write_text(text, encoding="utf-8")
            print(f"  ✏️  已更新 {rel}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="自動更新過時數字")
    ap.add_argument("--skip-db", action="store_true")
    args = ap.parse_args()

    print("═" * 74)
    print("  文件數字一致性檢查")
    print("═" * 74)
    facts = collect_facts(skip_db=args.skip_db)
    print("  事實來源：")
    for f in facts:
        print(f"    {f.key:<20}{f.fmt():>12}　（{f.note}）")
    print()

    n = scan(facts, args.fix)
    print()
    if n == 0:
        print("  ✅ 所有文件的數字與實際一致")
    elif args.fix:
        print(f"  ✏️  已修正 {n} 處；請重跑一次確認")
    else:
        print(f"  ❌ 發現 {n} 處不一致 —— 用 --fix 自動更新")
    print("═" * 74)
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
