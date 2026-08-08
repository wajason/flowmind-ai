#!/usr/bin/env python3
"""
check_answer_reproducibility.py — 端到端答案可重現性檢查
=============================================================================
【為什麼需要這支腳本】

`flowmind/drift.py` 量的是**單一 prompt 重複呼叫**的一致性，
`scripts/model_matrix.py` 就是用它測出 gemma4:26b 位元級 100%。

但那是在**短 prompt、無前文**的條件下測的。
真實查詢的條件完全不同：8 個 chunk、數千字脈絡，
而且同一個服務行程裡前面已經跑過別的問題。

實測時觀察到一個必須釐清的現象：同一個問題、同樣 T=0、
同一個模型，在不同執行情境下 citation_integrity 出現 0.500 與 0.667 的差異。
若屬實，代表**信心分數不可重現**，而我們整套稽核論述
（「可以回答當初這個建議是根據什麼給的」）就站不住。

這支腳本用兩個對照條件把原因隔離出來：

    A. 同一行程內重複 N 次              → 測純粹的重複呼叫
    B. 每次都開新行程、且前面不跑別的題  → 測跨行程
    C. 每次都開新行程、但先跑一題暖身    → 測「前一題是否污染後一題」

C 是關鍵：Ollama 的 KV cache 與批次排程狀態會隨前文改變，
這正是 IBM 論文（arXiv:2511.07585）指出的非決定性來源 ——
它來自**服務條件**，不是來自取樣溫度。

Usage:
    python scripts/check_answer_reproducibility.py
    python scripts/check_answer_reproducibility.py --runs 5
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q = "信保基金的保證手續費年費率最低是多少？在什麼情況下會被酌增？"
WARMUP = "債權讓與要對債務人生效，依民法需要什麼要件？"


def probe(question: str, warmup: str | None = None) -> dict:
    """在**目前這個行程**跑一次，回傳可比較的指紋。"""
    import rag_query                                   # noqa: PLC0415

    if warmup:
        with contextlib.redirect_stdout(io.StringIO()):
            rag_query.answer_question("SHARED", warmup)

    with contextlib.redirect_stdout(io.StringIO()):
        p = rag_query.answer_question("SHARED", question)

    bd = p.confidence_breakdown
    return {
        "confidence": p.confidence,
        "citation_integrity": bd["citation_integrity"],
        "n_claims": len(p.claims),
        "abstained": bool(p.abstain_reason),
        # 主張的敘述串接後取雜湊：比答案雜湊有用，
        # 因為拒答時答案會被清空，雜湊全變成空字串的雜湊而失去鑑別力
        "claims_text": " | ".join(c.statement for c in p.claims)[:400],
        "sources": sorted({c.source for c in p.chunks}) if hasattr(p, "chunks") else [],
    }


def _child(question: str, warmup: str | None) -> dict:
    """開新行程跑一次。"""
    code = (
        "import sys,json;sys.path.insert(0,%r);"
        "sys.argv=['x'];"
        "from scripts.check_answer_reproducibility import probe;"
        "print('@@'+json.dumps(probe(%r,%r),ensure_ascii=False))"
        % (str(ROOT), question, warmup)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, encoding="utf-8", cwd=str(ROOT), timeout=1800)
    for line in (r.stdout or "").splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    raise RuntimeError(f"子行程沒有回傳結果：{(r.stderr or '')[-400:]}")


def report(name: str, results: list[dict]) -> bool:
    uniq_conf = sorted({r["confidence"] for r in results})
    uniq_ci = sorted({r["citation_integrity"] for r in results})
    uniq_claims = {r["claims_text"] for r in results}
    stable = len(uniq_conf) == 1 and len(uniq_claims) == 1

    print(f"\n  {name}")
    print("  " + "─" * 72)
    for i, r in enumerate(results, 1):
        print(f"    第{i}次  信心 {r['confidence']:.3f}　"
              f"引用完整度 {r['citation_integrity']:.3f}　"
              f"主張 {r['n_claims']}　拒答 {r['abstained']}")
    print(f"    → 信心唯一值 {uniq_conf}　引用唯一值 {uniq_ci}　"
          f"主張文字 {len(uniq_claims)} 種")
    print(f"    → {'✅ 完全可重現' if stable else '⚠️ 不可重現'}")
    return stable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print("═" * 78)
    print("  端到端答案可重現性檢查（真實查詢條件，非短 prompt）")
    print("═" * 78)
    print(f"  問題：{Q}")
    print(f"  重複次數：{args.runs}")

    a = [probe(Q) for _ in range(args.runs)]
    ok_a = report("A. 同一行程內重複", a)

    b = [_child(Q, None) for _ in range(args.runs)]
    ok_b = report("B. 每次開新行程（前面不跑別題）", b)

    c = [_child(Q, WARMUP) for _ in range(args.runs)]
    ok_c = report("C. 每次開新行程，但先跑一題暖身", c)

    cross = sorted({r["confidence"] for r in a + b + c})

    print("\n" + "═" * 78)
    print("  判讀")
    print("═" * 78)
    if ok_a and ok_b and ok_c and len(cross) == 1:
        print("""
  三個條件下信心分數完全一致 → 端到端可重現。
  「可以回答當初這個建議是根據什麼給的」這個稽核宣稱成立。
""")
        rc = 0
    else:
        print(f"""
  ⚠️ 跨條件出現不同的信心分數：{cross}

  這代表非決定性來自**服務條件**（KV cache 狀態、批次排程），
  而不是取樣溫度 —— 正是 IBM 論文（arXiv:2511.07585）指出的現象。
  短 prompt 測出的 100% 一致性，在真實長脈絡查詢下不成立。

  影響：信心分數在不同執行條件下可能不同，
  因此稽核所需的可重現性**必須靠保存輸入輸出快照**達成，
  不能只靠「T=0 所以可重現」這個假設。
""")
        rc = 1

    if ok_a and not (ok_b and ok_c):
        print("  註：同一行程內穩定、跨行程不穩定 —— 差異來自服務端狀態而非程式邏輯。\n")

    print("═" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
