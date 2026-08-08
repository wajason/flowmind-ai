#!/usr/bin/env python3
"""
diagnose_abstention.py — 過度保守的根因診斷
=============================================================================
【要回答的問題】

50 題評測中，22 個失敗有 **14 個是「本題有答案卻拒答」**。
過度保守是壓倒性的失敗模式，但「保守」本身不是缺陷 ——
在授信場域拒答比亂答安全。真正要問的是：

    **拒答是因為證據真的不足，還是因為我們的驗證器把好答案也擋掉了？**

這兩者的處理方式完全相反，混在一起看只會得到「調門檻」這個錯誤結論。

【診斷方法：把責任歸屬拆開】

初步診斷已確認壓低信心的是 `citation_integrity`（引用逐字驗證的通過率），
不是新的斷言層級佐證，也不是門檻設定。所以往下要分辨的是：

    A. 模型的錯 —— 它「引用」的那段話原文裡根本沒有（改寫、拼湊、幻覺）
    B. 驗證器的錯 —— 原文裡**有**，只是我們比對不到
       （空白、全形半形、標點、跨 chunk 斷裂、刪節號）

分辨方式：對每一條驗不過的引用，用**逐步放寬**的比對再試一次，
看它在哪一層才被找到。在哪一層被找到，就說明是哪一種問題：

    L0 原樣比對          → 原本就該過（不會出現在失敗清單）
    L1 正規化後比對      → **驗證器的錯**：全形半形/空白/標點差異
    L2 忽略所有標點      → **驗證器的錯**：標點處理
    L3 最長公共子串 ≥80% → 灰色地帶：模型改寫了一點，但實質忠於原文
    L4 都找不到          → **模型的錯**：原文真的沒有這句話

**只有 L4 才是模型的問題。** L1/L2 是我們該修的；L3 需要人判斷要不要放行。

【為什麼不直接調門檻】

調門檻會讓 L4（真幻覺）跟著一起通過。
過度保守與幻覺是同一個門檻的兩端，往任何一端調都會犧牲另一端 ——
唯一能同時改善兩者的方式，是**把量測本身修正確**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_query                                          # noqa: E402
from flowmind import config, evidence                     # noqa: E402

EVAL_SET = config.DATA_DIR / "evalset" / "zh_finance_qa.jsonl"

_PUNCT = "，。、；：「」『』（）()《》〈〉！？…—－-·　 \t\n\r"


def _n1(s: str) -> str:
    """L1：全形半形 + 空白正規化。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def _n2(s: str) -> str:
    """L2：再拿掉所有標點。"""
    return "".join(ch for ch in _n1(s) if ch not in _PUNCT)


def _lcs_ratio(a: str, b: str) -> float:
    """最長公共子串比例（以較短者為分母）。"""
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return m.size / min(len(a), len(b))


def classify(quote: str, haystacks: list[str]) -> tuple[str, float]:
    """回傳 (層級, 最佳相似度)。"""
    q0 = quote or ""
    if not q0.strip():
        return "L4", 0.0

    for h in haystacks:
        if q0 in h:
            return "L0", 1.0
    q1 = _n1(q0)
    for h in haystacks:
        if q1 and q1 in _n1(h):
            return "L1", 1.0
    q2 = _n2(q0)
    for h in haystacks:
        if q2 and q2 in _n2(h):
            return "L2", 1.0

    best = max((_lcs_ratio(q2, _n2(h)) for h in haystacks), default=0.0)
    return ("L3" if best >= 0.80 else "L4"), best


LAYER_MEANING = {
    "L0": ("原樣就比對得到", "不該出現在失敗清單 —— 若出現代表驗證流程有 bug"),
    "L1": ("正規化後找到", "**驗證器的錯**：全形半形或空白差異"),
    "L2": ("忽略標點後找到", "**驗證器的錯**：標點處理"),
    "L3": ("實質相符但有改寫", "灰色地帶：需人判斷是否放行"),
    "L4": ("原文真的沒有", "**模型的錯**：改寫、拼湊或幻覺"),
}


def load_questions(limit: int | None) -> list[dict]:
    if not EVAL_SET.exists():
        raise FileNotFoundError(f"找不到評測集：{EVAL_SET}")
    rows = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="SHARED")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="docs/ABSTENTION_DIAGNOSIS.json")
    args = ap.parse_args()

    rows = load_questions(args.limit or None)
    print("═" * 78)
    print("  過度保守根因診斷：是證據不足，還是驗證器擋掉了好答案？")
    print("═" * 78)
    print(f"  題數 {len(rows)}　模型 {config.EXTRACT_MODEL}\n")

    tally: dict[str, int] = {k: 0 for k in LAYER_MEANING}
    details, abstained_n = [], 0

    for i, row in enumerate(rows, 1):
        q = row.get("q") or row.get("question")
        if not q:
            continue
        # 只看「本來就該答得出來卻拒答」的題目 —— 那才叫過度保守。
        # must_abstain 為 true 的題目拒答是**正確**的，混進來會稀釋掉訊號。
        if row.get("must_abstain"):
            continue

        import io, contextlib                             # noqa: PLC0415
        with contextlib.redirect_stdout(io.StringIO()):
            pack = rag_query.answer_question(args.tenant, q)
        if not pack.abstain_reason:
            continue
        abstained_n += 1

        hay = [c.parent_content for c in pack.chunks] + \
              [c.child_content for c in pack.chunks]
        for c in pack.claims:
            if c.verdict in (evidence.Verdict.EXACT, evidence.Verdict.NEAR):
                continue
            layer, sim = classify(c.quote, hay)
            tally[layer] += 1
            details.append({
                "question": q[:70], "layer": layer, "similarity": round(sim, 3),
                "statement": c.statement[:90], "quote": c.quote[:120],
                "claimed_source": c.source,
            })
        print(f"  [{i:>2}] 拒答　{q[:52]}")

    total = sum(tally.values())
    print("\n" + "═" * 78)
    print(f"  拒答題數 {abstained_n}　驗不過的引用共 {total} 條")
    print("═" * 78)
    for k in ["L0", "L1", "L2", "L3", "L4"]:
        name, meaning = LAYER_MEANING[k]
        pct = f"{tally[k]/total:.1%}" if total else "—"
        print(f"  {k}　{name:<14}{tally[k]:>4} 條（{pct:>6}）　{meaning}")

    fixable = tally["L1"] + tally["L2"]
    print("\n" + "─" * 78)
    if total and fixable / total >= 0.30:
        print(f"""  判讀：{fixable}/{total} 條（{fixable/total:.0%}）是**驗證器可修的比對問題**。
  這些不該算模型的錯，修正比對規則就能同時降低過度保守
  而**不會**放寬對真幻覺（L4）的把關。""")
    elif total and tally["L4"] / total >= 0.60:
        print(f"""  判讀：{tally['L4']}/{total} 條（{tally['L4']/total:.0%}）是**模型真的沒有忠於原文**。
  拒答是正確行為。要改善只能換模型或改 prompt，
  **不能調門檻** —— 調了會讓真幻覺一起通過。""")
    else:
        print("  判讀：責任分散在多層，需逐條檢視 L3 灰色地帶。")
    print("─" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"model": config.EXTRACT_MODEL, "abstained_questions": abstained_n,
         "tally": tally, "details": details},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
