#!/usr/bin/env python3
"""
demo_multidoc.py — 多文件不一致：系統怎麼處理「文件之間講得不完全一樣」
=============================================================================
【這個 demo 要證明的事】

大部分 RAG 產品有一個沒說出口的假設：**所有文件都是一致的**。
於是當三份文件從不同角度講同一件事時，它們會被揉成一句聽起來很順、
但沒有任何一份文件真正這樣說過的話。

在授信場域這特別危險：法規講的是「要件」、要點講的是「保證條件」、
銀行商品說明講的是「服務內容」。三者不衝突，但**回答的是不同問題**。
把它們混成一段話，等於製造一個沒有出處的新主張。

本 demo 跑三種情境，展示系統的處理方式：

    情境 A  多份文件講同一個數值　→ 佐證分數提高
    情境 B  文件從不同角度回答　　→ 分別標示出處，不合併
    情境 C  文件之間數值不一致　　→ 明確指出分歧，並降低信心

【零 LLM 的部分】
「哪幾份文件說了什麼」與「它們是否一致」由 evidence.claim_corroboration()
以決定性方式判定 —— 不是請模型自己評估「我的來源可不可信」。
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rag_query                                           # noqa: E402
from flowmind import config                                # noqa: E402

SCENARIOS = [
    ("A　同一個數值，多份文件都提到",
     "信保基金供應商融資的信用保證成數最高是幾成？",
     "要點寫明九成。若白皮書等文件也提到相同數值，佐證分數會提高；"
     "若只有一份文件提到，佐證分數就低 —— 這個差別是算出來的，不是寫死的。"),
    ("B　三份文件從不同角度回答同一個主題",
     "應收帳款融資這件事，法規、信保要點、銀行商品說明各自談什麼？",
     "民法談債權讓與的**生效要件**、信保要點談**保證條件**、"
     "銀行說明談**服務內容**。三者不衝突但回答的是不同問題。"
     "正確行為是分別標示出處，而不是揉成一句沒有出處的話。"),
    ("C　不同銀行的同類商品，內容並不相同",
     "不同銀行的應收帳款承購服務，內容完全一樣嗎？",
     "玉山／永豐／中國信託三份說明各有側重。"
     "正確行為是指出差異，而不是給一個統一答案。"),
]


def run_one(title: str, question: str, why: str) -> dict:
    print("\n" + "═" * 78)
    print(f"  情境 {title}")
    print("═" * 78)
    print(f"  問題：{question}")
    print(f"  這題在測什麼：{why}\n")

    with contextlib.redirect_stdout(io.StringIO()):
        pack = rag_query.answer_question("SHARED", question)

    bd = pack.confidence_breakdown
    cd = bd.get("corroboration_detail", {}) or {}
    print(f"  信心 {pack.confidence:.3f}"
          f"（拒答門檻 {config.CONFIDENCE_ABSTAIN_THRESHOLD}）"
          f"　{'⛔ 拒答' if pack.abstain_reason else '✅ 作答'}")
    print(f"  佐證 {bd.get('corroboration', 0):.3f}"
          f"　模式 {cd.get('mode', '—')}"
          f"　引用完整度 {bd.get('citation_integrity', 0):.3f}")

    for a in (cd.get("assertions") or [])[:4]:
        agree = len(a.get("agreeing_sources") or [])
        conflict = len(a.get("conflicting_sources") or [])
        flag = "⚠️ 有分歧" if conflict else "一致"
        print(f"    · {a.get('kind')}={a.get('value')}　"
              f"同意 {agree} 份　抵觸 {conflict} 份　{flag}")
        for s in (a.get("conflicting_sources") or [])[:3]:
            print(f"        抵觸來源：{s}")

    print(f"\n  引用的來源（{len(pack.sources)} 份）：")
    for s in pack.sources[:6]:
        print(f"    · {s}")

    if pack.abstain_reason:
        print(f"\n  拒答理由：{pack.abstain_reason[:150]}")
    else:
        print(f"\n  回答：{(pack.answer or '')[:220]}")

    if pack.unknowns:
        print(f"\n  已移除未通過驗證的敘述 {len(pack.unknowns)} 條")

    return {"question": question, "confidence": pack.confidence,
            "sources": len(pack.sources), "abstained": bool(pack.abstain_reason)}


def main() -> int:
    print("═" * 78)
    print("  多文件不一致：系統怎麼處理「文件之間講得不完全一樣」")
    print("═" * 78)
    print("""
  大部分 RAG 產品有一個沒說出口的假設：所有文件都是一致的。
  於是三份文件從不同角度講同一件事時，會被揉成一句聽起來很順、
  但沒有任何一份文件真正這樣說過的話。

  在授信場域這特別危險 —— 那等於製造一個沒有出處的新主張。""")

    results = [run_one(*s) for s in SCENARIOS]

    print("\n" + "═" * 78)
    print("  小結")
    print("═" * 78)
    for r in results:
        print(f"  信心 {r['confidence']:.3f}　來源 {r['sources']} 份　"
              f"{'拒答' if r['abstained'] else '作答'}　{r['question'][:40]}")
    print("""
  佐證分數由 evidence.claim_corroboration() 以決定性方式算出：
  抽出數值斷言，數有幾份**獨立文件**說了**同一個值**。
  三份都說「九成」比一份說「九成」可信；
  兩份說「九成」一份說「八成」則降低信心並具名指出分歧來源。

  這不是請模型自己評估「我的來源可不可信」——
  那等於讓被考的人自己改考卷。""")
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
