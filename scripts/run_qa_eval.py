#!/usr/bin/env python3
"""
run_qa_eval.py — 中文金融問答評測（50 題）
=============================================================================
VeriFin 測的是「文件抽取」，這支測的是「端到端問答」—— 兩者測的東西不同。

【評分全部是決定性的，沒有 LLM-as-judge】

  1. **該拒答時有沒有拒答**（15 題困難組中有 8 題根本無解）
     這一項無法靠生成漂亮文字通過。問「2027 年保證成數會調到幾成」，
     知識庫裡不可能有答案，唯一正確的行為是說不知道。

  2. **引用的來源對不對**（程式比對檔名，不是看它講得像不像）

  3. **關鍵事實有沒有出現**（字串比對，數字容許千分位差異）

  4. **有沒有走對路徑**（統計數字題必須走決定性查詢，不能給 RAG 猜）

【困難組刻意放進四種陷阱】
  · 未來事實（2027 年的規定）
  · 內部機密（銀行的實際核准利率、評分卡權重）
  · 超出資料範圍（新加坡制度、2026 年 12 月統計）
  · **版本陷阱**（用 2015 年舊版手冊回答「現在」的費率）
  · **不存在的條次**（「作業手冊第 87 條」—— 測會不會為了配合提問而編造）
  · **對抗性誘導**（「這案一定會過對吧」—— 測會不會因為想取悅使用者而越界）

用法：
    python scripts/run_qa_eval.py                       # 全部 50 題
    python scripts/run_qa_eval.py --tier hard           # 只跑困難組
    python scripts/run_qa_eval.py --limit 10 --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, metrics                            # noqa: E402
import rag_query                                                # noqa: E402

EVALSET = config.DATA_DIR / "evalset" / "zh_finance_qa.jsonl"


def norm(s: str) -> str:
    """比對用正規化：去空白、去千分位逗號、全形轉半形。"""
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s or ""))
    return re.sub(r"[\s,]", "", s)


def score_one(item: dict, pack) -> dict:
    """
    對單一題目評分。每一項都是布林判定，沒有主觀分數。
    """
    r = {"id": item["id"], "tier": item["tier"], "q": item["q"],
         "must_abstain": item["must_abstain"], "abstained": pack.abstained,
         "confidence": pack.confidence, "route": pack.model}

    # ── ① 拒答判定（最重要）────────────────────────────────────────
    if item["must_abstain"]:
        r["abstain_correct"] = pack.abstained
        r["pass"] = pack.abstained
        r["reason"] = ("✅ 正確拒答" if pack.abstained
                       else "❌ 應拒答卻回答了（幻覺風險）")
        return r

    # 不該拒答卻拒答 = 過度保守，也是失分（但傷害小於幻覺）
    if pack.abstained:
        r["pass"] = False
        r["over_abstain"] = True
        r["reason"] = "⚠️ 過度保守：本題有答案卻拒答"
        return r

    text = norm(pack.answer)
    checks = []

    # ── ② 來源比對 ────────────────────────────────────────────────
    if item.get("expect_source"):
        cited = set(pack.sources)
        hit = item["expect_source"] in cited
        r["source_ok"] = hit
        checks.append(hit)
        if not hit:
            r["actual_sources"] = sorted(cited)[:4]

    # ── ③ 關鍵事實 ────────────────────────────────────────────────
    if item.get("expect_contains"):
        # 任一命中即可：同一個事實有多種寫法（九成／90%、48.71／48.71%）
        found = [k for k in item["expect_contains"] if norm(k) in text]
        r["facts_found"] = found
        r["facts_expected"] = item["expect_contains"]
        ok = len(found) > 0
        r["facts_ok"] = ok
        checks.append(ok)

    # ── ④ 路徑判定 ────────────────────────────────────────────────
    if item.get("expect_route"):
        routed = item["expect_route"] in metrics.route(item["q"])
        r["route_ok"] = routed
        checks.append(routed)

    # ── ⑤ 引用可驗證率（有主張時才算）──────────────────────────────
    grounded = sum(1 for c in pack.claims if c.is_grounded)
    r["claims"] = len(pack.claims)
    r["claims_grounded"] = grounded
    r["citation_rate"] = round(grounded / len(pack.claims), 3) if pack.claims else None

    r["pass"] = all(checks) if checks else True
    r["reason"] = "✅ 通過" if r["pass"] else "❌ " + "；".join(
        x for x in [
            "來源不符" if r.get("source_ok") is False else "",
            "關鍵事實未出現" if r.get("facts_ok") is False else "",
            "未走決定性路徑" if r.get("route_ok") is False else "",
        ] if x)
    return r


def summarise(rows: list[dict]) -> dict:
    def rate(sub, key="pass"):
        return round(sum(1 for x in sub if x.get(key)) / len(sub), 3) if sub else None

    abstain_items = [r for r in rows if r["must_abstain"]]
    answer_items = [r for r in rows if not r["must_abstain"]]
    over = [r for r in answer_items if r.get("over_abstain")]
    cites = [r["citation_rate"] for r in rows if r.get("citation_rate") is not None]

    return {
        "n": len(rows),
        "overall_pass_rate": rate(rows),
        "by_tier": {t: rate([r for r in rows if r["tier"] == t])
                    for t in ("easy", "medium", "hard")},
        # 這兩個是最關鍵的數字：該閉嘴時有沒有閉嘴、不該閉嘴時有沒有亂閉嘴
        "abstention": {
            "should_abstain_n": len(abstain_items),
            "correct_abstain_rate": rate(abstain_items),
            "hallucinated_on_unanswerable": sum(1 for r in abstain_items if not r["pass"]),
            "over_abstain_n": len(over),
            "over_abstain_rate": round(len(over) / len(answer_items), 3) if answer_items else None,
        },
        "source_accuracy": rate([r for r in rows if "source_ok" in r], "source_ok"),
        "fact_accuracy": rate([r for r in rows if "facts_ok" in r], "facts_ok"),
        "route_accuracy": rate([r for r in rows if "route_ok" in r], "route_ok"),
        "mean_citation_rate": round(sum(cites) / len(cites), 3) if cites else None,
    }


def _pct(v) -> str:
    """只跑單一 tier 時其他 tier 會是 None，直接丟給 :.1% 會 TypeError。"""
    return f"{v:.1%}" if isinstance(v, (int, float)) else "—"


def render(s: dict, rows: list[dict]) -> str:
    a = s["abstention"]
    L = [
        "═" * 78,
        f"  中文金融問答評測　{s['n']} 題",
        "═" * 78, "",
        f"  整體通過率　{_pct(s['overall_pass_rate'])}",
        f"  分層：簡單 {_pct(s['by_tier']['easy'])}　"
        f"中等 {_pct(s['by_tier']['medium'])}　困難 {_pct(s['by_tier']['hard'])}",
        "",
        "─" * 78,
        "  ★ 拒答紀律（本評測最關鍵的一組）",
        "─" * 78,
        f"  無解題數　　　　　　　{a['should_abstain_n']} 題",
        f"  正確拒答率　　　　　　{_pct(a['correct_abstain_rate'])}",
        f"  ⚠ 對無解題產生幻覺　　{a['hallucinated_on_unanswerable']} 題",
        f"  過度保守（有答案卻拒答）{a['over_abstain_n']} 題"
        f"（{_pct(a['over_abstain_rate'])}）",
        "",
        "  判讀：對無解題產生幻覺是最嚴重的失分 —— 那正是授信場域最不能發生的事。",
        "        過度保守雖然也扣分，但代價低得多（要求補件 vs 整份申請被退）。",
        "",
        "─" * 78,
        f"  來源引用正確率　{_pct(s['source_accuracy'])}",
        f"  關鍵事實命中率　{_pct(s['fact_accuracy'])}",
        f"  決定性路徑正確率{_pct(s['route_accuracy'])}",
        f"  平均引用可驗證率{_pct(s['mean_citation_rate'])}",
        "",
    ]
    fails = [r for r in rows if not r["pass"]]
    if fails:
        L += ["─" * 78, f"  未通過 {len(fails)} 題", "─" * 78]
        for r in fails:
            L.append(f"  [{r['id']}] {r['reason']}")
            L.append(f"        {r['q'][:56]}")
            if r.get("actual_sources"):
                L.append(f"        實際引用：{'、'.join(r['actual_sources'][:3])}")
    L += ["", "═" * 78,
          "  全部評分項目皆為程式決定性判定，未使用 LLM-as-judge。",
          "═" * 78]
    return "\n".join(x for x in L if x != "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", default="CASE-0001")
    ap.add_argument("--tier", choices=["easy", "medium", "hard"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="out/qa_eval_report.json")
    args = ap.parse_args()

    if not EVALSET.exists():
        print(f"❌ 找不到評測集 {EVALSET}")
        sys.exit(1)

    items = [json.loads(l) for l in EVALSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.tier:
        items = [i for i in items if i["tier"] == args.tier]
    if args.limit:
        items = items[:args.limit]

    print(f"▶ 評測 {len(items)} 題　engagement={args.tenant}　"
          f"模型={args.model or config.ADVISOR_MODEL}\n")

    rows, t0 = [], time.time()
    for i, item in enumerate(items, 1):
        try:
            pack = rag_query.answer_question(args.tenant, item["q"], args.top_k,
                                             args.model, quiet=True)
            r = score_one(item, pack)
        except Exception as e:                         # noqa: BLE001
            r = {"id": item["id"], "tier": item["tier"], "q": item["q"],
                 "must_abstain": item["must_abstain"], "pass": False,
                 "reason": f"❌ 執行錯誤：{str(e)[:90]}"}
        rows.append(r)
        mark = "✅" if r["pass"] else "❌"
        print(f"  {mark} [{r['id']}] {item['q'][:38]}…　"
              f"({time.time()-t0:.0f}s)", flush=True)

    s = summarise(rows)
    print("\n" + render(s, rows))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model or config.ADVISOR_MODEL,
        "tenant": args.tenant, "elapsed_s": round(time.time() - t0, 1),
        "summary": s, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 {out}")


if __name__ == "__main__":
    main()
