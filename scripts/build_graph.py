#!/usr/bin/env python3
"""
build_graph.py — 建構知識圖譜、驗證多跳查詢、匯出 Obsidian
=============================================================================
Usage:
    python scripts/build_graph.py --rebuild
    python scripts/build_graph.py --scope-check "日本的保證成數是多少？"
    python scripts/build_graph.py --export-obsidian out/obsidian
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import config, db, graph                              # noqa: E402


def cmd_rebuild(tenants: list[str]) -> None:
    print("═" * 78)
    print("  建構知識圖譜")
    print("═" * 78)

    print("\n▶ SHARED：文件 → 法域／期間／主題（解 U-01 的範圍圖）")
    s = graph.build_shared_graph()
    print(f"    節點 {s.total_nodes}：" +
          "　".join(f"{k} {v}" for k, v in sorted(s.nodes.items())))
    print(f"    關係 {s.total_edges}：" +
          "　".join(f"{k} {v}" for k, v in sorted(s.edges.items())))

    for t in tenants:
        if t == "SHARED":
            continue
        print(f"\n▶ {t}：企業交易圖")
        g = graph.build_engagement_graph(t)
        if not g.total_nodes:
            print("    （無憑證資料，略過）")
            continue
        print(f"    節點 {g.total_nodes}　關係 {g.total_edges}")


def cmd_scope_check(question: str) -> None:
    """展示圖如何解決「主題相似但指涉對象不存在」。"""
    from flowmind import evidence

    # 從問題抽出可能的法域與主題
    juris = [j for j, kws in graph.JURISDICTIONS.items()
             if any(k in question for k in kws)]
    topics = [t for t, kws in graph.TOPICS.items()
              if any(k in question for k in kws)]
    if not juris:
        juris = ["台灣"]      # 未指定時預設本國

    r = graph.scope_check(question, juris, topics)

    print("═" * 84)
    print(f"  指涉範圍檢查　「{question}」")
    print("═" * 84, "")
    print(f"  問題指涉的法域：{'、'.join(juris)}")
    print(f"  問題指涉的主題：{'、'.join(topics) or '（未指定）'}\n")
    print("  " + "─" * 80)
    for j, v in r["jurisdictions"].items():
        mark = "✅" if v["covered"] else "⛔"
        print(f"  {mark} 法域「{j}」："
              + (f"知識庫有 {len(v['documents'])} 份**適用於此法域**的文件"
                 if v["covered"] else "**知識庫沒有適用於此法域的文件**"))
        for d in v["documents"][:3]:
            print(f"        · {d['source']}")
        if v.get("mentioned_in"):
            print(f"        ⚠️ 但有 {len(v['mentioned_in'])} 份文件**提及**此法域：")
            for m in v["mentioned_in"][:3]:
                print(f"            · {m['source']}（提及 {m['mentions']:.0f} 次）")
            print(f"        ⚠️ 「提及」不等於「適用」——"
                  f"那是國際比較性敘述，不是該國的制度規定。")
    for t, v in r["topics"].items():
        mark = "✅" if v["covered"] else "⛔"
        print(f"  {mark} 主題「{t}」："
              + (f"{len(v['documents'])} 份文件" if v["covered"] else "未涵蓋"))

    print("\n  " + "─" * 80)
    if r["answerable"]:
        print("  結論：✅ 知識庫涵蓋此問題的指涉範圍，可以作答")
    else:
        print(f"  結論：⛔ 知識庫**不涵蓋** {'、'.join(r['uncovered_jurisdictions'])}，"
              f"應拒答")
        print("        正確回應：說明本知識庫僅涵蓋台灣制度，"
              "並可提供台灣的對應規定供參考，")
        print("        但必須明確標示「這不是您問的數字」。")

    # 對照組：純字串比對會怎麼判
    print("\n  " + "─" * 80)
    print("  【與字串比對的差別】")
    print("    字串比對只能回答「文本裡有沒有出現『日本』」——")
    print("    白皮書把日本當比較對象提到過，字串比對就會誤判為「有涵蓋」。")
    print("    圖回答的是「有沒有一份**適用於**日本的文件」，")
    print("    那份白皮書的 applies_to 指向台灣，所以不會誤判。")
    print("═" * 84)


def cmd_relations(tenant: str) -> None:
    print("═" * 84)
    print(f"  關係網絡分析　{tenant}")
    print("═" * 84)

    circ = graph.circular_trades(tenant)
    print(f"\n▶ 循環交易偵測（A 開票給 B 且 B 也開票給 A）")
    if not circ:
        print("    未發現循環交易。")
    for c in circ:
        print(f"    ⚠️ {c['company_a']} ⇄ {c['company_b']}　"
              f"A→B {c['a_to_b_amount']:,.0f}　B→A {c['b_to_a_amount']:,.0f}")
    print("    （這是純憑證比對抓不到的：兩張發票各自完全合法，"
          "只有放進同一張圖才看得出是循環）")

    reps = graph.shared_representatives(tenant)
    print(f"\n▶ 共同負責人偵測（關係企業的直接證據）")
    if not reps:
        print("    無負責人資料。需匯入經濟部商工登記公示資料後才能分析。")
        print("    （目前 crosscheck 的 RELATED-01 只能靠統編前綴比對，屬弱訊號）")
    for r in reps:
        print(f"    ⚠️ {r['person']} 同時擔任 {r['company_count']} 家公司負責人："
              f"{'、'.join(r['companies'][:4])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tenants", nargs="*", default=["SHARED", "CASE-0001", "CASE-9999"])
    ap.add_argument("--scope-check", metavar="QUESTION")
    ap.add_argument("--relations", metavar="TENANT")
    ap.add_argument("--multi-hop", nargs=2, metavar=("TENANT", "LABEL"))
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--export-obsidian", metavar="DIR")
    ap.add_argument("--obsidian-tenant", default="CASE-0001")
    args = ap.parse_args()

    if args.rebuild:
        cmd_rebuild(args.tenants)
    if args.scope_check:
        cmd_scope_check(args.scope_check)
    if args.relations:
        cmd_relations(args.relations)
    if args.multi_hop:
        tenant, label = args.multi_hop
        rows = graph.multi_hop(tenant, label, args.hops)
        print(f"\n從「{label}」出發 {args.hops} 跳可達 {len(rows)} 個節點：")
        for r in rows[:30]:
            print(f"  {r['hops']} 跳　{r['node_type']:<12}{r['label'][:40]:<42}"
                  f"{'→'.join(r['edge_path'])}")
    if args.export_obsidian:
        d = Path(args.export_obsidian)
        w = graph.export_obsidian(args.obsidian_tenant, d)
        print(f"\n✅ Obsidian vault → {d.resolve()}")
        print(f"   節點 note {w['nodes']} 份、MOC {w['moc']} 份")
        print(f"   用 Obsidian 開啟該資料夾，切換到「關係圖檢視」即可探索。")

    if not any([args.rebuild, args.scope_check, args.relations,
                args.multi_hop, args.export_obsidian]):
        ap.print_help()


if __name__ == "__main__":
    main()
