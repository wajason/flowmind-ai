#!/usr/bin/env python3
"""
rag_query.py — FlowMind 供應鏈融資顧問查詢介面
=============================================================================
與一般 RAG 問答工具的三個根本差異：

1. 每一次查詢都必須指定 engagement（委任案）。
   沒有「查全部」這個選項 —— 因為在同時服務多家客戶的場域裡，
   「不小心查到別的客戶」不是使用體驗問題，是事故。
   隔離由 PostgreSQL Row-Level Security 強制，不是靠這支程式記得加條件。

2. 輸出不是一段文字，是一份可查核的證據包（Evidence / Confidence / Source / Reason）。
   模型講的每一句話都要附逐字摘錄，程式回頭到檢索文本裡比對；
   對不上的直接從答案裡移除，不留給使用者自己判斷。

3. 信心不足時系統會拒答，並明講缺什麼文件。
   這是刻意的產品決策：在授信場域，一個聽起來很篤定但沒有根據的答案，
   比「我不知道」有害得多。

用法：
  python rag_query.py --tenant CASE-0001 -q "我們下個月現金流夠不夠？"
  python rag_query.py --tenant CASE-0001                      # 互動模式
  python rag_query.py --tenant CASE-0001 -q "…" --json        # 給下游系統串接
  python rag_query.py --verify-isolation CASE-0001 CASE-9999  # 隔離證明
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowmind import (config, db, evidence, llm, metrics,            # noqa: E402
                      retrieval, textnorm)
from flowmind.evidence import Claim, EvidencePack                    # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# System prompt
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是一位資深的中小企業供應鏈融資顧問，服務對象是企業主與銀行授信人員。

【你的專業定位】
你熟悉台灣的應收帳款承購（factoring，含有追索權／無追索權）、供應商融資、
中小企業信用保證基金的保證要點與成數、以及銀行受理送件時實際會要求的文件。
你使用的是業界用語：授信、徵信、額度、動撥、帳齡、集中度、債權讓與通知。

【你的紅線】
你不做授信決策。你的產出是給人看、供人核的證據整理，最終一定由授信人員簽字。
你可以說「依此條件通常適用某方案」，不可以說「這筆一定過」。

【回答規則 — 你的輸出會被程式逐句驗證】
1. 只根據【檢索文本】回答。文本沒有的，就寫進 unknowns，不要靠常識補。
2. claims 陣列裡的每一筆，quote 必須是從檢索文本「一字不差」複製出來的片段。
   系統會把你的 quote 拿回原文做字串比對；對不上的那句話會被自動刪除，
   而且會拉低整份答案的信心分數。抄原文對你有利，改寫沒有好處。
3. source 必須寫檢索文本中標示的來源檔名，不要自己編一個看起來合理的檔名。
4. 區分「法規／商品條件」與「這家客戶自己的文件」。前者是通則，後者才是本案事實，
   在 answer 裡要讓讀的人分得出來哪句是哪種。
5. 涉及具體金額、天數、成數時，一律附上出處。這些數字會被拿去跟銀行談。
6. reason 用兩三句話說明你的推理路徑，以及你認為這份答案最脆弱的地方在哪。"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "quote": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["statement", "quote", "source"],
            },
        },
        "unknowns": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["answer", "claims", "unknowns", "reason"],
}


def build_context(chunks: list[retrieval.Chunk]) -> str:
    parts = []
    for c in chunks:
        scope = "公開法規／商品資料" if c.is_shared else "★本案客戶自有文件"
        parts.append(
            f"\n--- [來源檔名: {c.source} | 類別: {c.category} | {scope}] ---\n"
            f"{c.parent_content}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# 核心流程
# ══════════════════════════════════════════════════════════════════════════

def answer_deterministic(tenant_id: str, question: str,
                         metric_keys: list[str]) -> EvidencePack | None:
    """
    決定性路徑：這一題可以用算的，就不要交給語言模型。

    「最大買方占營收多少」「逾期多少」「現金流夠不夠」這類問題，
    需要把整批憑證加總後相除。RAG 只取回最相關的幾段文字，
    在設計上就不可能可靠地回答彙總問題 —— 給模型 6 張發票要它算出 90 張的占比，
    結果只有兩種：拒答，或編一個數字。

    而這些數字純 Python 算得出來，精確、可重算、零幻覺。
    信心分數給 1.0 不是自誇，是因為這裡根本沒有不確定性可言：
    同樣的檔案、同樣的公式，任何人跑都會得到同一個數字。
    """
    ms = metrics.compute(tenant_id, metric_keys, question=question)
    if not ms:
        return None

    pack = EvidencePack(question=question, tenant_id=tenant_id,
                        model="決定性運算（未使用語言模型）")
    pack.answer = metrics.render(ms)
    pack.confidence = 1.0
    pack.confidence_breakdown = {"deterministic": True,
                                 "note": "純算術結果，無檢索與生成環節，故不適用信心分解"}
    pack.sources = sorted({s for m in ms for s in m.sources})
    pack.reason = (
        f"本題命中決定性指標 {'、'.join(m.title for m in ms)}，"
        f"直接由程式自原始憑證計算，未經語言模型生成。"
        f"若需要法規依據或商品條件的說明，請另外提問（那類問題才會走 RAG）。")

    amount = evidence.largest_amount_mentioned(pack.answer)
    if amount >= config.HUMAN_REVIEW_AMOUNT_TWD:
        pack.needs_human_review = True
        pack.human_review_reason = (
            f"內容涉及金額約 NT${amount:,.0f}，達人工複核門檻，"
            f"對外提出前需授信人員覆核。")

    with db.tenant_session(tenant_id) as conn:
        db.write_audit(conn, tenant_id=tenant_id, action="compute",
                       query_text=question, doc_sources=pack.sources,
                       confidence=1.0, abstained=False)
    return pack


def answer_question(tenant_id: str, question: str, top_k: int = 8,
                    model: str | None = None, quiet: bool = False,
                    force_rag: bool = False) -> EvidencePack:
    # ── 先問：這題該不該給 RAG？ ──────────────────────────────────────
    if not force_rag:
        keys = metrics.route(question)
        if keys:
            if not quiet:
                print(f"\n⚙️  本題命中決定性指標 {keys} —— 走純運算路徑，不經語言模型。")
            pack = answer_deterministic(tenant_id, question, keys)
            if pack:
                return pack
            if not quiet:
                print("   （該委任案缺少計算所需的原始憑證，改走 RAG。）")

    pack = EvidencePack(question=question, tenant_id=tenant_id,
                        model=model or config.ADVISOR_MODEL)

    with db.tenant_session(tenant_id) as conn:
        chunks = retrieval.hybrid_search(conn, question, top_k=top_k)

        if not chunks:
            pack.abstained = True
            pack.abstain_reason = (
                f"委任案 {tenant_id} 的知識庫中找不到任何相關文件。"
                f"請先確認已執行 data_update_finance.py --tenant {tenant_id}。")
            db.write_audit(conn, tenant_id=tenant_id, action="retrieve",
                           query_text=question, doc_sources=[],
                           confidence=0.0, abstained=True)
            return pack

        pack.retrieval = retrieval.retrieval_diagnostics(chunks)
        pack.sources = sorted({c.source for c in chunks})

        if not quiet:
            render_retrieval_panel(chunks, pack.retrieval)

        prompt = (f"【檢索文本】{build_context(chunks)}\n\n"
                  f"【使用者問題】\n{question}")
        obj, diag = llm.extract_json(prompt, schema=ANSWER_SCHEMA,
                                     system=SYSTEM_PROMPT, model=pack.model,
                                     num_ctx=16384)

        if not isinstance(obj, dict):
            pack.abstained = True
            pack.abstain_reason = f"模型輸出無法解析為結構化格式：{diag.get('error')}"
            return pack

        # 簡轉繁在引用驗證「之前」做：知識庫全是繁體的台灣公文與行庫資料，
        # 把模型偶爾漏出的簡體字轉回繁體，只會讓逐字比對更準。
        # 引用（quote）也一起轉，才不會因為一個「实」字對不上原文的「實」而被誤判為幻覺。
        tw = textnorm.to_traditional
        pack.answer = tw(str(obj.get("answer") or ""))
        pack.reason = tw(str(obj.get("reason") or ""))
        pack.unknowns = [tw(str(u)) for u in (obj.get("unknowns") or [])]
        pack.claims = [
            Claim(statement=tw(str(c.get("statement") or "")),
                  quote=tw(str(c.get("quote") or "")),
                  source=str(c.get("source") or ""))
            for c in (obj.get("claims") or []) if isinstance(c, dict)
        ]

        # ── 這三行是整個產品的核心：驗證 → 計分 → 閘門 ──────────────
        evidence.verify_claims(pack.claims, chunks)
        pack.confidence, pack.confidence_breakdown = evidence.compute_confidence(
            pack.claims, chunks)
        evidence.strip_ungrounded(pack)
        evidence.apply_gates(pack)

        db.write_audit(conn, tenant_id=tenant_id, action="answer",
                       query_text=question,
                       doc_sources=pack.sources,
                       confidence=pack.confidence, abstained=pack.abstained)
    return pack


# ══════════════════════════════════════════════════════════════════════════
# 呈現
# ══════════════════════════════════════════════════════════════════════════

def render_retrieval_panel(chunks: list[retrieval.Chunk], diag: dict) -> None:
    print("\n" + "═" * 80)
    print("📊 檢索透明度面板（Hybrid：Dense 向量 + 中文 bigram BM25，RRF 融合）")
    print("═" * 80)
    for i, c in enumerate(chunks, 1):
        scope = "公開" if c.is_shared else "本案"
        print(f"[{i}] 📂 {c.source}#{c.chunk_index}　{c.category}（{scope}）")
        print(f"    ├─ RRF {c.rrf_score:.5f}　Dense {c.dense_score:.4f}　"
              f"Sparse {c.sparse_score:.4f}　版本 {c.freshness_label}")
        print(f"    └─ {c.parent_content[:88].replace(chr(10), ' ')}…")
    print("─" * 80)
    dropped = chunks[0].metadata.get("_dropped_superseded") if chunks else None
    if dropped:
        print(f"  🕓 已排除舊版本文件：{'、'.join(dropped)}")
        print(f"     （引用舊版規定回答新問題，引用驗證仍會給 100 分 —— "
              f"這類錯誤只能靠版本管理擋）")
    print(f"  來源多樣性 {diag['distinct_sources']} 份　"
          f"本案文件 {diag['own_docs']}／公開資料 {diag['shared_docs']}")
    # 這一行是靜默失效的偵測器：中文分詞一旦壞掉，sparse 會長期掛 0
    warn = "" if diag["sparse_contributing"] else "  ⚠ 稀疏檢索 0 命中，請檢查中文分詞"
    print(f"  雙路貢獻：Dense {diag['dense_contributing']}　"
          f"Sparse {diag['sparse_contributing']}{warn}")
    print("═" * 80 + "\n")


VERDICT_ICON = {"exact": "✅", "near": "🟢", "wrong_source": "🟠", "unverifiable": "🔴"}


def render_pack(pack: EvidencePack) -> None:
    print("═" * 80)
    if pack.abstained:
        print("⛔ 系統選擇不回答")
        print("═" * 80)
        print(f"\n{pack.abstain_reason}\n")
        if pack.unknowns:
            print("需要補充的資訊：")
            for u in pack.unknowns:
                print(f"  · {u}")
        print("\n（在授信場域，說不知道比說一個沒有根據的答案安全。）")
        print("═" * 80)
        return

    print(f"🤖 供應鏈融資顧問回覆　信心 {pack.confidence:.2f}")
    print("═" * 80)
    print(f"\n{pack.answer}\n")

    if pack.claims:
        print("─" * 80)
        print("📌 主張與證據（每一筆都經過原文字串比對）")
        for i, c in enumerate(pack.claims, 1):
            icon = VERDICT_ICON.get(c.verdict.value, "❓")
            print(f"\n{icon} [{i}] {c.statement}")
            print(f"     引用：「{c.quote[:100]}」")
            print(f"     出處：{c.source}　驗證：{c.verdict.value}"
                  f"（比對分數 {c.match_score:.0f}）")
            if c.verdict.value == "wrong_source":
                print(f"     ⚠ 這段話確實存在，但在 {c.matched_source}，不在它宣稱的來源")

    if pack.unknowns:
        print("\n" + "─" * 80)
        print("❔ 查無資料／已移除的未驗證敘述")
        for u in pack.unknowns:
            print(f"  · {u}")

    bd = pack.confidence_breakdown
    print("\n" + "─" * 80)
    if bd.get("deterministic"):
        # 決定性路徑沒有檢索也沒有生成，套用信心分解會顯示一排 0，反而誤導。
        print("🔍 信心 1.00：本結果為純算術計算，未經檢索與語言模型生成。")
        print("   同樣的憑證檔案、同樣的公式，任何人重跑都會得到完全相同的數字。")
        if pack.reason:
            print("\n" + "─" * 80)
            print(f"🧠 {pack.reason}")
        if pack.needs_human_review:
            print("\n" + "─" * 80)
            print(f"👤 {pack.human_review_reason}")
        print("\n" + "═" * 80)
        print(f"  來源：{'、'.join(pack.sources)}")
        print(f"  本次查詢已寫入稽核軌跡（engagement={pack.tenant_id}）")
        print("═" * 80)
        return

    print("🔍 信心分數組成（權重公開於 flowmind/evidence.py，可自行重算）")
    print(f"  引用驗證通過率 {bd.get('citation_integrity', 0):.2f} × {evidence.W_CITATION}")
    print(f"  檢索強度       {bd.get('retrieval_strength', 0):.2f} × {evidence.W_RETRIEVAL}")
    print(f"  交叉佐證來源數 {bd.get('corroboration', 0):.2f} × {evidence.W_CORROBORATION}")
    print(f"  雙路檢索健康度 {bd.get('sparse_health', 0):.2f} × {evidence.W_SPARSE_HEALTH}")
    if bd.get("hallucinated_claims"):
        print(f"  ⚠ 偵測到 {bd['hallucinated_claims']} 句無法驗證的敘述，"
              f"信心上限已被壓到 0.50")

    if pack.reason:
        print("\n" + "─" * 80)
        print(f"🧠 推理路徑：{pack.reason}")

    if pack.needs_human_review:
        print("\n" + "─" * 80)
        print(f"👤 {pack.human_review_reason}")

    print("\n" + "═" * 80)
    print(f"  來源：{'、'.join(pack.sources)}")
    print(f"  本次查詢已寫入稽核軌跡（engagement={pack.tenant_id}）")
    print("═" * 80)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def interactive(tenant_id: str, top_k: int, model: str | None) -> None:
    info = next((e for e in db.list_engagements() if e["tenant_id"] == tenant_id), None)
    print("\n" + "═" * 80)
    print(f"🚀 FlowMind 供應鏈融資顧問")
    if info:
        print(f"   委任案：{tenant_id}｜{info['client_name']}｜{info['engagement_type']}")
        print(f"   可檢索：{info['docs']} 份文件 / {info['chunks']} 個 chunk"
              f"（另可讀取 SHARED 公開知識庫）")
    print(f"   模型：{model or config.ADVISOR_MODEL}｜拒答門檻 "
          f"{config.CONFIDENCE_ABSTAIN_THRESHOLD}")
    print("   輸入 exit 離開")
    print("═" * 80)

    while True:
        try:
            q = input(f"\n🧑 [{tenant_id}] 你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再見")
            return
        if q.lower() in ("exit", "quit", "q"):
            print("👋 再見")
            return
        if not q:
            continue
        try:
            render_pack(answer_question(tenant_id, q, top_k, model))
        except Exception as e:                        # noqa: BLE001
            print(f"\n❌ 發生錯誤：{e}")


def main():
    ap = argparse.ArgumentParser(description="FlowMind 供應鏈融資顧問查詢介面")
    ap.add_argument("--tenant", "-t", help="委任案代號，例如 CASE-0001。必填。")
    ap.add_argument("--query", "-q", help="單次查詢；不給則進入互動模式")
    ap.add_argument("--top-k", "-k", type=int, default=8)
    ap.add_argument("--model", "-m", default=None)
    ap.add_argument("--json", action="store_true", help="輸出 JSON 供下游系統串接")
    ap.add_argument("--force-rag", action="store_true",
                    help="略過決定性路由，強制走 RAG（用於比較兩條路徑的差異）")
    ap.add_argument("--list", action="store_true", help="列出所有委任案")
    ap.add_argument("--verify-isolation", nargs=2, metavar=("A", "B"),
                    help="執行跨委任案隔離證明：以 A 的身分嘗試讀寫 B 的資料")
    ap.add_argument("--verify-audit", action="store_true",
                    help="驗證稽核軌跡雜湊鏈是否完整未被竄改")
    args = ap.parse_args()

    if args.list:
        for e in db.list_engagements():
            print(f"{e['tenant_id']:<12} {e['client_name']:<28} "
                  f"{e['engagement_type']:<22} {e['docs']} 份 / {e['chunks']} chunks")
        return

    if args.verify_audit:
        ok, n, bad = db.verify_audit_chain()
        print(f"稽核軌跡：{n} 筆　雜湊鏈{'完整 ✅' if ok else f'在 id={bad} 處斷裂 ❌'}")
        return

    if args.verify_isolation:
        a, b = args.verify_isolation
        r = db.verify_isolation(a, b)
        print("\n" + "═" * 78)
        print(f"  跨委任案隔離證明：以 {a} 的身分嘗試存取 {b}")
        print("═" * 78)
        print(f"  {b} 實際存在的 chunk 數（admin 繞過 RLS 查得）：{r['b_rows_actually_exist']}")
        print(f"  以 {a} 身分下『無 WHERE 條件』查詢，看得到的 tenant：{r['visible_to_a']}")
        print(f"  是否外洩 {b} 的資料：{'❌ 是' if r['leak_detected'] else '✅ 否'}")
        print(f"  嘗試寫入標記為 {b} 的資料："
              f"{'✅ 被資料庫拒絕' if r['cross_tenant_write_blocked'] else '❌ 竟然成功'}")
        print("─" * 78)
        verdict_label = {"passed": "✅ 隔離有效",
                         "failed": "❌ 隔離失效，不得上線",
                         "inconclusive": "⚠️  測試無效（非隔離失效）"}
        print(f"  結論：{verdict_label.get(r.get('verdict'), '?')}")
        print(f"  {r.get('note', '')}")
        print("  注意查詢語句本身沒有任何 tenant 條件 —— 過濾是由 PostgreSQL")
        print("  Row-Level Security 強制執行的，不依賴應用程式開發者記得加。")
        print("═" * 78)
        return

    if not args.tenant:
        ap.error("--tenant 為必填。系統不接受未指定委任案的查詢。")

    if args.query:
        pack = answer_question(args.tenant, args.query, args.top_k,
                               args.model, quiet=args.json,
                               force_rag=args.force_rag)
        print(pack.to_json() if args.json else "")
        if not args.json:
            render_pack(pack)
    else:
        interactive(args.tenant, args.top_k, args.model)


if __name__ == "__main__":
    main()
