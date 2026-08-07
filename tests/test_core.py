#!/usr/bin/env python3
"""
test_core.py — 核心邏輯回歸測試（不需要資料庫、不需要 LLM）
=============================================================================
執行：  python tests/test_core.py

刻意寫成不依賴 pytest、不依賴資料庫、不依賴 Ollama 的獨立腳本 ——
這些測試保護的是系統裡「不該出錯」的那一層：
統一編號檢核碼、引用驗證、計分公式。
如果連這些都要先啟動 Docker 才能驗證，實務上就不會有人跑。

其中「引用驗證」的負向測試最重要：
它證明的不是「正確引用會通過」，而是**「錯誤引用會被擋下來」**。
一個只測正向案例的驗證器，跟沒有驗證器沒兩樣。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowmind import crosscheck, evidence, textnorm, verifin       # noqa: E402
from flowmind.evidence import Claim, Verdict                       # noqa: E402
from flowmind.retrieval import Chunk                               # noqa: E402
from flowmind.verifin import FieldPrediction                       # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}　{detail}")


def section(t: str) -> None:
    print(f"\n── {t} " + "─" * (66 - len(t)))


# ══════════════════════════════════════════════════════════════════════════
def test_tax_id() -> None:
    section("統一編號檢核碼（財政部演算法）")
    # 22099131 為公開可查的真實統編，用來驗證演算法本身是對的
    check("已知有效統編通過", textnorm.validate_tax_id("22099131"))
    check("末碼被改動後不通過", not textnorm.validate_tax_id("22099132"))
    check("全 1 不通過", not textnorm.validate_tax_id("11111111"))
    check("長度不足回 False", not textnorm.validate_tax_id("1234567"))
    check("None 回 False", not textnorm.validate_tax_id(None))
    check("含分隔符仍可正規化", textnorm.validate_tax_id("22-099-131"))


def test_cjk() -> None:
    section("中文 bigram 分詞")
    toks = textnorm.tokenize("應收帳款承購 AB-45678901")
    check("中文切成 bigram", "應收" in toks and "帳款" in toks, toks[:6])
    check("英數 token 完整保留", "ab-45678901" in toks, toks)
    q = textnorm.to_fts_query("無追索權")
    check("tsquery 以 OR 串接", "|" in q, q)
    check("tsquery 不含 tsquery 保留字元",
          not any(ch in q for ch in "&!():'"), q)


# ══════════════════════════════════════════════════════════════════════════
SOURCE_TEXT = (
    "四、信用保證之授信\n"
    "本要點所稱供應商融資，係指供應商憑中心廠商之訂單、發票（含電子發票）、"
    "支票、預約付款通知及其他經本基金同意得以佐證交易真實性之文件撥貸。\n"
    "六、信用保證成數\n"
    "信用保證成數最高九成。本基金得視個別中心廠商及供應商之信用狀況酌減。"
)


def _chunks() -> list[Chunk]:
    return [Chunk(source="要點.md", chunk_index=0, tenant_id="SHARED",
                  child_content=SOURCE_TEXT, parent_content=SOURCE_TEXT,
                  category="融資商品說明", dense_score=0.9,
                  sparse_score=0.1, rrf_score=0.03)]


def _verify(quote: str, source: str = "要點.md") -> Claim:
    c = Claim(statement="s", quote=quote, source=source)
    evidence.verify_claims([c], _chunks())
    return c


def test_citation_positive() -> None:
    section("引用驗證：應該通過的情況")
    check("逐字引用 → exact",
          _verify("信用保證成數最高九成。").verdict == Verdict.EXACT)
    check("標點與空白差異不影響",
          _verify("信用保證成數最高九成").verdict == Verdict.EXACT)
    check("刪節號省略中段仍通過",
          _verify("供應商憑中心廠商之訂單、發票…得以佐證交易真實性之文件撥貸。"
                  ).verdict == Verdict.EXACT)
    check("檔名多打空格仍能對應（打字誤差不該被當成幻覺）",
          _verify("信用保證成數最高九成。", source="要點 .md").verdict == Verdict.EXACT)


def test_citation_negative() -> None:
    section("引用驗證：必須被擋下來的情況（本檔最重要的一組）")
    check("原文沒說過的話 → unverifiable",
          _verify("信用保證成數最高十成。").verdict == Verdict.UNVERIFIABLE)
    check("聽起來很像但改寫過 → 不得判為 exact",
          _verify("本要點規定保證成數之上限為百分之九十。").verdict != Verdict.EXACT)
    # 這一項證明順序約束有效：兩個片段都真的存在於原文，
    # 但順序顛倒。若不檢查順序，模型就能把原文各處的碎片
    # 拼成一句原文從未表達過的意思，而每個片段都「驗證通過」。
    check("片段存在但順序顛倒 → 不得通過",
          _verify("信用保證成數最高九成…本要點所稱供應商融資"
                  ).verdict != Verdict.EXACT)
    check("引用過短 → unverifiable",
          _verify("九成").verdict == Verdict.UNVERIFIABLE)
    check("完全不存在的來源名 → 不得矇混通過",
          _verify("信用保證成數最高九成。",
                  source="完全無關的檔案.pdf").verdict != Verdict.EXACT)


def test_confidence_gate() -> None:
    section("信心分數與拒答閘門")
    good = [_verify("信用保證成數最高九成。")]
    bad = [_verify("信用保證成數最高十成。")]
    c_good, _ = evidence.compute_confidence(good, _chunks())
    c_bad, bd = evidence.compute_confidence(bad, _chunks())
    check("全部引用可驗證時信心較高", c_good > c_bad, f"{c_good} vs {c_bad}")
    check("出現幻覺時信心被壓在 0.5 以下", c_bad <= 0.50, c_bad)
    check("幻覺數量有被記錄", bd["hallucinated_claims"] == 1)


# ══════════════════════════════════════════════════════════════════════════
def test_hpes() -> None:
    section("VeriFin HPES 計分")
    src = "總計 1,197,000 元　買方 宏昇機械"

    def score(preds, gold):
        return verifin.score_document("d", "T", src, preds, gold)

    r_guess = score([FieldPrediction("total", "999"),
                     FieldPrediction("buyer", "亂猜公司")],
                    {"total": "1197000", "buyer": "宏昇機械"})
    r_abst = score([FieldPrediction("total", None),
                    FieldPrediction("buyer", None)],
                   {"total": "1197000", "buyer": "宏昇機械"})

    h_guess = verifin.hpes([r_guess])["hpes_raw"]
    h_abst = verifin.hpes([r_abst])["hpes_raw"]
    check("全部猜錯的分數低於全部留白", h_guess < h_abst, f"{h_guess} vs {h_abst}")
    check("全部留白為 0 分", h_abst == 0.0, h_abst)

    # λ=2 時損益平衡點應為 2/3
    check("損益平衡點 = λ/(1+λ)",
          abs(verifin.hpes([r_abst])["break_even_confidence"] - 2 / 3) < 1e-3)

    r_ok = score([FieldPrediction("total", "1,197,000")], {"total": "1197000"})
    check("金額千分位差異視為相符", r_ok.correct == 1)

    r_cord = score([FieldPrediction("tax_id", None)], {"tax_id": None})
    check("標準答案為 null 時留白算正確留白", r_cord.abstain_correct == 1)


def test_counterfactual() -> None:
    section("VeriFin 反事實擾動")
    import random
    text = "發票號碼 AB45678901，總計 1,197,000 元。"
    gold = {"invoice_number": "AB45678901", "total": "1,197,000"}
    new_text, new_gold, changed = verifin.make_counterfactual(text, gold, random.Random(1))
    check("有欄位被擾動", len(changed) > 0, changed)
    check("原始值已從文本中消失",
          all(str(gold[f]) not in new_text for f in changed))
    check("新標準答案出現在新文本中",
          all(str(new_gold[f]) in new_text for f in changed))
    check("擾動後格式保持一致（仍有千分位）",
          "," in new_gold.get("total", ",") if "total" in changed else True)


# ══════════════════════════════════════════════════════════════════════════
def test_crosscheck() -> None:
    section("決定性交叉驗證")
    base = dict(doc_type="AR_INVOICE", invoice_number="AB10000001",
                invoice_date="2026-01-01", due_date="2026-03-02",
                payment_terms_days=60, seller_ban="22099131",
                seller_name="賣方", buyer_ban="04595257", buyer_name="買方",
                sales_amount=100000, tax_amount=5000, total_amount=105000,
                status="PENDING")

    clean = [dict(base)]
    rep = crosscheck.run_all(clean)
    fired = {f["check_id"] for f in rep["findings"] if not f["passed"]}
    check("乾淨資料不觸發統編/加總/自我交易警示",
          not ({"TAXID-01", "ARITH-01", "FRAUD-01"} & fired), fired)

    self_deal = dict(base, buyer_ban=base["seller_ban"])
    check("自我交易被抓到",
          not crosscheck.check_self_dealing([self_deal])[0].passed)

    bad_sum = dict(base, total_amount=999999)
    check("加總不符被抓到",
          not crosscheck.check_invoice_arithmetic([bad_sum])[0].passed)

    bad_ban = dict(base, buyer_ban="04595258")
    check("無效統編被抓到",
          not crosscheck.check_tax_ids([bad_ban])[0].passed)

    dup = [dict(base), dict(base, invoice_number="AB10000002")]
    check("同買方同金額同日期的重複請款被抓到",
          not crosscheck.check_duplicates(dup)[1].passed)


def test_router() -> None:
    section("決定性問題路由")
    from flowmind import metrics
    check("集中度問題走決定性路徑",
          "concentration" in metrics.route("最大買方占營收多少？"))
    check("現金流問題走決定性路徑",
          "cashflow" in metrics.route("我們下個月現金流夠不夠？"))
    check("法規問題不走決定性路徑（應交給 RAG）",
          metrics.route("無追索權承購的法律依據是什麼？") == [],
          metrics.route("無追索權承購的法律依據是什麼？"))


def test_table_label_index() -> None:
    """
    這一組測的是一個真實踩過的坑（50 題評測 H11）：

    統計表某張 CSV 的列標籤裡有 "2015"（年份值），於是
    「依 2015 年版作業手冊，現在的費率是多少？」被判定為統計查詢，
    系統回了無關的年份數字**並給信心 1.00** ——
    因為決定性路徑固定給滿分信心，這等於整個繞過拒答閘門。

    版本陷阱題本來就該拒答，卻拿到滿分信心，是最糟的一種失敗。
    """
    section("統計表標籤索引（防止路由誤觸發）")
    from flowmind import tables

    check("純數字不得成為類別標籤", not tables._is_meaningful_label("2015"))
    check("年月不得成為類別標籤", not tables._is_meaningful_label("115年07月"))
    check("通用詞不得成為類別標籤", not tables._is_meaningful_label("合計"))
    check("兩字通用詞不得成為類別標籤", not tables._is_meaningful_label("其他"))
    check("真實行業名稱可以成為標籤", tables._is_meaningful_label("機械設備製造業"))
    check("縣市名稱可以成為標籤", tables._is_meaningful_label("台北市"))

    q_trap = "依 2015 年版作業手冊，現在的保證手續費率是多少？"
    check("版本陷阱題不得誤觸發統計路由",
          "statistics" not in metrics_route(q_trap), metrics_route(q_trap))
    q_real = "機械設備製造業的承保融資金額是多少？"
    check("真實統計題仍正確觸發", "statistics" in metrics_route(q_real))


def metrics_route(q: str):
    from flowmind import metrics
    return metrics.route(q)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 70)
    print("  FlowMind 核心邏輯回歸測試")
    print("═" * 70)
    for fn in (test_tax_id, test_cjk, test_citation_positive,
               test_citation_negative, test_confidence_gate, test_hpes,
               test_counterfactual, test_crosscheck, test_router,
               test_table_label_index):
        fn()
    print("\n" + "═" * 70)
    print(f"  通過 {PASS}　失敗 {FAIL}")
    print("═" * 70)
    sys.exit(1 if FAIL else 0)
