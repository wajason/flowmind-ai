"""
flowmind.evidence — Evidence / Confidence / Source / Reason 輸出契約
=============================================================================
【這支檔案是整個產品的核心主張】

市面上絕大多數 RAG 產品的「可解釋性」，是請 LLM 自己在句尾寫上 [來源: xxx.pdf]。
問題在於：那一串引用標籤本身就是模型生成的 token，
模型可以在完全沒讀那份文件的情況下，寫出格式完美的引用。
換句話說，「看起來有引用」和「真的有根據」是兩件事，
而使用者、甚至評審，從輸出畫面上分辨不出來。

FlowMind 的做法是把引用變成**可程式驗證的斷言**：
  1. 強制模型輸出的每一個主張，都必須附上一段「逐字摘錄」(quote)。
  2. 程式回頭到實際檢索到的文本裡做字串比對，確認那段話真的存在。
  3. 對不上的主張，不是扣分而已 —— 是直接從答案裡移除，並標記為未驗證。

這條路徑上沒有任何一步經過語言模型，所以它不可能被「講得更有說服力」騙過。
模型唯一能提高分數的方法，就是真的去讀檢索到的文件。這就是不可 gameable 的意思。

信心分數同理：它由檢索強度、佐證來源數、引用驗證通過率這些可量測的量算出來，
模型無法透過語氣影響它。低於門檻時系統直接拒答並說明缺什麼，
而不是給一個聽起來很篤定的答案 —— 在授信場域，這是唯一負責任的行為。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from rapidfuzz import fuzz

from . import config
from .retrieval import Chunk, retrieval_diagnostics


# ══════════════════════════════════════════════════════════════════════════
# 1. 文字正規化：比對前先拉齊全形/半形、空白、標點
# ══════════════════════════════════════════════════════════════════════════

_WS = re.compile(r"[\s　]+")
_PUNCT = re.compile(r"[，。、；：？！「」『』（）()\[\]【】\-—–_·．.,:;!?\"'`*#>]")


def normalize_for_match(text: str) -> str:
    """
    引用比對用的正規化。刻意不做同義詞替換或斷詞，
    因為那會讓「近似」被當成「相符」，等於把驗證變回信任。
    只處理不影響語義的表面差異：全形半形、空白、標點、大小寫。
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _WS.sub("", text)
    text = _PUNCT.sub("", text)
    return text.lower()


# ══════════════════════════════════════════════════════════════════════════
# 2. 主張與驗證結果
# ══════════════════════════════════════════════════════════════════════════

class Verdict(str, Enum):
    EXACT = "exact"            # 逐字命中所引來源
    NEAR = "near"              # 高度近似（≥95）—— 通常是 OCR 或標點差異
    WRONG_SOURCE = "wrong_source"   # 內容存在，但不在它宣稱的那份文件裡
    UNVERIFIABLE = "unverifiable"   # 檢索文本裡根本找不到 → 判定為幻覺


@dataclass
class Claim:
    statement: str
    quote: str
    source: str
    chunk_index: Optional[int] = None
    verdict: Verdict = Verdict.UNVERIFIABLE
    match_score: float = 0.0
    matched_source: Optional[str] = None

    @property
    def is_grounded(self) -> bool:
        return self.verdict in (Verdict.EXACT, Verdict.NEAR)


@dataclass
class EvidencePack:
    """一次問答的完整輸出契約。這個結構會原樣寫進稽核紀錄。"""
    question: str
    tenant_id: str
    answer: str = ""
    claims: list[Claim] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    confidence_breakdown: dict = field(default_factory=dict)
    abstained: bool = False
    abstain_reason: str = ""
    needs_human_review: bool = False
    human_review_reason: str = ""
    sources: list[str] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    model: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        d["claims"] = [{**asdict(c), "verdict": c.verdict.value} for c in self.claims]
        return json.dumps(d, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# 3. 引用驗證：純字串比對，不經過任何模型
# ══════════════════════════════════════════════════════════════════════════

NEAR_MATCH_THRESHOLD = 95.0   # rapidfuzz partial_ratio
MIN_QUOTE_CHARS = 8           # 太短的引用（「是」「應收帳款」）驗證起來沒有意義
SOURCE_NAME_THRESHOLD = 88.0  # 檔名模糊比對門檻


def _canon_source(name: str) -> str:
    """檔名比對用的正規化：去掉所有空白、統一大小寫、拿掉常見的裝飾字元。"""
    s = unicodedata.normalize("NFKC", str(name or "")).lower()
    return re.sub(r"[\s`'\"《》〈〉\[\]()]", "", s)


# 刪節號的各種寫法。中英文、全形半形都要涵蓋。
_ELLIPSIS = re.compile(r"\.{2,}|…+|。{3,}|．{2,}|\s*\(略\)\s*|\s*〔略〕\s*")
MIN_FRAGMENT_CHARS = 4


def _find_ordered(fragments: list[str], haystack: str) -> bool:
    """
    要求每一個片段都逐字出現，而且**順序與原文一致**。

    為什麼要管順序：不管順序的話，模型可以從文件各處挑幾個零碎詞湊成一句
    原文根本沒說過的話，然後每個片段都「驗證通過」。
    加上順序約束後，通過驗證就等同於「這句話確實是原文某一段的縮寫」。
    這仍然是純字串比對，沒有引入任何模糊判斷。
    """
    pos = 0
    for frag in fragments:
        idx = haystack.find(frag, pos)
        if idx < 0:
            return False
        pos = idx + len(frag)
    return True


def _split_quote(normalized_quote: str, raw_quote: str) -> list[str]:
    """
    依刪節號把引用切成片段並正規化。
    太短的片段（少於 4 字）丟掉 —— 「之」「本」這種片段在任何文件裡都找得到，
    留著只會讓驗證變寬鬆。
    """
    parts = [normalize_for_match(p) for p in _ELLIPSIS.split(raw_quote)]
    frags = [p for p in parts if len(p) >= MIN_FRAGMENT_CHARS]
    return frags if len(frags) > 1 else [normalized_quote]


def _resolve_source(claimed: str, canon: dict[str, str]) -> Optional[str]:
    """
    把模型寫的來源名對應回真實檔名。

    兩段式：先用去空白／小寫化的鍵精準對；對不上再做模糊比對。
    刻意設 88 分的門檻而不是更低 —— 這裡要修的是「打字誤差」，
    不是讓模型隨便寫個檔名都能被湊到某份文件上。
    對不上就回 None，讓後續流程照常判定為未驗證。
    """
    if not claimed:
        return None
    key = _canon_source(claimed)
    if key in canon:
        return canon[key]
    best, best_score = None, 0.0
    for k, real in canon.items():
        sc = fuzz.ratio(key, k)
        if sc > best_score:
            best, best_score = real, sc
    return best if best_score >= SOURCE_NAME_THRESHOLD else None


def verify_claims(claims: list[Claim], chunks: list[Chunk]) -> list[Claim]:
    """
    對每一個主張，回到實際送進 LLM 的文本裡驗證它的逐字摘錄。

    先在「它自己宣稱的來源」裡找；找不到再擴大到全部檢索文本。
    這個區分很重要：內容存在但引錯來源（WRONG_SOURCE）在授信報告裡
    仍然是嚴重問題 —— 授信人員會照著引用去翻那份文件，然後翻不到。
    """
    by_source: dict[str, str] = {}
    for c in chunks:
        by_source.setdefault(c.source, "")
        by_source[c.source] += normalize_for_match(c.parent_content) + "\n"
    corpus_all = "\n".join(by_source.values())

    # 檔名對照表。語言模型抄檔名時很容易多打一個空格
    # （實測遇過「信保基金 - 供應商融資信用保證要點.md」對上
    #  「信保基金-供應商融資信用保證要點.md」），
    # 若因此把一句**引用完全正確**的話判成幻覺，那是驗證器的錯，不是模型的錯。
    # 這裡先用「去空白、小寫化」的鍵去對，仍對不上才做模糊比對找最接近的真實檔名。
    canon = {_canon_source(s): s for s in by_source}

    for claim in claims:
        q = normalize_for_match(claim.quote)
        if len(q) < MIN_QUOTE_CHARS:
            claim.verdict = Verdict.UNVERIFIABLE
            claim.match_score = 0.0
            continue

        resolved = _resolve_source(claim.source, canon)
        if resolved and resolved != claim.source:
            claim.source = resolved          # 修正成真實檔名，讓輸出的引用可以直接點開
        # 引用可能含刪節號（「訂單、發票…之文件撥貸」），這是正當的引用慣例。
        # 拆成片段後要求每一段都逐字命中且順序正確，等同於「原文某段的縮寫」。
        frags = _split_quote(q, claim.quote)

        own = by_source.get(claim.source, "")
        if own and _find_ordered(frags, own):
            claim.verdict, claim.match_score = Verdict.EXACT, 100.0
            claim.matched_source = claim.source
            continue

        score = fuzz.partial_ratio(q, own) if own else 0.0
        if score >= NEAR_MATCH_THRESHOLD:
            claim.verdict, claim.match_score = Verdict.NEAR, score
            claim.matched_source = claim.source
            continue

        # 內容是否存在於其他來源？
        hit_src = next((s for s, t in by_source.items() if _find_ordered(frags, t)), None)
        if hit_src:
            claim.verdict, claim.match_score = Verdict.WRONG_SOURCE, 100.0
            claim.matched_source = hit_src
            continue

        best_src, best = None, 0.0
        for s, t in by_source.items():
            sc = fuzz.partial_ratio(q, t)
            if sc > best:
                best, best_src = sc, s
        if best >= NEAR_MATCH_THRESHOLD:
            claim.verdict, claim.match_score = Verdict.WRONG_SOURCE, best
            claim.matched_source = best_src
        else:
            claim.verdict, claim.match_score = Verdict.UNVERIFIABLE, best

    return claims


def citation_integrity(claims: list[Claim]) -> float:
    """通過驗證的主張佔比。沒有任何主張時回 0，不回 1 —— 空答案不算滿分。"""
    if not claims:
        return 0.0
    return sum(1 for c in claims if c.is_grounded) / len(claims)


# ══════════════════════════════════════════════════════════════════════════
# 4. 信心分數：來自可量測訊號，模型無從影響
# ══════════════════════════════════════════════════════════════════════════

# 權重公開在程式碼裡，不是黑盒。任何人都可以檢查、質疑、重算。
W_CITATION      = 0.45   # 引用驗證通過率 —— 權重最高，因為它最難造假
W_RETRIEVAL     = 0.20   # 檢索強度（top-1 的絕對語意相似度）
W_CORROBORATION = 0.20   # 幾份彼此獨立的文件支持這個結論
W_SPARSE_HEALTH = 0.15   # 稠密與稀疏是否都有貢獻（單路命中通常較脆弱）

# ── 檢索強度為什麼用 dense 分數而不是 RRF ─────────────────────────────────
# 原本用 top_rrf / MAX_RRF 當檢索強度，這是**概念上的誤用**：
# RRF 是為了「融合兩份排名」設計的，它只看名次不看相似度 ——
# 排第一名永遠得 1/(60+1)，不管那筆結果到底有多相關。
#
# 實測（各 4 題，見 docs/SDD.md）：
#     有答案的問題   top_dense 0.691~0.799   rrf/MAX 0.969~1.000
#     知識庫沒有的   top_dense 0.581~0.653   rrf/MAX 0.739~1.000
#
# RRF 完全分不出來（無解題也能拿 1.000），dense 相似度則有乾淨的分界。
# 這直接造成一個嚴重後果：問「信保基金的內部授信評分卡權重為何」
# （公開文件必然沒有）拿到信心 0.852 —— 因為模型從相關但不對題的段落
# 引用了真實存在的句子，引用驗證通過，而檢索強度又被 RRF 灌成滿分。
#
# 下限/上限是從實測資料設的，**樣本只有 8 題**，換 embedding 模型或
# 大幅擴充知識庫後必須重新校準（scripts/calibrate_confidence.py）。
DENSE_FLOOR = 0.55   # 低於此值：知識庫幾乎確定沒有涵蓋這個問題
DENSE_CEIL  = 0.80   # 高於此值：語意高度吻合

# ══════════════════════════════════════════════════════════════════════════
# 範圍詞驗證：語意相似度在原理上抓不到的那一類失敗
# ══════════════════════════════════════════════════════════════════════════
# 【這一段是一次失敗的校準逼出來的，過程值得記錄】
#
# 原本的想法是用 dense 相似度當「知識庫有沒有涵蓋這個問題」的訊號，
# 門檻取自 8 題探測樣本（0.66）。後來用**獨立校準集**（20 題，
# 與評測集零重疊）重測，結果是：
#
#     可答問題   dense 最低 0.6280
#     無解問題   dense 最高 0.7409  ←「日本的中小企業信用保證協會保證成數是多少？」
#
# 兩組完全重疊，門檻不成立。原本的 0.66 是對那 8 題過擬合出來的。
#
# 根因很清楚：「日本的保證成數」與「台灣的保證成數」在語意空間裡幾乎重合，
# **embedding 分不出「日本的」和「台灣的」**。這不是模型不夠好，
# 是向量相似度這個工具在原理上就不處理「指涉對象是誰」。
#
# 同一個根因也解釋了評測裡另外三題失敗：
#   · 「2027 年的保證成數」—— 主題對，年份不存在
#   · 「作業手冊第 87 條」—— 主題對，條次不存在
#   · 「2026 年 12 月的統計」—— 主題對，期間超出資料範圍
#
# 正確的訊號是**確定性的**：問題裡指定了一個範圍（國家／年份／條次），
# 而檢索到的文本裡根本沒有這個範圍詞 —— 那就代表答案不在這批文本裡，
# 不管相似度多高、不管模型引用得多漂亮。
#
# 這是字串比對，不是語意判斷，所以它不會有「差不多就算了」的問題。

# 明確的外國／境外範圍詞。知識庫是台灣的制度文件，
# 一旦問題指定了境外範圍而文本沒有提到，答案必然不在裡面。
_FOREIGN_SCOPE = [
    "日本", "韓國", "新加坡", "香港", "澳門", "中國大陸", "大陸地區", "美國",
    "歐盟", "英國", "德國", "法國", "越南", "泰國", "馬來西亞", "印尼",
    "菲律賓", "印度", "澳洲", "加拿大", "GDPR", "Basel", "巴塞爾",
]
_YEAR = re.compile(r"(?:民國\s*)?(\d{2,4})\s*年")
_ARTICLE = re.compile(r"第\s*([一二三四五六七八九十百\d]+)\s*條")


def extract_scope_terms(question: str) -> list[str]:
    """
    從問題中抽出「限定範圍」的詞：境外地名、年份、條次。

    只抽這三類是刻意的 —— 它們有兩個共同性質：
      (a) 可以用字串比對確認「文本裡有沒有」，不需要語意判斷
      (b) 一旦問題指定了它而文本沒有，答案就確定不在這批文本裡
    公司名、人名之類的也是範圍詞，但誤判成本高（客戶自己的名字本來就
    不會出現在法規裡），暫不納入。
    """
    terms: list[str] = []
    q = unicodedata.normalize("NFKC", question or "")
    for kw in _FOREIGN_SCOPE:
        if kw in q:
            terms.append(kw)
    for m in _YEAR.finditer(q):
        y = m.group(1)
        # 兩位數視為民國年（115年），四位數視為西元年
        terms.append(f"{y}年")
    for m in _ARTICLE.finditer(q):
        terms.append(f"第{m.group(1)}條")
    return list(dict.fromkeys(terms))


def missing_scope_terms(question: str, chunks: list[Chunk]) -> list[str]:
    """回傳「問題有指定、但檢索文本完全沒提到」的範圍詞。"""
    terms = extract_scope_terms(question)
    if not terms:
        return []
    corpus = normalize_for_match(" ".join(c.parent_content for c in chunks))
    return [t for t in terms if normalize_for_match(t) not in corpus]


# ── 覆蓋率硬閘門：加權平均擋不住的一類失敗 ────────────────────────────────
# 信心是加權和，而 citation_integrity 佔 0.45。問題是**模型永遠找得到
# 某段真實文字可以引用** —— 問「信保基金的內部評分卡權重為何」，
# 它會從「保證成數」「徵信」相關段落抄一句真的存在的話，
# 引用驗證通過、citation_integrity 拿滿分，信心就被拉到 0.85。
#
# 引用是真的，但它沒有回答那個問題。**可驗證 ≠ 切題。**
#
# 加權和沒辦法處理這件事：把 citation 權重調低會傷害正常情況。
# 需要的是一個閘門 —— 如果整個知識庫裡語意最接近的一段都不夠接近，
# 那就是「沒有涵蓋這個問題」，此時模型引用得多漂亮都不該有高信心。
#
# 門檻取自實測（8 題）：
#     有答案   top_dense 最低 0.691
#     知識庫沒有 top_dense 最高 0.653
# 取 0.66 作為分界。**樣本只有 8 題，這個門檻必須隨知識庫擴充重新校準**，
# 換 embedding 模型後更是一定要重測。
#
# dense 門檻由 scripts/calibrate_confidence.py 從**獨立校準集**推導，
# 目標函數是「在零誤攔可答題的約束下，最大化無解題攔截率」。
# 這個目標直接編碼本場域的成本不對稱：誤攔一題只是要求補件，
# 放過一題無解的則會給出有信心的錯誤答案。
#
# 2026-08-08 校準結果（20 題 dev set，與 50 題評測集零重疊）：
#     可答 12 題  dense 最低 0.6280
#     無解  8 題  dense 最高 0.7409（「日本的保證成數」—— 語意與台灣的幾乎重合）
#     門檻 = 0.6280 − 安全邊際 0.03 = 0.598
#     攔截 5/8 無解題，誤攔 0/12 可答題
#
# 安全邊際不是隨手加的：20 題樣本的最小值本身有抽樣誤差，
# 門檻貼齊最小值等於對這 20 題過擬合。
# 換 embedding 模型或大幅擴充知識庫後必須重新校準。
DENSE_COVERAGE_GATE = 0.598
COVERAGE_GATE_CAP   = 0.35   # 低於門檻時的信心上限（低於拒答門檻 0.45）

# 各種上限一律定義成「比拒答門檻低一點」，而不是各自寫死一個數字。
# 這樣 .env 調整 CONFIDENCE_ABSTAIN_THRESHOLD 時，
# 「含幻覺必定拒答」這個不變量不會悄悄失效。
GATE_MARGIN = 0.05

# 「曾經掰過但已被移除」的扣分係數。
# 輸出是乾淨的所以不判死，但這次生成會掰本身是可靠度訊號。
# 0.8 代表信心打八折 —— 一個有根據但生成過程不穩的答案，
# 比一個全程穩定的答案該低一級，但不該被完全丟掉。
HALLUCINATION_PENALTY = 0.8


def hallucination_cap() -> float:
    return max(0.0, config.CONFIDENCE_ABSTAIN_THRESHOLD - GATE_MARGIN)


def compute_confidence(claims: list[Claim], chunks: list[Chunk],
                       question: str = "",
                       had_hallucination: bool = False) -> tuple[float, dict]:
    """
    計算信心分數。

    `claims` 應該是**移除幻覺之後**、實際會輸出的那批主張 ——
    計分要對得起實際輸出的內容，不是對得起中間狀態。

    `had_hallucination` 則記錄「模型原本有沒有掰過」。
    這件事仍然要罰（模型會掰代表這次生成不夠可靠），
    但罰的方式是扣分而不是直接判死 —— 因為那句話已經被移除了，
    輸出給使用者的內容是乾淨的。
    """
    diag = retrieval_diagnostics(chunks)

    ci = citation_integrity(claims)
    if diag["n"]:
        raw = (diag["top_dense"] - DENSE_FLOOR) / (DENSE_CEIL - DENSE_FLOOR)
        retrieval_strength = max(0.0, min(1.0, raw))
    else:
        retrieval_strength = 0.0

    grounded_sources = {c.matched_source for c in claims if c.is_grounded and c.matched_source}
    corroboration = min(1.0, len(grounded_sources) / 3.0)   # 3 份獨立來源即視為充分

    sparse_health = (diag["sparse_contributing"] / diag["n"]) if diag["n"] else 0.0

    score = (W_CITATION * ci
             + W_RETRIEVAL * retrieval_strength
             + W_CORROBORATION * corroboration
             + W_SPARSE_HEALTH * sparse_health)

    # ── 硬性上限：只要有任何一個主張被判定為幻覺 ──────────────────────
    # 這是刻意設計的不對稱：九句對、一句瞎編的報告，
    # 在授信場域的可用性不是 90%，而是接近 0。
    #
    # 上限**綁定在拒答門檻之下**，不是一個獨立的魔術數字。
    # 這一點是被 50 題評測抓出來的：原本上限寫死 0.50、拒答門檻 0.45，
    # 兩個數字各自訂、從沒放在一起檢查，結果是
    # 「內含幻覺的答案」信心剛好 0.500，卡在門檻之上被放行 ——
    # 三題無解題（不存在的條次、已失效的版本、超出期間的統計）
    # 就是這樣溜過去的。
    #
    # 綁定之後這個不變量由建構保證：**含幻覺的答案一定觸發拒答**，
    # 而且之後有人調整拒答門檻時，關係仍然成立。
    # 仍在清單裡的幻覺（呼叫端沒先移除）→ 直接判死，因為它會被輸出。
    if any(c.verdict == Verdict.UNVERIFIABLE for c in claims):
        score = min(score, hallucination_cap())
    elif had_hallucination:
        # 曾經掰過但已被移除：輸出是乾淨的，所以不判死；
        # 但「這次生成會掰」本身是可靠度訊號，扣一段分數。
        # 扣分幅度刻意固定且公開，不是可調的旋鈕。
        score *= HALLUCINATION_PENALTY
    if any(c.verdict == Verdict.WRONG_SOURCE for c in claims):
        score = min(score, 0.65)

    # 覆蓋率硬閘門：知識庫裡語意最接近的一段都不夠接近 → 這題沒被涵蓋。
    # 此時不管模型引用得多漂亮，都不該有高信心（見上方常數的說明）。
    # 範圍詞驗證優先於相似度：問題指定了某個範圍（境外地名／年份／條次），
    # 而檢索文本完全沒提到 → 答案確定不在這批文本裡。這是字串比對，
    # 不受「語意上很像」影響 —— 而語意上很像正是這類問題最危險的地方。
    missing = missing_scope_terms(question, chunks) if question else []
    dense_gated = bool(diag["n"]) and diag["top_dense"] < DENSE_COVERAGE_GATE
    coverage_gated = bool(missing) or dense_gated
    if coverage_gated:
        score = min(score, COVERAGE_GATE_CAP)

    return round(score, 3), {
        "citation_integrity": round(ci, 3),
        "retrieval_strength": round(retrieval_strength, 3),
        "top_dense": round(diag.get("top_dense", 0.0), 4),
        "corroboration": round(corroboration, 3),
        "sparse_health": round(sparse_health, 3),
        "weights": {"citation": W_CITATION, "retrieval": W_RETRIEVAL,
                    "corroboration": W_CORROBORATION, "sparse": W_SPARSE_HEALTH},
        "coverage_gated": coverage_gated,
        "missing_scope_terms": missing,
        "dense_gated": dense_gated,
        "coverage_gate_threshold": DENSE_COVERAGE_GATE,
        "hallucinated_claims": sum(1 for c in claims if c.verdict == Verdict.UNVERIFIABLE),
        "misattributed_claims": sum(1 for c in claims if c.verdict == Verdict.WRONG_SOURCE),
        "grounded_sources": sorted(s for s in grounded_sources if s),
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. 拒答閘門與人工複核
# ══════════════════════════════════════════════════════════════════════════

_AMOUNT = re.compile(r"(?:NT\$|新台幣|台幣)?\s*([\d][\d,]{4,})\s*(?:元|萬元|萬|億)?")


def largest_amount_mentioned(text: str) -> float:
    """
    粗略抓出答案中提到的最大金額，用來判斷是否要求人工複核。
    刻意寧可高估：抓錯而多送一次人工複核的成本，遠低於漏掉一筆大額建議。
    """
    best = 0.0
    for m in _AMOUNT.finditer(text or ""):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        tail = text[m.end() - 3:m.end() + 2]
        if "億" in tail:
            val *= 1e8
        elif "萬" in tail:
            val *= 1e4
        best = max(best, val)
    return best


def apply_gates(pack: EvidencePack) -> EvidencePack:
    """
    兩道閘門：
      (a) 信心低於門檻 → 拒答。不是輸出「我不太確定，但可能是…」，
          而是明確說出缺哪份文件、要補什麼件。這對使用者才有行動價值。
      (b) 金額超過門檻 → 標記為必須由授信人員覆核。
          這對齊 Anthropic 金融服務套件的原則：AI 產出協助盡職調查，
          但不做具法律效力的授信決策，最終一定要有人簽字。
    """
    if pack.confidence < config.CONFIDENCE_ABSTAIN_THRESHOLD:
        pack.abstained = True
        missing = []
        bd = pack.confidence_breakdown
        scope_missing = bd.get("missing_scope_terms") or []
        if scope_missing:
            missing.append(
                f"問題指定的範圍「{'、'.join(scope_missing)}」"
                f"完全未出現在檢索到的文件中 —— "
                f"檢索到的內容雖然主題相近，但講的不是這個對象")
        if bd.get("dense_gated"):
            missing.append(
                f"知識庫未涵蓋此主題（最相近文件的語意相似度 "
                f"{bd.get('top_dense', 0):.2f}，低於覆蓋門檻 "
                f"{bd.get('coverage_gate_threshold', 0):.2f}）")
        if bd.get("citation_integrity", 0) < 0.6:
            missing.append("模型提出的主張無法在現有文件中逐字驗證")
        if bd.get("retrieval_strength", 0) < 0.4 and not bd.get("coverage_gated"):
            missing.append("知識庫中沒有與此問題足夠相關的文件")
        if bd.get("corroboration", 0) < 0.34:
            missing.append("僅有單一來源支持，缺乏交叉佐證")
        pack.abstain_reason = (
            f"信心分數 {pack.confidence:.2f} 低於門檻 "
            f"{config.CONFIDENCE_ABSTAIN_THRESHOLD:.2f}，系統選擇不回答。原因："
            + "；".join(missing or ["綜合證據強度不足"])
        )
        pack.answer = ""
        pack.claims = [c for c in pack.claims if c.is_grounded]

    amount = largest_amount_mentioned(pack.answer)
    if amount >= config.HUMAN_REVIEW_AMOUNT_TWD:
        pack.needs_human_review = True
        pack.human_review_reason = (
            f"內容涉及金額約 NT${amount:,.0f}，達人工複核門檻 "
            f"NT${config.HUMAN_REVIEW_AMOUNT_TWD:,.0f}，需授信人員覆核後方可對外提出。"
        )
    return pack


def strip_ungrounded(pack: EvidencePack) -> EvidencePack:
    """
    把未通過驗證的主張從答案正文移除，並收進「已移除」清單。

    這是刻意的產品決策：不把可疑內容留在正文裡讓使用者自己判斷。
    真實授信人員一天看幾十份案子，任何需要「自己再核對一次」的輸出，
    實際上都會被直接跳過。要嘛給可信的，要嘛明說給不出來。
    """
    removed = [c for c in pack.claims if not c.is_grounded]
    for c in removed:
        # 用逐字比對把該句從答案中拿掉；拿不掉就整段標註，不硬改文字
        if c.statement and c.statement in pack.answer:
            pack.answer = pack.answer.replace(c.statement, "")
        # 兩種未通過的原因要分開講。以前一律寫「查無此段落」，
        # 但 WRONG_SOURCE 的情況是「內容找得到，只是出處標錯了」——
        # 混為一談會讓使用者以為系統在亂刪，反而不信任驗證結果。
        if c.verdict == Verdict.WRONG_SOURCE:
            reason = (f"內容確實存在於 {c.matched_source}，"
                      f"但宣稱出處為 {c.source}，引用歸屬錯誤")
        else:
            reason = f"宣稱出處 {c.source}，但檢索文本中查無此段落"
        pack.unknowns.append(f"（已移除未經驗證的敘述）{c.statement[:60]}… — {reason}")
    pack.answer = re.sub(r"\n{3,}", "\n\n", pack.answer).strip()
    return pack
