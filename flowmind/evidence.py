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
W_RETRIEVAL     = 0.20   # 檢索強度（top RRF 相對於理論上限）
W_CORROBORATION = 0.20   # 幾份彼此獨立的文件支持這個結論
W_SPARSE_HEALTH = 0.15   # 稠密與稀疏是否都有貢獻（單路命中通常較脆弱）

MAX_RRF = 2.0 / (60 + 1)   # 兩路都排第一時的 RRF 理論上限


def compute_confidence(claims: list[Claim], chunks: list[Chunk]) -> tuple[float, dict]:
    diag = retrieval_diagnostics(chunks)

    ci = citation_integrity(claims)
    retrieval_strength = min(1.0, diag["top_rrf"] / MAX_RRF) if diag["n"] else 0.0

    grounded_sources = {c.matched_source for c in claims if c.is_grounded and c.matched_source}
    corroboration = min(1.0, len(grounded_sources) / 3.0)   # 3 份獨立來源即視為充分

    sparse_health = (diag["sparse_contributing"] / diag["n"]) if diag["n"] else 0.0

    score = (W_CITATION * ci
             + W_RETRIEVAL * retrieval_strength
             + W_CORROBORATION * corroboration
             + W_SPARSE_HEALTH * sparse_health)

    # 硬性上限：只要有任何一個主張被判定為幻覺，整體信心不得超過 0.5。
    # 這是刻意設計的不對稱 —— 九句對、一句瞎編的報告，
    # 在授信場域的可用性不是 90%，而是接近 0。
    if any(c.verdict == Verdict.UNVERIFIABLE for c in claims):
        score = min(score, 0.50)
    if any(c.verdict == Verdict.WRONG_SOURCE for c in claims):
        score = min(score, 0.65)

    return round(score, 3), {
        "citation_integrity": round(ci, 3),
        "retrieval_strength": round(retrieval_strength, 3),
        "corroboration": round(corroboration, 3),
        "sparse_health": round(sparse_health, 3),
        "weights": {"citation": W_CITATION, "retrieval": W_RETRIEVAL,
                    "corroboration": W_CORROBORATION, "sparse": W_SPARSE_HEALTH},
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
        if bd.get("citation_integrity", 0) < 0.6:
            missing.append("模型提出的主張無法在現有文件中逐字驗證")
        if bd.get("retrieval_strength", 0) < 0.4:
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
