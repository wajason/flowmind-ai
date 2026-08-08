#!/usr/bin/env python3
"""
query_plan.py — 查詢理解層：問題分解 · 實體連結 · 中繼資料過濾
=============================================================================
【這三件事其實是同一件事的三個階段】

    ① 問題分解 Query Decomposition
       把「範圍條件」從「真正在問的東西」裡拆出來。
       「**日本**的保證成數是多少」→ 範圍=日本，核心=保證成數是多少
       不拆開的話，「日本」這兩個字會被丟進向量檢索裡當語意訊號，
       而它其實是個**過濾條件**，不是語意內容。

    ② 實體連結 Entity Linking
       把拆出來的字串對應到知識圖譜裡**具體的節點 ID**。
       「臺灣」「台灣」「中華民國」「本國」都要連到同一個
       jurisdiction 節點，否則後面每一層都要各自處理同義詞。

    ③ 中繼資料過濾 Metadata Filtering
       把連結好的實體轉成檢索時的硬過濾條件。

【最重要的設計決定：幾乎所有東西都不該當硬過濾】

這是本模組最反直覺、但也最需要講清楚的一點。

一個下錯的硬過濾會**靜默刪掉正確答案** —— 使用者只會看到「查無資料」，
不會知道答案其實在知識庫裡、只是被自己的過濾器擋掉了。
這種錯誤比「多召回幾個不相關的段落」嚴重得多，因為後者會被
後面的引用驗證與信心計分擋下來，而前者無聲無息。

所以本模組採取**非對稱的證據標準**：

    要下硬過濾 → 需要「這個訊號幾乎不可能出錯」等級的證據
    要出警示   → 只要有合理疑慮就可以

實際檢視每一種訊號後，只有兩種夠格當硬過濾：

    ✅ 版本過濾（已被取代的文件）
       這是制度事實：被取代的辦法就不是現行規定。
       已由 retrieval.hybrid_search(include_superseded=False) 實作。

    ✅ 問題**明確點名**某份文件
       「依 2025 年作業手冊，代位清償流程為何」—— 使用者自己指定了範圍，
       尊重它不會刪掉他要的答案，因為那正是他要的。

其餘一律**不下推**，只產生警示：

    ❌ 年份 → 2015 年公布的法規在 2026 年仍然有效。
             用年份過濾會把現行規定濾掉，這是明顯的錯誤。
    ❌ 法域 → 比較性段落經常同時提到兩國，過濾會誤傷。
             改由 graph.scope_check() 在**回答層**判斷可不可答，
             以及 evidence.chunk_jurisdiction_conflict() 標示個別段落。
    ❌ 主題 → 主題邊界模糊，「應收帳款」與「供應鏈金融」高度重疊，
             過濾掉一邊經常會濾掉正解。

**能做卻選擇不做，而且說得出為什麼不做**，
比把所有能加的過濾都加上去更接近工程判斷。

本模組**零 LLM 呼叫**。所有分解與連結都是決定性的字串與圖譜查詢，
因此它的行為可以被逐條複查，也不會在不同執行間漂移。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import psycopg2.extras

from . import db
from .graph import JURISDICTIONS

# ══════════════════════════════════════════════════════════════════════════
# 範圍標記
# ══════════════════════════════════════════════════════════════════════════

# 年份：民國年與西元年都要吃。民國 114 年 = 西元 2025 年。
_ROC_YEAR = re.compile(r"(?:民國)?\s*(\d{2,3})\s*年(?:度|版)?")
_AD_YEAR = re.compile(r"(?:西元)?\s*(19\d{2}|20\d{2})\s*年(?:度|版)?")

# 條號：「第 3 條」「第三條」「第 3 條第 2 項」
_ARTICLE = re.compile(r"第\s*([0-9]{1,3}|[一二三四五六七八九十百]+)\s*條"
                      r"(?:\s*第\s*([0-9]{1,2}|[一二三四五六七八九十]+)\s*項)?")

# 這些詞出現時，代表使用者在指定「哪一版」而不是在問內容
_VERSION_HINT = re.compile(r"(最新|現行|修正後|修正前|舊版|新版|前一版|上一版)")


@dataclass
class LinkedEntity:
    """問題中的一段文字 → 知識圖譜的一個節點。"""
    surface: str            # 問題中實際出現的字串
    node_type: str          # jurisdiction / topic / period / document
    label: str              # 圖譜中的正規標籤
    node_id: Optional[str] = None
    via: str = "exact"      # exact（字面相同）/ alias（同義詞）

    def __str__(self) -> str:
        v = "" if self.via == "exact" else f"（別名 → {self.label}）"
        return f"{self.surface}{v}[{self.node_type}]"


@dataclass
class QueryPlan:
    original: str
    core_question: str                       # 移除範圍標記後剩下的
    entities: list[LinkedEntity] = field(default_factory=list)
    years: list[int] = field(default_factory=list)          # 一律轉西元
    articles: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)             # 可安全下推的
    advisories: list[str] = field(default_factory=list)     # 不下推但要提醒的

    def entities_of(self, node_type: str) -> list[LinkedEntity]:
        return [e for e in self.entities if e.node_type == node_type]

    def render(self) -> str:
        L = ["查詢計畫", "─" * 66,
             f"  原問題　{self.original}",
             f"  核心問題 {self.core_question}"]
        if self.entities:
            L.append("  實體連結 " + "、".join(str(e) for e in self.entities))
        if self.years:
            L.append(f"  年份　　 {self.years}（已轉西元）")
        if self.articles:
            L.append(f"  條號　　 {self.articles}")
        L.append(f"  硬過濾　 {self.filters or '無（刻意保守，見模組說明）'}")
        for a in self.advisories:
            L.append(f"  ⚠️ {a}")
        return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════════
# ② 實體連結
# ══════════════════════════════════════════════════════════════════════════

def _load_nodes(tenant_id: str) -> list[dict]:
    """
    載入圖譜節點。目前規模約 112 個（83 文件 / 11 期間 / 9 主題 / 9 法域），
    全載入比對是可行的；若日後成長到數萬個，這裡要改成倒排索引。
    **把這個前提寫下來**，是因為「現在夠快」不等於「永遠夠快」，
    而一個沒寫下前提的效能決定，會在資料長大時變成沒人知道原因的慢。
    """
    with db.tenant_session(tenant_id) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT node_id, node_type, label FROM kg_nodes "
                "WHERE tenant_id IN (%s, 'SHARED')", (tenant_id,))
            return [dict(r) for r in cur.fetchall()]


def link_entities(question: str, nodes: list[dict]) -> list[LinkedEntity]:
    """
    決定性實體連結。長標籤優先比對 ——
    否則「中小企業信用保證基金」會先被「中小企業」吃掉，
    連到錯誤的（較籠統的）節點。
    """
    out: list[LinkedEntity] = []
    seen: set[tuple[str, str]] = set()

    # 先做法域的同義詞歸一，因為「臺灣/台灣/中華民國/本國」是同一件事，
    # 而圖譜裡只有一個正規標籤
    for canon, aliases in JURISDICTIONS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in question:
                key = ("jurisdiction", canon)
                if key not in seen:
                    seen.add(key)
                    out.append(LinkedEntity(
                        surface=alias, node_type="jurisdiction", label=canon,
                        via="exact" if alias == canon else "alias"))
                break

    for n in sorted(nodes, key=lambda x: len(x["label"] or ""), reverse=True):
        label = (n["label"] or "").strip()
        if len(label) < 2 or label not in question:
            continue
        key = (n["node_type"], label)
        if key in seen:
            continue
        seen.add(key)
        out.append(LinkedEntity(surface=label, node_type=n["node_type"],
                                label=label, node_id=n["node_id"]))
    return out


# ══════════════════════════════════════════════════════════════════════════
# ① 問題分解
# ══════════════════════════════════════════════════════════════════════════

def _roc_to_ad(n: int) -> Optional[int]:
    """民國年轉西元。只接受合理範圍，避免把「第 3 條」之類的數字誤判成年份。"""
    return n + 1911 if 60 <= n <= 200 else None


def extract_years(question: str) -> list[int]:
    years: set[int] = set()
    for m in _AD_YEAR.finditer(question):
        years.add(int(m.group(1)))
    for m in _ROC_YEAR.finditer(question):
        n = int(m.group(1))
        if 1900 <= n <= 2100:          # 已經是西元，_AD_YEAR 會抓到，這裡略過
            continue
        ad = _roc_to_ad(n)
        if ad:
            years.add(ad)
    return sorted(years)


def decompose(question: str) -> tuple[str, list[int], list[str]]:
    """
    把範圍標記從問題中拆出來，回傳 (核心問題, 年份, 條號)。

    核心問題是**移除範圍標記後**的字串，用途是餵給向量檢索 ——
    因為「日本」「2025 年」這類詞是過濾條件，不是語意內容，
    留在裡面會把檢索往錯的方向拉。
    """
    years = extract_years(question)
    articles = [m.group(0) for m in _ARTICLE.finditer(question)]

    core = question
    for pat in (_AD_YEAR, _ROC_YEAR, _ARTICLE, _VERSION_HINT):
        core = pat.sub(" ", core)
    for aliases in JURISDICTIONS.values():
        for alias in sorted(aliases, key=len, reverse=True):
            core = core.replace(alias, " ")
    core = re.sub(r"\s+", " ", core).strip()

    # 全被拆光的話（例如問題只有「2025 年」三個字），退回原問題。
    # 空字串送進向量檢索會得到毫無意義的結果，而那種失敗很難查。
    return (core or question), years, articles


# ══════════════════════════════════════════════════════════════════════════
# ③ 中繼資料過濾（刻意保守）
# ══════════════════════════════════════════════════════════════════════════

def build_plan(question: str, tenant_id: str = "SHARED",
               nodes: Optional[list[dict]] = None) -> QueryPlan:
    core, years, articles = decompose(question)
    nodes = nodes if nodes is not None else _load_nodes(tenant_id)
    entities = link_entities(question, nodes)

    plan = QueryPlan(original=question, core_question=core,
                     entities=entities, years=years, articles=articles)

    # ── 唯一下推的硬過濾：問題明確點名了某份文件 ──────────────────────
    # 這安全的理由是它不會刪掉使用者要的東西 —— 那正是他指定的東西。
    docs = [e.label for e in entities if e.node_type == "document"]
    if docs:
        plan.filters["sources"] = docs
        plan.advisories.append(
            f"問題明確指名 {len(docs)} 份文件，檢索限縮於這些文件："
            + "、".join(docs[:3]))

    # ── 以下一律不下推，只出警示 ──────────────────────────────────────
    foreign = [e.label for e in entities
               if e.node_type == "jurisdiction" and e.label != "台灣"]
    if foreign:
        plan.advisories.append(
            f"問題指涉境外法域（{'、'.join(foreign)}）。"
            "**不做硬過濾** —— 比較性段落常同時提到兩國，過濾會誤傷；"
            "改由 graph.scope_check() 判斷可否回答、"
            "evidence.chunk_jurisdiction_conflict() 標示個別段落。")

    if years:
        plan.advisories.append(
            f"問題提到年份 {years}。**不做硬過濾** —— "
            "早年公布的法規在今天仍可能有效，用年份濾會把現行規定濾掉；"
            "版本正確性改由 documents.doc_status 的取代關係處理。")

    if articles:
        plan.advisories.append(
            f"問題指定條號 {articles}。**不做硬過濾** —— "
            "條號在不同法規中重複出現，光靠條號無法定位；"
            "改由引用驗證確認答案確實出自該條。")

    return plan


__all__ = ["LinkedEntity", "QueryPlan", "build_plan", "decompose",
           "link_entities", "extract_years"]
