"""
flowmind.drift — 輸出漂移驗證（Output Drift Validation）
=============================================================================
【這個指標的來源與為什麼我們原本漏掉了它】

參考：Khatchadourian & Franco, "LLM Output Drift: Cross-Provider Validation &
Mitigation for Financial Workflows," arXiv:2511.07585 (2025)（IBM）

我們一直在測「答案對不對」（HPES、CVR、拒答紀律），
但**從來沒測過「同一個問題問兩次，答案一不一樣」**。

這在金融場域是獨立且更基本的要求：

> 金融機構需要可重現的輸出以滿足稽核要求，
> 無論不確定性是否可能提升模型的創造力或能力。
> 若 LLM 在相同 prompt 下改變輸出結構（JSON 鍵名格式、SQL 結構），
> 就違反了 Basel III、Dodd-Frank、MiFID II 的稽核與法遵標準。

也就是說：**一個平均準確率 90% 但每次答案都不一樣的系統，
在銀行是不能用的；一個準確率 85% 但完全可重現的系統可以。**
因為稽核要問的是「當初這個建議是根據什麼給的」，
如果重跑得到不同答案，這個問題就無法回答。

【論文最反直覺的發現：一致性悖論】

    Granite-3-8B (IBM)   輸出一致性 100%
    Qwen2.5-7B           輸出一致性 100%
    GPT-OSS-120B         輸出一致性  12.5%（95% CI: 3.5–36.0%）
    Fisher's exact test  p < 0.0001

**模型越大，輸出越不穩定。** 這直接推翻「越大越好」的預設，
也回頭支持我們刻意用小模型的決定（見 DECISIONS.md D-02）——
但當時我們是為了「暴露 harness 缺陷」，不知道還有這個理由。

【三層模型分級（論文提出，我們採用）】

  Tier 1 稽核級（零漂移不可妥協）：數值抽取、SQL 生成
  Tier 2 有界不確定（需 invariant 檢查）：實體抽取、摘要
  Tier 3 專家主導（強制 human-in-the-loop）：複雜策略與推理

我們的 `llm.extract_json()` 屬於 Tier 1，**必須零漂移**。
本模組就是驗證它是否真的做到。

【我們量三個層次的一致性，而不是只看字串相等】

  1. **位元級**：輸出字串完全相同（最嚴格，稽核要的就是這個）
  2. **結構級**：JSON 鍵集合與型別相同（鍵順序、空白可不同）
  3. **語意級**：抽出的欄位值相同（格式化差異可接受）

分三層是因為它們對應不同的補救成本：
位元級不同但結構級相同 → 加一層正規化就能修；
結構級就不同 → 下游解析會壞，是真正的問題。
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class DriftResult:
    label: str
    runs: int
    bitwise_identical: int          # 與第一次完全相同的次數
    structural_identical: int       # JSON 鍵集合相同的次數
    semantic_identical: int         # 欄位值相同的次數
    distinct_outputs: int           # 出現過幾種不同的輸出
    latencies: list[float] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def bitwise_rate(self) -> float:
        return self.bitwise_identical / self.runs if self.runs else 0.0

    @property
    def structural_rate(self) -> float:
        return self.structural_identical / self.runs if self.runs else 0.0

    @property
    def semantic_rate(self) -> float:
        return self.semantic_identical / self.runs if self.runs else 0.0

    @property
    def tier(self) -> int:
        """
        依論文的三層分級判定。
        用**位元級**一致性判 Tier 1，因為稽核要求的是可重現的完整輸出，
        不是「意思一樣就好」。
        """
        if self.bitwise_rate >= 0.99:
            return 1
        if self.structural_rate >= 0.95:
            return 2
        return 3

    @property
    def tier_label(self) -> str:
        return {1: "Tier 1 稽核級（零漂移）",
                2: "Tier 2 有界不確定（需 invariant 檢查）",
                3: "Tier 3 專家主導（強制人工覆核）"}[self.tier]


def _canon_struct(obj: Any) -> str:
    """結構指紋：只保留鍵的階層與型別，忽略值。"""
    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            # 列表只看第一個元素的結構；長度差異算在語意層而非結構層
            return [walk(o[0])] if o else []
        return type(o).__name__
    return json.dumps(walk(obj), ensure_ascii=False, sort_keys=True)


def _canon_semantic(obj: Any) -> str:
    """語意指紋：值正規化後排序序列化，忽略鍵順序與空白。"""
    def norm(o):
        if isinstance(o, dict):
            return {k: norm(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            return sorted((json.dumps(norm(x), ensure_ascii=False, sort_keys=True)
                           for x in o))
        if isinstance(o, str):
            return o.strip()
        if isinstance(o, float) and o == int(o):
            return int(o)
        return o
    return json.dumps(norm(obj), ensure_ascii=False, sort_keys=True)


def measure(label: str, call: Callable[[], tuple[str, Any, float]],
            runs: int = 8) -> DriftResult:
    """
    重複呼叫同一個推論並量測輸出一致性。

    `call` 回傳 (原始字串, 解析後物件或 None, 耗時秒數)。
    第一次的輸出作為基準，後續與它比對 —— 這與論文的做法一致，
    也符合稽核情境：「當初那次的輸出」才是被記錄下來的那份。
    """
    raw_hashes, struct_fps, sem_fps, lats, samples = [], [], [], [], []
    err = None
    for _ in range(runs):
        try:
            raw, obj, lat = call()
        except Exception as e:                         # noqa: BLE001
            err = str(e)[:160]
            break
        raw_hashes.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
        struct_fps.append(_canon_struct(obj) if obj is not None else raw)
        sem_fps.append(_canon_semantic(obj) if obj is not None else raw)
        lats.append(lat)
        if len(samples) < 3:
            samples.append(raw[:400])

    n = len(raw_hashes)
    if n == 0:
        return DriftResult(label, 0, 0, 0, 0, 0, error=err)

    return DriftResult(
        label=label, runs=n,
        bitwise_identical=sum(1 for h in raw_hashes if h == raw_hashes[0]),
        structural_identical=sum(1 for s in struct_fps if s == struct_fps[0]),
        semantic_identical=sum(1 for s in sem_fps if s == sem_fps[0]),
        distinct_outputs=len(set(raw_hashes)),
        latencies=lats, samples=samples, error=err)


def render_table(results: list[DriftResult]) -> str:
    L = [
        "═" * 100,
        "  輸出漂移驗證（Output Drift Validation）",
        "  參考：Khatchadourian & Franco, arXiv:2511.07585 (IBM, 2025)",
        "═" * 100, "",
        f"  {'模型／任務':<26}{'次數':>5}{'位元級':>9}{'結構級':>9}{'語意級':>9}"
        f"{'相異輸出':>9}{'中位延遲':>10}  分級",
        "  " + "─" * 96,
    ]
    for r in results:
        if r.error:
            L.append(f"  {r.label:<26}{'—':>5}  ❌ {r.error[:52]}")
            continue
        med = statistics.median(r.latencies) if r.latencies else 0.0
        L.append(f"  {r.label:<26}{r.runs:>5}{r.bitwise_rate:>9.0%}"
                 f"{r.structural_rate:>9.0%}{r.semantic_rate:>9.0%}"
                 f"{r.distinct_outputs:>9}{med:>9.1f}s  {r.tier_label}")
    L += [
        "", "─" * 100,
        "  【三個層次的差別，以及為什麼都要看】",
        "    位元級 —— 輸出字串完全相同。**稽核要求的就是這個**：",
        "                重跑必須得到當初記錄下來的那份輸出，否則無法回答",
        "                「當初這個建議是根據什麼給的」。",
        "    結構級 —— JSON 鍵集合與型別相同。位元級不同但結構級相同，",
        "                代表只是空白或鍵順序差異，加一層正規化就能修。",
        "    語意級 —— 抽出的欄位值相同。結構級就不同的話，下游解析會壞，",
        "                那是真正的問題，不是格式問題。",
        "",
        "  【論文的一致性悖論】",
        "    Granite-3-8B 與 Qwen2.5-7B 達 100% 一致；GPT-OSS-120B 僅 12.5%",
        "    （Fisher's exact test，p < 0.0001）。**模型越大，輸出越不穩定。**",
        "    這推翻了「越大越好」的預設 —— 在需要稽核的場域，可重現性是硬需求。",
        "═" * 100,
    ]
    return "\n".join(L)
