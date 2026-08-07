"""
flowmind.textnorm — 中文稀疏檢索的分詞層
=============================================================================
【為什麼需要這支檔案：一個中文 RAG 幾乎人人都踩、但幾乎沒人發現的坑】

PostgreSQL 沒有內建中文分詞器。原本沿用的 AnalogGenie 程式碼寫的是：

    to_tsvector('english', 中文內容)

對英文論文完全正確，但餵中文時，english parser 找不到空白與詞界，
會把整段中文吐成極少數幾個超長 token。實測後果：
  * 稀疏（BM25）那一路幾乎永遠 0 命中
  * Hybrid Retrieval 名義上是雙路召回，實際上退化成純向量單路
  * 而 RRF 融合分數仍然算得出來、面板仍然亮著綠燈 —— 所以「看起來正常」

在供應鏈金融場域這件事的代價很具體：
「統一編號 84726193」「應收帳款承購」「無追索權」這種
必須逐字精準命中的關鍵詞，正是稠密向量最不擅長、必須靠稀疏檢索補的部分。

【解法】不引入 pg_jieba/zhparser（要編譯 C 擴充，Windows 上很痛苦，
且中斷了「docker compose up 就能跑」的可重現性），改用字元 bigram：

    「應收帳款承購」→ 應收 收帳 帳款 款承 承購

再用 to_tsvector('simple', bigram字串) 建 GIN 索引。
這是資訊檢索領域對 CJK 的標準做法（Lucene 的 CJKAnalyzer 就是這樣做的），
無字典、無外部相依、對未登錄詞（公司名、產品型號）反而比詞典分詞更強韌。
"""

from __future__ import annotations

import re

# 中日韓統一表意文字 + 擴充區 + 相容表意文字
_CJK = r"㐀-䶿一-鿿豈-﫿"
_CJK_RUN = re.compile(f"[{_CJK}]+")
# 英數 token：保留 AB-45678901 這種發票號碼、SWIFT code、統編
_ALNUM = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*|\d[\d,\.\-]*")

# tsquery 的保留字元，混進去會讓 to_tsquery 直接語法錯誤
_TSQUERY_UNSAFE = re.compile(r"[&|!():'<>\\\s]")


def cjk_bigrams(text: str) -> list[str]:
    """把中文連續片段切成字元 bigram；單字的片段保留原字。"""
    out: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def tokenize(text: str) -> list[str]:
    """混合語言分詞：中文走 bigram，英數走原樣保留（小寫化）。"""
    tokens = cjk_bigrams(text)
    tokens += [t.lower().rstrip(",.") for t in _ALNUM.findall(text)]
    return [t for t in tokens if t]


def to_fts_document(text: str, max_chars: int = 20000) -> str:
    """
    產生要餵給 to_tsvector('simple', ...) 的字串。
    截斷是刻意的：parent chunk 最長 2500 字元，20000 是給整份文件摘要用的上限，
    超過就代表呼叫端傳錯東西了，寧可截斷也不要讓單列 tsvector 爆掉 1MB 上限。
    """
    return " ".join(tokenize(text[:max_chars]))


def to_fts_query(query: str, max_tokens: int = 64) -> str:
    """
    產生 to_tsquery('simple', ...) 用的查詢字串，token 之間用 OR。

    為什麼是 OR 不是 AND：使用者問「宏昇機械的應收帳款什麼時候到期」，
    用 AND 幾乎必然 0 命中（bigram 全中的機率極低）。用 OR 才會回到
    我們要的 BM25 行為：命中越多、越罕見的 bigram，ts_rank 分數越高。
    """
    seen: list[str] = []
    for t in tokenize(query):
        t = _TSQUERY_UNSAFE.sub("", t)
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= max_tokens:
            break
    return " | ".join(seen)


# ── 簡轉繁 ────────────────────────────────────────────────────────────────
# 本地中文模型（不分廠牌）偶爾會漏出簡體字：實測 skill_builder 產出的
# 20,776 字裡有 8 個（实/准/务/权）。比例極低，但這份檔案是要交給銀行看的，
# 出現一個簡體字就會讓人懷疑整份文件是不是從對岸資料抄來的。
#
# 用 's2tw'（字元層）而不是 's2twp'（含詞彙替換）：
# 我們要修的是漏出來的簡體字，不是把已經正確的繁體用語再改寫一遍。
# 詞彙層轉換有機會動到引文，反而破壞逐字引用的比對。
_converter = None
_converter_failed = False


def to_traditional(text: str) -> str:
    """簡轉繁。轉換器載入失敗時原樣回傳，但只警告一次，不靜默吞掉。"""
    global _converter, _converter_failed
    if not text or _converter_failed:
        return text
    if _converter is None:
        try:
            from opencc import OpenCC
            _converter = OpenCC("s2tw")
            # 載入後立刻自我驗證：opencc 有些版本設定檔缺失時會回傳原字串
            # 而不是報錯，那種「靜默失敗」會讓人以為轉換有生效。
            if _converter.convert("测试") == "测试":
                raise RuntimeError("opencc 載入成功但轉換無效果（設定檔可能缺失）")
        except Exception as e:                         # noqa: BLE001
            _converter_failed = True
            print(f"  ⚠️  簡轉繁不可用（{e}）；輸出將保留模型原始用字。"
                  f"對外交付前請人工檢查是否混入簡體字。")
            return text
    return _converter.convert(text)


# 只放「在繁體中文裡確定不會出現」的簡體字。
# 刻意排除幾個常被誤判的正字：
#   准（核准、准許）、划（划算、划船）、据（拮据）、戶的異體字 户
# 這些在繁體都是合法用字，列進來會產生偽陽性，讓這個檢查失去可信度 ——
# 一個會誤報的檢查，用幾次之後就沒有人會理它了。
_SIMPLIFIED_PROBE = "实证责险产资济营务标说认识测规则处计关键权义银钱账价该报单价华丽"


def count_simplified(text: str) -> dict[str, int]:
    """交付前的自我檢查：數一數還有沒有簡體字漏網。"""
    from collections import Counter
    return dict(Counter(c for c in (text or "") if c in _SIMPLIFIED_PROBE))


def normalize_tax_id(raw: str | None) -> str | None:
    """台灣統一編號正規化：只留 8 位數字，長度不對就回 None（不猜、不補零）。"""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    return digits if len(digits) == 8 else None


def validate_tax_id(tax_id: str | None) -> bool:
    """
    財政部統一編號檢核碼演算法（民國 110 年新制，容許第 7 位為 7 的兩種算法）。

    這是一段「決定性運算」而不是交給 LLM 判斷 —— 也是本專案的一貫原則：
    凡是有明確規則可以算的，就不要讓語言模型去猜。
    一個統編是真是假，是可以在 0.1 毫秒內用純算術確定的事。
    """
    tax_id = normalize_tax_id(tax_id)
    if tax_id is None:
        return False
    weights = [1, 2, 1, 2, 1, 2, 4, 1]
    total = 0
    for digit, w in zip((int(c) for c in tax_id), weights):
        product = digit * w
        total += product // 10 + product % 10
    if total % 5 == 0:
        return True
    # 第 7 位為 7 時，該位的加權積可視為 10（進位）→ 總和 +1 後再檢查
    return tax_id[6] == "7" and (total + 1) % 5 == 0
