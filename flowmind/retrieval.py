"""
flowmind.retrieval — Hybrid Search（Dense + CJK Sparse）+ RRF
=============================================================================
沿用 AnalogGenie-RAG 已驗證的雙路召回 + RRF + Small-to-Big 架構，
針對中文金融文件做了三處關鍵修改：

1. 稀疏那一路改用中文 bigram（見 textnorm.py 的說明）。
   原版對中文用 to_tsvector('english', …)，稀疏檢索實際上是壞的但不會報錯。

2. 檢索結果一律經過 RLS 過濾，程式碼裡完全不出現 `WHERE tenant_id = …`。
   隔離由資料庫負責，這裡負責檢索品質，職責分離。

3. 開啟 pgvector 0.8 的 iterative scan。
   原因：HNSW 是先取 ef_search 個候選、再套 RLS 過濾。若某個 engagement 的文件
   只佔全庫 2%，過濾後可能只剩個位數結果，召回率悄悄崩掉。
   iterative scan 會在結果不足時自動再往下掃，這是「多租戶 + 向量檢索」
   兩個需求撞在一起時真正的技術難點，不是加個 WHERE 就沒事了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import embeddings, textnorm

RRF_K = 60


@dataclass
class Chunk:
    source: str
    chunk_index: int
    tenant_id: str
    child_content: str      # 400 字元的精準命中片段
    parent_content: str     # 注入 LLM 的 1500~2500 字元完整脈絡
    category: str
    dense_score: float
    sparse_score: float
    rrf_score: float
    metadata: dict = field(default_factory=dict)

    @property
    def is_shared(self) -> bool:
        return self.tenant_id == "SHARED"

    @property
    def provenance(self) -> str:
        """給使用者看的來源標籤。標明是共用法規還是這家客戶自己的文件，
        因為這兩者在授信對話裡的份量完全不同。"""
        scope = "公開資料" if self.is_shared else "本案文件"
        return f"{self.source}#{self.chunk_index}（{scope}）"

    @property
    def published(self) -> Optional[str]:
        return self.metadata.get("published")

    @property
    def doc_status(self) -> str:
        """current / superseded / reference / 未登錄"""
        return self.metadata.get("doc_status") or "未登錄"

    @property
    def freshness_label(self) -> str:
        icon = {"current": "", "superseded": " ⚠️已被取代",
                "reference": " 📎僅供參考"}.get(self.doc_status, " ❓未登錄")
        return f"{self.published or '—'}{icon}"


def hybrid_search(
    conn,
    query: str,
    *,
    top_k: int = 8,
    max_per_source: int = 3,
    candidate_pool: int = 200,
    categories: Optional[list[str]] = None,
    include_superseded: bool = False,
) -> list[Chunk]:
    """
    雙路召回後用 RRF 融合，再做來源多樣性與版本過濾。

    max_per_source 的用意：一份 300 頁的白皮書若不設限，很容易把 top-8 全部佔滿，
    導致「銀行商品說明書」這種只有 3 頁但關鍵的文件永遠進不了 context。

    include_superseded 預設 False：已被新版取代的文件不進檢索。
    這擋的是引用驗證擋不住的一類錯誤 —— 系統可以完全正確地引用
    2015 年的舊作業手冊回答 2026 年的問題，引用驗證還會給 exact 100 分，
    因為那句話確實在那份文件裡。**引用是真的，答案是錯的。**
    要查歷史版本時才明確打開這個開關。

    注意這裡是**過濾**而不是調整分數。刻意不去動 RRF 分數：
    分數一旦被人為加權，「檢索強度」這個信心分項就不再是可獨立解讀的量了。
    """
    qvec = embeddings.to_pgvector(embeddings.embed_one(query))
    tsq = textnorm.to_fts_query(query)

    cat_filter = ""
    params_extra: list = []
    if categories:
        cat_filter = "AND metadata->>'category' = ANY(%s)"
        params_extra = [categories]

    rows: list[tuple] = []
    with conn.cursor() as cur:
        # 這幾個參數只影響本連線，不改全域設定。
        #
        # ── 為什麼是 strict_order 而不是 relaxed_order ────────────────────
        # 原本用 relaxed_order。實測發現**同一個 query 連跑三次，
        # 召回的 chunk 組合會不同**（top-1 與其分數相同，後段有變）。
        # relaxed_order 允許 HNSW 以「大致有序」的方式回傳結果，
        # 換取速度 —— 代價是結果不可重現。
        #
        # 這件事的後果比「排序稍微不同」嚴重得多：
        #   1. 後段 chunk 一變，top_dense 之外的覆蓋率判定就可能翻面，
        #      信心分數因此在 0.90 與 0.40（覆蓋率閘門上限）之間跳。
        #      50 題評測的 A/B 比較實際被這個因素污染過。
        #   2. 稽核問「當初這個建議是根據哪幾份文件」時，
        #      重跑可能給出不同的文件清單 —— 那個回答就不可信。
        #
        # 在授信這種要留證的場域，**可重現性優先於延遲**。
        # strict_order 保證同一個 query 得到同一批結果。
        # 若日後語料成長到 strict_order 太慢，正確的做法是
        # 把檢索結果快照存進 audit_log，而不是換回 relaxed_order。
        try:
            cur.execute("SET hnsw.iterative_scan = strict_order")
            cur.execute("SET hnsw.ef_search = 200")
        except Exception:                              # noqa: BLE001
            conn.rollback()   # pgvector < 0.8 沒有這些參數，退回一般掃描即可

        # ── 每一個 ORDER BY 都補上 id 當最終決勝鍵 ────────────────────────
        #
        # 沒有 id 的話，這三個排序**都不是全序**，而不是全序的 ORDER BY
        # 配上 LIMIT，回傳哪幾列在 SQL 語意上就是未定義的：
        #
        #   dense   距離相同的 chunk（近乎重複的段落）順序不定
        #   sparse  **ts_rank 產生大量相同分數**，LIMIT 200 取哪 200 筆是任意的
        #           —— 這是最大的元凶
        #   rrf     兩路排名相同時併出相同的 rrf
        #
        # 實測：同一個 query 連跑四次得到 4 種不同的 chunk 組合。
        # 後果不只是「排序稍微不同」——後段 chunk 一變，覆蓋率判定就可能翻面，
        # 信心分數在 0.90 與 0.40（覆蓋率閘門上限）之間跳，
        # 50 題評測的 A/B 比較實際被這個因素污染過。
        #
        # 對一個要留證的產品來說更嚴重的是：稽核問「當初根據哪幾份文件」時，
        # 重跑會給出不同的清單。加 id 之後排序成為全序，結果可重現。
        sql = f"""
            WITH dense AS (
                SELECT id, tenant_id, source, chunk_index, content, metadata,
                       1 - (embedding <=> %s::vector) AS dense_score,
                       ROW_NUMBER() OVER (
                           ORDER BY embedding <=> %s::vector, id
                       ) AS rank_d
                FROM documents
                WHERE embedding IS NOT NULL {cat_filter}
                ORDER BY embedding <=> %s::vector, id
                LIMIT {candidate_pool}
            ),
            sparse AS (
                SELECT id, tenant_id, source, chunk_index, content, metadata,
                       ts_rank(fts_vector, to_tsquery('simple', %s)) AS sparse_score,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(fts_vector, to_tsquery('simple', %s)) DESC,
                                    id
                       ) AS rank_s
                FROM documents
                WHERE fts_vector @@ to_tsquery('simple', %s) {cat_filter}
                ORDER BY ts_rank(fts_vector, to_tsquery('simple', %s)) DESC, id
                LIMIT {candidate_pool}
            )
            SELECT COALESCE(d.tenant_id, s.tenant_id),
                   COALESCE(d.source, s.source),
                   COALESCE(d.chunk_index, s.chunk_index),
                   COALESCE(d.content, s.content),
                   COALESCE(d.metadata, s.metadata),
                   COALESCE(d.dense_score, 0),
                   COALESCE(s.sparse_score, 0),
                   COALESCE(1.0 / ({RRF_K} + d.rank_d), 0.0)
                 + COALESCE(1.0 / ({RRF_K} + s.rank_s), 0.0) AS rrf
            FROM dense d FULL OUTER JOIN sparse s ON d.id = s.id
            ORDER BY rrf DESC, COALESCE(d.id, s.id)
            LIMIT {top_k * 6};
        """
        # 參數順序必須與 SQL 中 %s 的**出現順序**完全一致。
        # 順序錯了不會拋錯，只會安靜地用錯參數 —— 所以逐段列出來對：
        #   dense  : qvec(select) qvec(row_number) [cats] qvec(order by)
        #   sparse : tsq(ts_rank) tsq(row_number) tsq(where) [cats] tsq(order by)
        # 最後那個 tsq 是加上決勝鍵時新增的 ORDER BY 帶來的。
        params: list = [qvec, qvec]
        if categories:
            params.append(categories)
        params.append(qvec)
        params += [tsq, tsq, tsq]
        if categories:
            params.append(categories)
        params.append(tsq)

        cur.execute(sql, params)
        rows = cur.fetchall()

    # 版本過濾 + 多樣性過濾
    seen: dict[str, int] = {}
    out: list[Chunk] = []
    dropped_superseded: set[str] = set()
    for (tenant, source, idx, content, meta, dense_s, sparse_s, rrf) in rows:
        meta = meta if isinstance(meta, dict) else {}
        if not include_superseded and meta.get("doc_status") == "superseded":
            dropped_superseded.add(source)
            continue
        if seen.get(source, 0) >= max_per_source:
            continue
        seen[source] = seen.get(source, 0) + 1
        out.append(Chunk(
            tenant_id=tenant,
            source=source,
            chunk_index=idx,
            child_content=content,
            parent_content=meta.get("parent_content", content),
            category=meta.get("category", "未分類"),
            dense_score=float(dense_s),
            sparse_score=float(sparse_s),
            rrf_score=float(rrf),
            metadata=meta,
        ))
        if len(out) >= top_k:
            break

    if dropped_superseded:
        # 記在第一個 chunk 上，讓透明度面板能顯示「有舊版本被擋掉」。
        # 靜靜地過濾掉東西而不告訴使用者，是另一種形式的不透明。
        if out:
            out[0].metadata.setdefault(
                "_dropped_superseded", sorted(dropped_superseded))
    return out


def retrieval_diagnostics(chunks: list[Chunk]) -> dict:
    """
    給信心分數與透明度面板用的檢索健康度指標。

    `sparse_contributing` 特別重要：如果它長期是 0，代表中文分詞又壞了，
    整個 Hybrid Retrieval 已經退化成單路，但分數面板還是會照常顯示。
    把它做成明確指標，就不會再有那種「看起來正常」的靜默失效。
    """
    if not chunks:
        return {"n": 0, "top_rrf": 0.0, "distinct_sources": 0,
                "sparse_contributing": 0, "dense_contributing": 0,
                "own_docs": 0, "shared_docs": 0}
    return {
        "n": len(chunks),
        "top_rrf": max(c.rrf_score for c in chunks),
        "top_dense": max(c.dense_score for c in chunks),
        "distinct_sources": len({c.source for c in chunks}),
        "sparse_contributing": sum(1 for c in chunks if c.sparse_score > 0),
        "dense_contributing": sum(1 for c in chunks if c.dense_score > 0),
        "own_docs": sum(1 for c in chunks if not c.is_shared),
        "shared_docs": sum(1 for c in chunks if c.is_shared),
    }
