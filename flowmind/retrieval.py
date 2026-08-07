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


def hybrid_search(
    conn,
    query: str,
    *,
    top_k: int = 8,
    max_per_source: int = 3,
    candidate_pool: int = 200,
    categories: Optional[list[str]] = None,
) -> list[Chunk]:
    """
    雙路召回後用 RRF 融合，再做來源多樣性過濾。

    max_per_source 的用意：一份 300 頁的白皮書若不設限，很容易把 top-8 全部佔滿，
    導致「銀行商品說明書」這種只有 3 頁但關鍵的文件永遠進不了 context。
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
        # 這兩個參數只影響本連線，不改全域設定
        try:
            cur.execute("SET hnsw.iterative_scan = relaxed_order")
            cur.execute("SET hnsw.ef_search = 200")
        except Exception:                              # noqa: BLE001
            conn.rollback()   # pgvector < 0.8 沒有這些參數，退回一般掃描即可

        sql = f"""
            WITH dense AS (
                SELECT id, tenant_id, source, chunk_index, content, metadata,
                       1 - (embedding <=> %s::vector) AS dense_score,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank_d
                FROM documents
                WHERE embedding IS NOT NULL {cat_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT {candidate_pool}
            ),
            sparse AS (
                SELECT id, tenant_id, source, chunk_index, content, metadata,
                       ts_rank(fts_vector, to_tsquery('simple', %s)) AS sparse_score,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(fts_vector, to_tsquery('simple', %s)) DESC
                       ) AS rank_s
                FROM documents
                WHERE fts_vector @@ to_tsquery('simple', %s) {cat_filter}
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
            ORDER BY rrf DESC
            LIMIT {top_k * 6};
        """
        # 參數順序必須與 SQL 中 %s 的出現順序完全一致：
        #   dense  : qvec(select) qvec(row_number) [cats] qvec(order by)
        #   sparse : tsq(ts_rank) tsq(row_number) tsq(where) [cats]
        params: list = [qvec, qvec]
        if categories:
            params.append(categories)
        params.append(qvec)
        params += [tsq, tsq, tsq]
        if categories:
            params.append(categories)

        cur.execute(sql, params)
        rows = cur.fetchall()

    # 多樣性過濾
    seen: dict[str, int] = {}
    out: list[Chunk] = []
    for (tenant, source, idx, content, meta, dense_s, sparse_s, rrf) in rows:
        meta = meta if isinstance(meta, dict) else {}
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
