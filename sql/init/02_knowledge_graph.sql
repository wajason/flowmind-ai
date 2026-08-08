-- ═══════════════════════════════════════════════════════════════════════════
-- 知識圖譜（以 PostgreSQL 遞迴 CTE 實作，不引入圖資料庫）
--
-- 【為什麼需要圖，以及為什麼不用 Neo4j】
--
-- 向量相似度有一個原理性的限制（見 docs/DECISIONS.md U-01）：
-- 它衡量「語意接近」，而「日本的保證成數」與「台灣的保證成數」語意上就是接近的。
-- **embedding 分不出指涉對象是誰。**
--
-- 解法不是換更好的 embedding，是把「指涉對象」從文字裡拉出來變成**結構**。
-- 一旦「這份文件適用於哪個國家、哪個年度、哪個主體」是圖上的節點與邊，
-- 「日本 ≠ 台灣」就是一次 JOIN 的事，不是相似度的事。
--
-- 不用 Neo4j 的理由很實際：
--   · 多一個服務就多一份維運成本與一份可能不同步的資料副本
--   · 我們的圖是稀疏且淺的（2–4 跳），遞迴 CTE 完全夠用
--   · 資料已經在 PostgreSQL 裡，圖與 RLS、稽核共用同一套隔離機制 ——
--     換成外部圖庫就要把資訊隔離牆再實作一次，那是新的破口
--
-- 【圖的兩個用途】
--   1. 指涉範圍比對（解 U-01）：文件 -[適用於]-> 國家/年度/法規版本
--   2. 關係網絡追查：企業 -[負責人]-> 自然人 <-[負責人]- 另一家企業
--      這是純憑證比對抓不到的造假樣態（關係企業循環交易）
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS kg_nodes (
    node_id    TEXT PRIMARY KEY,
    -- company 企業 / person 自然人 / document 文件 / invoice 發票 /
    -- contract 合約 / jurisdiction 法域 / period 期間 / topic 主題
    node_type  TEXT NOT NULL,
    label      TEXT NOT NULL,
    tenant_id  TEXT NOT NULL DEFAULT 'SHARED',
    attrs      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS kg_nodes_type_idx   ON kg_nodes(node_type);
CREATE INDEX IF NOT EXISTS kg_nodes_tenant_idx ON kg_nodes(tenant_id);
CREATE INDEX IF NOT EXISTS kg_nodes_label_trgm ON kg_nodes USING GIN (label gin_trgm_ops);

CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id    BIGSERIAL PRIMARY KEY,
    src_id     TEXT NOT NULL REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
    dst_id     TEXT NOT NULL REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
    -- issued_to 開票給 / party_to 為合約當事人 / represented_by 負責人 /
    -- applies_to 適用於 / supersedes 取代 / cites 引用 / paid_via 收款經由
    edge_type  TEXT NOT NULL,
    tenant_id  TEXT NOT NULL DEFAULT 'SHARED',
    weight     NUMERIC,
    attrs      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (src_id, dst_id, edge_type, tenant_id)
);
CREATE INDEX IF NOT EXISTS kg_edges_src_idx    ON kg_edges(src_id, edge_type);
CREATE INDEX IF NOT EXISTS kg_edges_dst_idx    ON kg_edges(dst_id, edge_type);
CREATE INDEX IF NOT EXISTS kg_edges_tenant_idx ON kg_edges(tenant_id);

-- ── RLS：圖與文件共用同一套隔離機制 ─────────────────────────────────────
-- 這是不用外部圖庫最重要的理由：換成 Neo4j 就要把資訊隔離牆再實作一次。
ALTER TABLE kg_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_nodes FORCE  ROW LEVEL SECURITY;
ALTER TABLE kg_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE kg_edges FORCE  ROW LEVEL SECURITY;

DROP POLICY IF EXISTS kg_nodes_read ON kg_nodes;
CREATE POLICY kg_nodes_read ON kg_nodes FOR SELECT
    USING (tenant_id = current_setting('app.tenant_id', true) OR tenant_id = 'SHARED');
DROP POLICY IF EXISTS kg_nodes_write ON kg_nodes;
CREATE POLICY kg_nodes_write ON kg_nodes FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS kg_edges_read ON kg_edges;
CREATE POLICY kg_edges_read ON kg_edges FOR SELECT
    USING (tenant_id = current_setting('app.tenant_id', true) OR tenant_id = 'SHARED');
DROP POLICY IF EXISTS kg_edges_write ON kg_edges;
CREATE POLICY kg_edges_write ON kg_edges FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

GRANT SELECT, INSERT, UPDATE, DELETE ON kg_nodes, kg_edges TO flowmind_app;
GRANT USAGE, SELECT ON SEQUENCE kg_edges_edge_id_seq TO flowmind_app;

-- ═══════════════════════════════════════════════════════════════════════════
-- 多跳查詢：從一個節點出發，走 N 跳能到哪些節點
--
-- 遞迴 CTE 帶 path 陣列做環路偵測 —— 沒有這個保護，
-- 一旦圖裡有環（而關係企業的圖一定有環），查詢會無限展開。
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION kg_neighbors(
    p_start TEXT, p_max_hops INT DEFAULT 3,
    p_edge_types TEXT[] DEFAULT NULL)
RETURNS TABLE(node_id TEXT, node_type TEXT, label TEXT,
              hops INT, path TEXT[], edge_path TEXT[]) AS $fn$
BEGIN
    RETURN QUERY
    WITH RECURSIVE walk AS (
        SELECT n.node_id, n.node_type, n.label,
               0 AS hops, ARRAY[n.node_id] AS path, ARRAY[]::TEXT[] AS edge_path
        FROM kg_nodes n WHERE n.node_id = p_start

        UNION ALL

        SELECT n.node_id, n.node_type, n.label,
               w.hops + 1, w.path || n.node_id, w.edge_path || e.edge_type
        FROM walk w
        JOIN kg_edges e ON (e.src_id = w.node_id OR e.dst_id = w.node_id)
        JOIN kg_nodes n ON n.node_id = CASE WHEN e.src_id = w.node_id
                                            THEN e.dst_id ELSE e.src_id END
        WHERE w.hops < p_max_hops
          AND NOT n.node_id = ANY(w.path)
          AND (p_edge_types IS NULL OR e.edge_type = ANY(p_edge_types))
    )
    SELECT DISTINCT ON (walk.node_id)
           walk.node_id, walk.node_type, walk.label, walk.hops, walk.path, walk.edge_path
    FROM walk WHERE walk.hops > 0
    ORDER BY walk.node_id, walk.hops;
END;
$fn$ LANGUAGE plpgsql STABLE;

-- ═══════════════════════════════════════════════════════════════════════════
-- 循環交易偵測：A 開票給 B，B 也開票給 A
-- 兩張發票各自完全合法，只有放進同一張圖才看得出是循環。
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION kg_find_circular_trades()
RETURNS TABLE(company_a TEXT, company_b TEXT,
              a_to_b_amount NUMERIC, b_to_a_amount NUMERIC) AS $fn$
    SELECT n1.label, n2.label, SUM(e1.weight), SUM(e2.weight)
    FROM kg_edges e1
    JOIN kg_edges e2 ON e1.src_id = e2.dst_id AND e1.dst_id = e2.src_id
    JOIN kg_nodes n1 ON n1.node_id = e1.src_id
    JOIN kg_nodes n2 ON n2.node_id = e1.dst_id
    WHERE e1.edge_type = 'issued_to' AND e2.edge_type = 'issued_to'
      AND e1.src_id < e1.dst_id
    GROUP BY n1.label, n2.label;
$fn$ LANGUAGE sql STABLE;

-- ═══════════════════════════════════════════════════════════════════════════
-- 共同負責人偵測：A 與 B 的負責人是同一個自然人
-- 資料來源為經濟部商工登記公示資料。這是關係企業最直接的證據，
-- 也是統編前綴比對（RELATED-01，弱訊號）真正該被取代的方式。
-- ═══════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION kg_find_shared_representatives()
RETURNS TABLE(person TEXT, companies TEXT[], company_count INT) AS $fn$
    SELECT p.label, ARRAY_AGG(c.label ORDER BY c.label), COUNT(*)::INT
    FROM kg_edges e
    JOIN kg_nodes c ON c.node_id = e.src_id AND c.node_type = 'company'
    JOIN kg_nodes p ON p.node_id = e.dst_id AND p.node_type = 'person'
    WHERE e.edge_type = 'represented_by'
    GROUP BY p.label
    HAVING COUNT(*) > 1;
$fn$ LANGUAGE sql STABLE;
