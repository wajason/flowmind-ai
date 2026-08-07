-- ═══════════════════════════════════════════════════════════════════════════
-- FlowMind AI — 資料庫初始化（僅在 volume 第一次建立時由 docker-entrypoint 執行）
--
-- 這支 SQL 的核心不是「建表」，而是把「多客戶資料隔離」從
-- 「開發者記得在 SQL 加 WHERE tenant_id」降級成「資料庫層強制執行」。
--
-- 為什麼要這樣做？
--   會計師事務所、銀行授信部門、財顧公司同時服務數十家客戶，
--   法遵上必須有 information barrier（資訊隔離牆 / Chinese Wall）。
--   業界稽核時不會接受「我們的程式碼有加 WHERE」這種說法，
--   因為那是「開發者自律」，一次 code review 疏漏就是重大個資/營業秘密事故。
--   PostgreSQL Row-Level Security (RLS) 才是可稽核的技術控制點：
--   即使應用程式忘記加條件、或被 SQL injection 打穿，資料庫仍然擋住。
-- ═══════════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector：HNSW 稠密向量索引
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- 中文檔名/公司名模糊比對
CREATE EXTENSION IF NOT EXISTS pgcrypto;    -- audit log 的雜湊鏈

-- ───────────────────────────────────────────────────────────────────────────
-- 1. 應用程式角色
--    刻意「不是」superuser、「不是」table owner —— 這兩種身分都會繞過 RLS。
--    平常跑 pipeline 與查詢一律用這個角色連線。
-- ───────────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'flowmind_app') THEN
        CREATE ROLE flowmind_app LOGIN PASSWORD 'flowmind_app_pw';
    END IF;
END
$$;

-- ───────────────────────────────────────────────────────────────────────────
-- 2. Engagement（委任案）主檔
--    用語刻意採會計師事務所/財顧業的 "engagement"，不是工程師的 "project"：
--    一個客戶可能同時有多個 engagement（例如「2026 聯貸案」與「應收帳款承購案」），
--    而同一批文件的可見範圍是綁在 engagement 上，不是綁在公司上。
--    這是實務上資料隔離真正的最小單位。
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS engagements (
    tenant_id       TEXT PRIMARY KEY,               -- 'SHARED' 或 'CASE-0001'
    client_name     TEXT NOT NULL,
    engagement_type TEXT NOT NULL,                  -- 應收帳款承購 / 信保送件 / 聯貸案 ...
    industry_code   TEXT,                           -- 行業標準分類（對齊中小企業統計）
    status          TEXT NOT NULL DEFAULT 'active', -- active / on_hold / closed
    -- 保存期限：金融機構往來文件依商業會計法/銀行法多為 5 年起算，
    -- 到期後應可被自動清理，這是個資法「特定目的消失」的技術落實點。
    retention_until DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ
);

INSERT INTO engagements (tenant_id, client_name, engagement_type)
VALUES ('SHARED', '（共用知識庫）', '法規與融資商品公開資料')
ON CONFLICT (tenant_id) DO NOTHING;

-- ───────────────────────────────────────────────────────────────────────────
-- 3. documents：向量知識庫
--    fts_vector 存的是「中文字元 bigram」而非 to_tsvector('english', 中文)。
--    原因見 README §為什麼不用 to_tsvector('chinese')：
--    PostgreSQL 沒有內建中文分詞，對中文餵 english config 會退化成
--    「整段變一個 token」，BM25 稀疏檢索形同失效 —— 這是中文 RAG 最常見的隱形 bug。
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES engagements(tenant_id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024),                       -- BAAI/bge-m3
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    fts_vector  tsvector,
    file_hash   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, source, chunk_index)
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS documents_fts_idx      ON documents USING GIN (fts_vector);
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS documents_tenant_idx   ON documents (tenant_id);

-- ───────────────────────────────────────────────────────────────────────────
-- 4. RLS：資料庫層的資訊隔離牆
--    FORCE 是關鍵字 —— 沒有它，table owner 仍然可以無視 policy 讀到全部資料。
-- ───────────────────────────────────────────────────────────────────────────
ALTER TABLE documents   ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents   FORCE  ROW LEVEL SECURITY;
ALTER TABLE engagements ENABLE ROW LEVEL SECURITY;
ALTER TABLE engagements FORCE  ROW LEVEL SECURITY;

-- 讀：看得到「自己這個 engagement」+「共用知識庫」
-- 未設定 app.tenant_id 時 current_setting 回 NULL，比較結果為 NULL → 該列被濾掉，
-- 也就是 fail-closed（預設看不到東西），而不是 fail-open（預設全看得到）。
DROP POLICY IF EXISTS documents_read ON documents;
CREATE POLICY documents_read ON documents
    FOR SELECT
    USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR tenant_id = 'SHARED'
    );

-- 寫：只能動自己這個 engagement。
-- 即使被授權讀 SHARED，也絕對不能寫 SHARED —— 避免某個客戶的資料
-- 因為程式 bug 被寫進共用知識庫，變成其他所有客戶都檢索得到。
DROP POLICY IF EXISTS documents_write ON documents;
CREATE POLICY documents_write ON documents
    FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS engagements_read ON engagements;
CREATE POLICY engagements_read ON engagements
    FOR SELECT
    USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR tenant_id = 'SHARED'
        OR current_setting('app.role', true) = 'admin'   -- 管理主控台列出所有案件
    );

DROP POLICY IF EXISTS engagements_write ON engagements;
CREATE POLICY engagements_write ON engagements
    FOR ALL
    USING      (current_setting('app.role', true) = 'admin')
    WITH CHECK (current_setting('app.role', true) = 'admin');

-- ───────────────────────────────────────────────────────────────────────────
-- 5. 稽核軌跡（audit trail）
--    金融場域的內控稽核會問「誰、在什麼時候、用什麼問題、看到了哪幾份文件」。
--    prev_hash 串成雜湊鏈：任何一列被事後竄改或刪除，後續驗證都會斷鏈，
--    讓稽核紀錄本身也是可驗證的（tamper-evident），而不只是「有記就好」。
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor        TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    action       TEXT NOT NULL,          -- retrieve / ingest / export / answer
    query_text   TEXT,
    doc_sources  TEXT[],                 -- 這次實際被送進 LLM 的來源檔案
    confidence   NUMERIC(4,3),
    abstained    BOOLEAN,                -- 系統是否選擇拒答（可稽核的「留白」）
    prev_hash    TEXT,
    row_hash     TEXT
);
CREATE INDEX IF NOT EXISTS audit_log_tenant_ts_idx ON audit_log (tenant_id, ts DESC);

CREATE OR REPLACE FUNCTION audit_chain() RETURNS TRIGGER AS $$
DECLARE
    last_hash TEXT;
BEGIN
    SELECT row_hash INTO last_hash FROM audit_log ORDER BY id DESC LIMIT 1;
    NEW.prev_hash := COALESCE(last_hash, 'GENESIS');
    NEW.row_hash := encode(digest(
        NEW.prev_hash || NEW.ts::text || NEW.actor || NEW.tenant_id ||
        NEW.action || COALESCE(NEW.query_text, '') ||
        COALESCE(array_to_string(NEW.doc_sources, ','), ''),
        'sha256'), 'hex');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_chain_trg ON audit_log;
CREATE TRIGGER audit_chain_trg BEFORE INSERT ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_chain();

-- audit_log 刻意「只准新增、不准改刪」，連 app 角色都沒有 UPDATE/DELETE 權限。
GRANT SELECT, INSERT                     ON audit_log            TO flowmind_app;
GRANT SELECT, INSERT, UPDATE, DELETE     ON documents            TO flowmind_app;
GRANT SELECT, INSERT, UPDATE, DELETE     ON engagements          TO flowmind_app;
GRANT USAGE, SELECT                      ON ALL SEQUENCES IN SCHEMA public TO flowmind_app;
GRANT USAGE                              ON SCHEMA public        TO flowmind_app;
