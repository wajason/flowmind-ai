-- ═══════════════════════════════════════════════════════════════════════════
-- 委任案財務紀錄：發票／應收帳款、合約、銀行流水
--
-- 【這支 SQL 補的是一個被忽略的隔離缺口】
--
-- 先前的隔離故事是「用 Row-Level Security 保護委任案資料」，
-- 但實際上 RLS 只保護了 documents 與 chunks（也就是**文件與其向量**）。
-- 真正敏感的財務明細 —— 發票、應收帳款、合約條款、銀行流水 ——
-- 一直放在磁碟上的 JSON/CSV 檔案裡，由 metrics.load_engagement_files() 直接讀。
--
-- 也就是說：**資料庫層的隔離保證，管不到最敏感的那批資料。**
-- 檔案系統上只要路徑拼對了就讀得到，沒有任何強制隔離。
--
-- 這不是理論上的漏洞。一個把 tenant_id 串進路徑的小 bug，
-- 或一次目錄權限設定失誤，就會讓 A 客戶的發票被 B 客戶的查詢讀到，
-- 而且不會有任何錯誤訊息 —— 這正是最難發現的那種洩漏。
--
-- 【搬進資料庫之後多拿到的三件事】
--
--   1. 隔離變成資料庫強制，與文件同一套機制，不再有兩套規則
--   2. 監控查詢可以用 SQL 表達（帳期逾期、集中度、趨勢惡化），
--      不必把整包檔案讀進記憶體再用 Python 迴圈算
--   3. 每一筆警示都能附上**產生它的那幾列**，警示因此可以被逐列複查
--
-- 【刻意保留檔案作為輸入】
-- 檔案仍然是資料的來源（客戶就是給你檔案），
-- 這裡做的是**匯入**，不是取代。原始檔保留，因為稽核要能回到原件。
-- ═══════════════════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────────────────
-- 1. 發票／應收帳款
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fin_invoices (
    tenant_id       TEXT NOT NULL,
    invoice_number  TEXT NOT NULL,
    -- 買方（債務人）。應收帳款融資真正在意的是**這一方**的信用，
    -- 因為還款來源是買方付款，不是賣方營運。
    buyer_name      TEXT,
    buyer_ban       TEXT,
    seller_name     TEXT,
    seller_ban      TEXT,
    issue_date      DATE,
    due_date        DATE,
    amount          NUMERIC(18, 2),
    tax_amount      NUMERIC(18, 2),
    total_amount    NUMERIC(18, 2),
    payment_terms_days INTEGER,
    status          TEXT,          -- OPEN / PAID / OVERDUE / WRITTEN_OFF / ...
    paid_date       DATE,
    contract_id     TEXT,
    -- 原始整列保留：匯入時如果我們的欄位對應理解錯了，
    -- 還原得回去。丟掉原始資料的匯入是不可逆的錯誤。
    raw             JSONB NOT NULL DEFAULT '{}'::JSONB,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, invoice_number)
);

CREATE INDEX IF NOT EXISTS idx_fin_inv_tenant_due
    ON fin_invoices (tenant_id, due_date);
CREATE INDEX IF NOT EXISTS idx_fin_inv_tenant_buyer
    ON fin_invoices (tenant_id, buyer_ban);
CREATE INDEX IF NOT EXISTS idx_fin_inv_tenant_status
    ON fin_invoices (tenant_id, status);

-- ───────────────────────────────────────────────────────────────────────────
-- 2. 合約
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fin_contracts (
    tenant_id       TEXT NOT NULL,
    contract_id     TEXT NOT NULL,
    counterparty    TEXT,
    counterparty_ban TEXT,
    signed_date     DATE,
    start_date      DATE,
    end_date        DATE,
    payment_terms_days INTEGER,
    contract_amount NUMERIC(18, 2),
    raw             JSONB NOT NULL DEFAULT '{}'::JSONB,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, contract_id)
);

-- ───────────────────────────────────────────────────────────────────────────
-- 3. 銀行流水
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fin_ledger (
    tenant_id       TEXT NOT NULL,
    entry_id        BIGSERIAL,
    txn_date        DATE,
    description     TEXT,
    counterparty    TEXT,
    amount          NUMERIC(18, 2),
    balance         NUMERIC(18, 2),
    ref_invoice     TEXT,
    raw             JSONB NOT NULL DEFAULT '{}'::JSONB,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, entry_id)
);

CREATE INDEX IF NOT EXISTS idx_fin_ledger_tenant_date
    ON fin_ledger (tenant_id, txn_date);

-- ───────────────────────────────────────────────────────────────────────────
-- 4. 監控警示
--    警示存進資料庫而不是只印在畫面上，因為「秘書」的價值在於
--    **有人沒看的時候它仍然記得**。同時也讓「這條警示什麼時候發的、
--    誰處理的」變成可稽核的紀錄。
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fin_alerts (
    tenant_id     TEXT NOT NULL,
    alert_id      BIGSERIAL,
    rule_id       TEXT NOT NULL,       -- WATCH-01 …
    severity      TEXT NOT NULL,       -- critical / warning / info
    title         TEXT NOT NULL,
    detail        TEXT NOT NULL,
    -- 觸發這條警示的實際資料列。警示必須能被逐列複查，
    -- 否則它就只是一個要人相信的紅字。
    evidence      JSONB NOT NULL DEFAULT '[]'::JSONB,
    -- 指紋：同一條規則、同一批證據不重複發。
    -- 沒有這個，每天掃描都會把同一件事再喊一次，
    -- 然後使用者就會開始忽略所有警示 —— 那等於沒有警示。
    fingerprint   TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at   TIMESTAMPTZ,
    PRIMARY KEY (tenant_id, alert_id),
    UNIQUE (tenant_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_fin_alerts_open
    ON fin_alerts (tenant_id, resolved_at, severity);

-- ───────────────────────────────────────────────────────────────────────────
-- 5. Row-Level Security
--
--    與 documents / kg_nodes 用完全相同的機制與寫法。
--    「同一套規則」本身就是安全性質 —— 兩套規則就會有兩套漏洞，
--    而且第二套通常沒人記得檢查。
--
--    這裡刻意**不**開放 SHARED：財務明細沒有「公開」這個概念，
--    法規與金融商品說明才有。一筆發票不屬於任何人共用。
-- ───────────────────────────────────────────────────────────────────────────
ALTER TABLE fin_invoices  ENABLE ROW LEVEL SECURITY;
ALTER TABLE fin_contracts ENABLE ROW LEVEL SECURITY;
ALTER TABLE fin_ledger    ENABLE ROW LEVEL SECURITY;
ALTER TABLE fin_alerts    ENABLE ROW LEVEL SECURITY;

-- FORCE：連表格擁有者也要受政策約束。
-- 少了這行，用擁有者身分連線就會靜默繞過所有隔離。
ALTER TABLE fin_invoices  FORCE ROW LEVEL SECURITY;
ALTER TABLE fin_contracts FORCE ROW LEVEL SECURITY;
ALTER TABLE fin_ledger    FORCE ROW LEVEL SECURITY;
ALTER TABLE fin_alerts    FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS fin_invoices_rw ON fin_invoices;
CREATE POLICY fin_invoices_rw ON fin_invoices FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS fin_contracts_rw ON fin_contracts;
CREATE POLICY fin_contracts_rw ON fin_contracts FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS fin_ledger_rw ON fin_ledger;
CREATE POLICY fin_ledger_rw ON fin_ledger FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS fin_alerts_rw ON fin_alerts;
CREATE POLICY fin_alerts_rw ON fin_alerts FOR ALL
    USING      (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));

GRANT SELECT, INSERT, UPDATE, DELETE
    ON fin_invoices, fin_contracts, fin_ledger, fin_alerts TO flowmind_app;
GRANT USAGE, SELECT ON SEQUENCE fin_ledger_entry_id_seq TO flowmind_app;
GRANT USAGE, SELECT ON SEQUENCE fin_alerts_alert_id_seq TO flowmind_app;
