-- ═══════════════════════════════════════════════════════════════════════════
-- 認證與授權鏈（Authentication → Authorization）
--
-- 【這支 SQL 補的是一個真實的安全缺口】
--
-- Row-Level Security 做的是**授權**：給定「這條連線代表 CASE-0001」，
-- 強制它只能看到 CASE-0001 的資料。
--
-- 但「這條連線憑什麼代表 CASE-0001」是**認證**問題，RLS 管不到。
-- 原本的做法是應用程式直接 `SET app.tenant_id = 'CASE-0001'` ——
-- 這代表**應用程式被打穿，RLS 就形同虛設**：攻擊者只要能執行任意 SQL，
-- 就能把自己設成任何 tenant。
--
-- 【修法：讓應用程式無法自行宣告身分】
--
-- 應用程式不再傳 tenant_id，改傳**工作階段權杖**。
-- 由一個 SECURITY DEFINER 函式驗證權杖、查出該使用者被授權的委任案，
-- 才設定 app.tenant_id。應用程式沒有直接設定 GUC 的權限。
--
-- 結果是一條可稽核的完整鏈：
--     權杖 → 使用者 → 授權紀錄（誰授權的、何時到期）→ 委任案 → RLS 過濾
--
-- 每一環都留在資料庫裡，內控稽核可以逐環查證。
--
-- 【誠實的限制】
-- 這仍然假設「權杖沒有被竊取」。正式部署應把權杖發放接上企業 SSO/OIDC，
-- 由 IdP 負責身分證明（含 MFA），本層只負責「這個已驗證身分能碰哪些委任案」。
-- 本層**不是**用來取代 SSO，是用來確保「就算 App 層被打穿，
-- 也無法憑空取得跨委任案的存取權」。
-- ═══════════════════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────────────────
-- 1. 主體（使用者／服務帳號）
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS principals (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    -- analyst 授信人員 / reviewer 覆核主管 / admin 系統管理 / service 服務帳號
    role         TEXT NOT NULL DEFAULT 'analyst',
    -- 外部身分提供者的 subject（接 SSO 時填入），本地測試可為 NULL
    idp_subject  TEXT UNIQUE,
    -- 本地密碼僅供開發與離線 demo；正式環境應一律走 SSO 而非本欄
    password_hash TEXT,
    status       TEXT NOT NULL DEFAULT 'active',   -- active / suspended / disabled
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ───────────────────────────────────────────────────────────────────────────
-- 2. 委任案存取授權
--
-- 這張表是「資訊隔離牆」的法遵載體：
-- 誰、被誰、在什麼時候、授權存取哪一個委任案、到什麼時候到期。
-- 會計師事務所的獨立性檢查與銀行的利益衝突迴避，查的就是這張表。
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS engagement_access (
    user_id     TEXT NOT NULL REFERENCES principals(user_id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL REFERENCES engagements(tenant_id) ON DELETE CASCADE,
    access_level TEXT NOT NULL DEFAULT 'read',      -- read / write / owner
    granted_by  TEXT NOT NULL,
    granted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 存取權限預設會到期。長期有效的授權是資訊隔離牆最常見的破口 ——
    -- 人員調動後權限沒收回，是稽核最常開的缺失。
    expires_at  TIMESTAMPTZ,
    revoked_at  TIMESTAMPTZ,
    PRIMARY KEY (user_id, tenant_id)
);
CREATE INDEX IF NOT EXISTS engagement_access_user_idx ON engagement_access(user_id);

-- ───────────────────────────────────────────────────────────────────────────
-- 3. 工作階段
--
-- 只存權杖的雜湊，不存權杖本身 —— 資料庫被讀走也無法冒用既有工作階段。
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES principals(user_id) ON DELETE CASCADE,
    issued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked_at  TIMESTAMPTZ,
    client_note TEXT
);
CREATE INDEX IF NOT EXISTS auth_sessions_user_idx ON auth_sessions(user_id);

-- ───────────────────────────────────────────────────────────────────────────
-- 4. 核心：以權杖建立工作階段上下文
--
-- SECURITY DEFINER 讓這個函式以擁有者權限執行，
-- 因此應用程式角色本身**不需要、也不應該有**直接寫 app.tenant_id 的能力。
-- 應用程式唯一能做的就是「出示權杖，請求進入某個委任案」，
-- 由這個函式決定准或不准。
-- ───────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION begin_session(p_token TEXT, p_tenant TEXT)
RETURNS TABLE(user_id TEXT, role TEXT, access_level TEXT) AS $$
DECLARE
    v_hash TEXT;
    v_user TEXT;
    v_role TEXT;
    v_level TEXT;
BEGIN
    v_hash := encode(digest(p_token, 'sha256'), 'hex');

    -- ① 認證：權杖有效嗎
    SELECT s.user_id INTO v_user
    FROM auth_sessions s
    WHERE s.token_hash = v_hash
      AND s.revoked_at IS NULL
      AND s.expires_at > NOW();
    IF v_user IS NULL THEN
        RAISE EXCEPTION 'AUTH_FAILED: 權杖無效、已撤銷或已過期';
    END IF;

    -- ② 主體狀態：帳號還能用嗎（離職／停權要能即時生效）
    SELECT p.role INTO v_role FROM principals p
    WHERE p.user_id = v_user AND p.status = 'active';
    IF v_role IS NULL THEN
        RAISE EXCEPTION 'AUTH_FAILED: 帳號已停用';
    END IF;

    -- ③ 授權：這個人被授權存取這個委任案嗎
    --    SHARED 是公開知識庫，所有作用中的帳號都可讀。
    IF p_tenant = 'SHARED' THEN
        v_level := 'read';
    ELSE
        SELECT a.access_level INTO v_level
        FROM engagement_access a
        WHERE a.user_id = v_user
          AND a.tenant_id = p_tenant
          AND a.revoked_at IS NULL
          AND (a.expires_at IS NULL OR a.expires_at > NOW());
        IF v_level IS NULL THEN
            -- 刻意不區分「委任案不存在」與「你沒有權限」：
            -- 區分開來會變成一個可以列舉客戶名單的側信道。
            RAISE EXCEPTION 'ACCESS_DENIED: 無此委任案的存取權';
        END IF;
    END IF;

    -- ④ 通過才設定 RLS 依賴的上下文
    PERFORM set_config('app.tenant_id', p_tenant, false);
    PERFORM set_config('app.role', v_role, false);
    PERFORM set_config('app.actor', v_user, false);

    RETURN QUERY SELECT v_user, v_role, v_level;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ───────────────────────────────────────────────────────────────────────────
-- 5. 權限：應用程式角色只能呼叫函式，不能直接改這幾張表
--
-- 這是整個設計的關鍵 —— 若應用程式能 INSERT engagement_access，
-- 它就能自己授權自己，整條鏈就白做了。
-- ───────────────────────────────────────────────────────────────────────────
REVOKE ALL ON principals, engagement_access, auth_sessions FROM PUBLIC;
GRANT SELECT ON principals, engagement_access, auth_sessions TO flowmind_app;
GRANT EXECUTE ON FUNCTION begin_session(TEXT, TEXT) TO flowmind_app;

-- ───────────────────────────────────────────────────────────────────────────
-- 6. 開發用種子資料
--    密碼雜湊為 sha256，僅供離線 demo；正式環境走 SSO 且不使用本地密碼。
-- ───────────────────────────────────────────────────────────────────────────
INSERT INTO principals (user_id, display_name, role, password_hash) VALUES
    ('alice', '王小美（企金授信 AO）', 'analyst',
     encode(digest('alice-dev-pw', 'sha256'), 'hex')),
    ('bob',   '陳大文（企金授信 AO）', 'analyst',
     encode(digest('bob-dev-pw', 'sha256'), 'hex')),
    ('carol', '林主管（授信覆核）',     'reviewer',
     encode(digest('carol-dev-pw', 'sha256'), 'hex')),
    ('sysadm','系統管理員',            'admin',
     encode(digest('admin-dev-pw', 'sha256'), 'hex'))
ON CONFLICT (user_id) DO NOTHING;
