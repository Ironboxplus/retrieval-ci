-- D1 schema for the OIDC Worker. Documents the tables the Worker reads/writes.
-- Uses IF NOT EXISTS so it is SAFE to run against the live DB — it will not
-- touch or migrate existing tables/data.
--
--   wrangler d1 execute oidc --file=schema.sql            # remote
--   wrangler d1 execute oidc --local --file=schema.sql    # local dev DB

-- Access keys ("CDK"). A normal key is single-use (status flipped to 'disabled'
-- on first authorize); admin/reusable keys persist.
CREATE TABLE IF NOT EXISTS cdks (
  cdk         TEXT PRIMARY KEY,
  status      TEXT    NOT NULL DEFAULT 'valid',   -- 'valid' | 'disabled'
  role        TEXT    NOT NULL DEFAULT 'user',    -- 'user' | 'admin'
  reusable    INTEGER NOT NULL DEFAULT 0,         -- 1 = may be used repeatedly
  expires_at  INTEGER                             -- ms epoch; set when disabled
);

-- Short-lived authorization codes (PKCE). Consumed (deleted) on /token exchange.
CREATE TABLE IF NOT EXISTS auth_codes (
  code                  TEXT PRIMARY KEY,
  client_id             TEXT    NOT NULL,
  cdk                   TEXT    NOT NULL,
  is_admin              INTEGER NOT NULL DEFAULT 0,
  expires_at            INTEGER NOT NULL,          -- ms epoch (5 min TTL)
  redirect_uri          TEXT    NOT NULL,
  nonce                 TEXT,
  code_challenge        TEXT    NOT NULL,
  code_challenge_method TEXT    NOT NULL DEFAULT 'S256'
);

-- Helps prune expired codes.
CREATE INDEX IF NOT EXISTS idx_auth_codes_expires_at ON auth_codes (expires_at);

-- Example seeds (uncomment to create keys):
-- INSERT INTO cdks (cdk, status, role, reusable) VALUES ('your-admin-key', 'valid', 'admin', 1);
-- INSERT INTO cdks (cdk, status, role, reusable) VALUES ('one-time-invite', 'valid', 'user', 0);
