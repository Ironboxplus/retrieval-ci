# cf-oidc — OIDC Provider on Cloudflare Workers + D1

> This is the identity-provider subsystem. The end-to-end public trust boundary is documented in [../overview.md](../overview.md); gateway authorization and billing remain server-side on BWH.

A minimal OpenID Connect provider. Users authenticate with an **access key (CDK)**;
the Worker issues an **RS256 id_token** via the **Authorization Code + PKCE** flow.
Deployed as the Cloudflare Worker script **`116`** at `https://116.ironbox.workers.dev`.

## Endpoints

| Path | Purpose |
|------|---------|
| `/.well-known/openid-configuration` | OIDC discovery |
| `/keys` | JWKS (public key for verifying id_tokens) |
| `/authorize` | GET = login form; POST = validate CDK, issue auth code |
| `/token` | exchange auth code (+ PKCE verifier) for an id_token |
| `/logout` | RP-initiated logout redirect |
| `/openapi.json`, `/docs` | API spec + Swagger UI |

### id_token claims (contract — do not break)

Relying parties (e.g. the retrival gateway) depend on these:

- `sub` = `cdk:<cdk>`
- `is_admin` (bool), `roles`, `groups` (`["admin","user"]` or `["user"]`)
- `cdk`, `client_id`, `preferred_username`
- standard: `iss`, `aud`, `iat`, `nbf`, `exp` (1h), `jti`, `nonce` (when supplied)

## Develop & deploy (Wrangler)

```bash
npm install              # gets wrangler
npm test                 # pure-function unit tests (node --test, no network)

# one-time: bind D1 + set signing keys
#   edit wrangler.toml -> database_id   (wrangler d1 list)
wrangler secret put PRIVATE_JWK         # RS256 private JWK (JSON, must include "kid")
wrangler secret put PUBLIC_JWK          # matching public JWK (served at /keys)
wrangler d1 execute oidc --file=schema.sql   # create tables (IF NOT EXISTS, safe)

npm run dev              # local (wrangler dev)
npm run deploy           # publish to script "116"
npm run tail             # live logs
```

Adopting this repo for the already-deployed script:
`npx create-cloudflare@latest 116 --existing-script 116` (pulls existing bindings).

## Security configuration

All knobs are in `wrangler.toml` `[vars]`. **Empty/unset = legacy behavior**, so an
existing deployment keeps working; fill them in to harden.

| Var | Effect |
|-----|--------|
| `ALLOWED_REDIRECT_URIS` | Space/comma list. `redirect_uri` (and logout redirect) must match an entry **exactly or by origin**. Empty ⇒ any valid `https` URI (loopback `http` allowed for dev). **Closes open-redirect / auth-code interception.** |
| `ALLOWED_CLIENT_IDS` | Space/comma `client_id` allowlist. Empty ⇒ any client. |
| `ALLOW_PLAIN_PKCE` | `"true"` re-enables the weak `plain` PKCE method. Default ⇒ **S256-only**. |

Always-on hardening (no config needed):

- **PKCE required** on every authorization; `code_verifier` enforced at `/token`.
- **Authorization codes are single-use** — atomically consumed (`DELETE ... changes`) so a replayed code is rejected.
- **`Cache-Control: no-store`** on token/redirect responses (OIDC compliance — no token caching).
- **Login form is hardened**: `X-Frame-Options: DENY` + `frame-ancestors 'none'` (anti-clickjacking of the access-key field), `nosniff`, `Referrer-Policy: no-referrer` (prevents `?code=` leaking via Referer), strict CSP, password-type input with `autocomplete="off"`.
- **`redirect_uri` rules**: absolute `http(s)` only, `https` (or loopback `http`) required, **no fragment** (RFC 6749 §3.1.2).

### Recommended production settings

```toml
ALLOWED_REDIRECT_URIS = "https://retrieval.rnarket.com"
ALLOWED_CLIENT_IDS = "retrival-web"
ALLOW_PLAIN_PKCE = "false"
```

> Note: secrets (`PRIVATE_JWK` / `PUBLIC_JWK`) are **never** committed — they live in
> Wrangler secrets. `.dev.vars` (gitignored) can hold them for `wrangler dev`.

## Layout

```
cf-oidc/
├── main.js            # the Worker (single file; deployed as script "116")
├── wrangler.toml      # config + security vars + D1 binding
├── schema.sql         # D1 tables (cdks, auth_codes) — IF NOT EXISTS, safe to run
├── package.json       # wrangler + test scripts
└── test/worker.test.js  # pure-function unit tests
```
