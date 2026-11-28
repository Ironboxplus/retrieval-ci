// ============================================================================
// OIDC Provider on Cloudflare Workers + D1
//
// Authorization Code + PKCE flow, CDK ("access key") authentication, RS256
// id_tokens. The token CONTRACT is load-bearing for downstream verifiers
// (retrival gateway/auth/oidc.py): sub = `cdk:<cdk>`, plus is_admin / roles /
// groups / cdk / client_id claims. Do NOT change claim names or the endpoint
// paths without updating every relying party.
//
// Security hardening (env-gated, backward compatible — empty/unset = legacy):
//   ALLOWED_REDIRECT_URIS   space/comma list; redirect_uri must match (exact or
//                           by origin). Empty => any valid https URI (loopback
//                           http allowed for dev). Blocks open-redirect / code
//                           interception.
//   ALLOWED_CLIENT_IDS      space/comma list; empty => any client_id.
//   ALLOW_PLAIN_PKCE        "true" re-enables the weak `plain` PKCE method.
//                           Default (unset) => S256-only.
// Plus: anti-clickjacking + CSP on the login form, no-store on token responses,
// atomic single-use of authorization codes.
// ============================================================================

const MAX_CDK_LEN = 512;
const AUTH_CODE_TTL_MS = 300000; // 5 min
const ID_TOKEN_TTL_S = 3600;     // 1 h

// ----------------------------------------------------------------------------
// 工具函数
// ----------------------------------------------------------------------------
function escapeHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function base64UrlEncode(str) {
  const base64 = btoa(new TextEncoder().encode(str).reduce((data, byte) => data + String.fromCharCode(byte), ''));
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function bufferToBase64Url(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function sha256Base64Url(input) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
  return bufferToBase64Url(digest);
}

async function createRS256JWT(payload, privateJwkRaw) {
  let privateJwk = typeof privateJwkRaw === 'string' ? JSON.parse(privateJwkRaw) : privateJwkRaw;
  const header = { alg: "RS256", typ: "JWT", kid: privateJwk.kid };
  const encodedHeader = base64UrlEncode(JSON.stringify(header));
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const dataToSign = `${encodedHeader}.${encodedPayload}`;

  const key = await crypto.subtle.importKey(
    "jwk", privateJwk, { name: "RSASSA-PKCS1-v1_5", hash: { name: "SHA-256" } }, false, ["sign"]
  );
  const signatureBuffer = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5", key, new TextEncoder().encode(dataToSign)
  );
  return `${dataToSign}.${bufferToBase64Url(signatureBuffer)}`;
}

// ----------------------------------------------------------------------------
// 安全工具：allowlist 解析 + redirect_uri 校验 + 响应头
// ----------------------------------------------------------------------------
function parseAllowlist(raw) {
  if (!raw) return [];
  return String(raw).split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
}

// redirect_uri must be a valid absolute http(s) URI with no fragment (RFC 6749
// §3.1.2). https everywhere; http only for loopback (local dev). When an
// allowlist is configured, the URI must match exactly or share an allowlisted
// origin — this is what stops an attacker pointing the auth code at their host.
function isAllowedRedirectUri(uri, allowlist) {
  if (!uri) return false;
  let parsed;
  try {
    parsed = new URL(uri);
  } catch {
    return false;
  }
  if (parsed.hash) return false;
  const isHttps = parsed.protocol === 'https:';
  const isLoopback = parsed.protocol === 'http:' &&
    (parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1' || parsed.hostname === '[::1]');
  if (!isHttps && !isLoopback) return false;
  if (allowlist.length === 0) return true; // legacy: any valid https URI
  return allowlist.includes(uri) || allowlist.includes(parsed.origin);
}

function json(obj, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", ...extraHeaders },
  });
}

// Strict headers for the CDK login form: deny framing (clickjacking would let an
// attacker overlay/steal the access key), no MIME sniffing, no referrer leak of
// the eventual ?code=, and a CSP that permits only the page's own inline CSS.
//
// NOTE: we deliberately do NOT set `form-action`. The login form POSTs to /authorize
// on this same origin, but a successful POST 302-redirects to the RP's redirect_uri
// (retrieval/bwh — a different origin). Chrome evaluates form-action across the whole
// submission redirect chain, so ANY value that doesn't enumerate every RP origin
// (including `'self'`) blocks the submit at click time — invisibly to curl/httpx, which
// don't enforce CSP (W122). The redirect_uri is already allowlisted server-side in
// validateAuthorizeParams, so form-action would be redundant; clickjacking stays covered
// by frame-ancestors 'none' + X-Frame-Options: DENY.
const LOGIN_HEADERS = {
  "Content-Type": "text/html;charset=UTF-8",
  "Cache-Control": "no-store",
  "X-Frame-Options": "DENY",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
  "Content-Security-Policy":
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'none'; base-uri 'none'",
};

// Swagger UI pulls assets from jsdelivr, so it needs a looser CSP than the login page.
const DOCS_HEADERS = {
  "Content-Type": "text/html;charset=UTF-8",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Content-Security-Policy":
    "default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'unsafe-inline' https://cdn.jsdelivr.net; img-src data: https://cdn.jsdelivr.net; font-src https://cdn.jsdelivr.net; connect-src 'self'",
};

// ----------------------------------------------------------------------------
// Worker 入口
// ----------------------------------------------------------------------------
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const ISSUER_URL = `${url.protocol}//${url.host}`;

    if (!env.MY_DB) return new Response("未绑定 D1 数据库", { status: 500 });

    // 安全配置（未设置时回退到旧行为，保证不破坏既有客户端）
    const allowedRedirects = parseAllowlist(env.ALLOWED_REDIRECT_URIS);
    const allowedClients = parseAllowlist(env.ALLOWED_CLIENT_IDS);
    const allowPlainPkce = env.ALLOW_PLAIN_PKCE === 'true';
    const supportedPkceMethods = allowPlainPkce ? ["S256", "plain"] : ["S256"];

    // --- OIDC Discovery ---
    if (url.pathname === '/.well-known/openid-configuration') {
      return new Response(JSON.stringify({
        issuer: ISSUER_URL,
        authorization_endpoint: `${ISSUER_URL}/authorize`,
        token_endpoint: `${ISSUER_URL}/token`,
        jwks_uri: `${ISSUER_URL}/keys`,
        end_session_endpoint: `${ISSUER_URL}/logout`,
        response_types_supported: ["code"],
        subject_types_supported: ["public"],
        id_token_signing_alg_values_supported: ["RS256"],
        code_challenge_methods_supported: supportedPkceMethods
      }), {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=3600",
          "Access-Control-Allow-Origin": "*"
        }
      });
    }

    // --- JWKS ---
    if (url.pathname === '/keys') {
      if (!env.PUBLIC_JWK) return new Response("Missing PUBLIC_JWK", { status: 500 });
      let publicJwk;
      try {
        publicJwk = typeof env.PUBLIC_JWK === 'string' ? JSON.parse(env.PUBLIC_JWK) : env.PUBLIC_JWK;
      } catch (e) {
        return new Response("PUBLIC_JWK 解析失败", { status: 500 });
      }
      return new Response(JSON.stringify({ keys: [publicJwk] }), {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": "public, max-age=3600",
          "Access-Control-Allow-Origin": "*"
        }
      });
    }

    // --- /authorize GET ---
    if (url.pathname === '/authorize' && request.method === 'GET') {
      const redirect_uri = url.searchParams.get('redirect_uri');
      const state = url.searchParams.get('state');
      const clientId = url.searchParams.get('client_id');
      const nonce = url.searchParams.get('nonce') || '';
      const codeChallenge = url.searchParams.get('code_challenge');
      const codeChallengeMethod = url.searchParams.get('code_challenge_method') || (allowPlainPkce ? 'plain' : 'S256');

      const err = validateAuthorizeParams({
        redirect_uri, clientId, codeChallenge, codeChallengeMethod,
        allowedRedirects, allowedClients, supportedPkceMethods,
      });
      if (err) return new Response(err, { status: 400 });

      const html = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sign In — OIDC Service</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{min-height:100vh;display:flex;justify-content:center;align-items:center;
  background:#0a0a0f;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  color:#e4e4e7;overflow:hidden}
body::before{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse at 20% 50%,rgba(99,102,241,.15),transparent 60%),
             radial-gradient(ellipse at 80% 20%,rgba(139,92,246,.12),transparent 50%);
  pointer-events:none}
.card{position:relative;width:380px;padding:48px 40px;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);
  border-radius:20px;
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  box-shadow:0 8px 32px rgba(0,0,0,.4)}
.icon{width:48px;height:48px;margin:0 auto 24px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:14px;font-size:22px}
h1{font-size:20px;font-weight:600;letter-spacing:-.3px;margin-bottom:6px;text-align:center}
.subtitle{font-size:13px;color:#71717a;text-align:center;margin-bottom:32px}
label{display:block;font-size:12px;font-weight:500;color:#a1a1aa;margin-bottom:8px;letter-spacing:.4px;text-transform:uppercase}
input[type="password"]{width:100%;padding:12px 16px;
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);
  border-radius:10px;color:#e4e4e7;font-size:15px;
  transition:border-color .2s,box-shadow .2s;outline:none}
input[type="password"]::placeholder{color:#52525b}
input[type="password"]:focus{border-color:rgba(99,102,241,.6);box-shadow:0 0 0 3px rgba(99,102,241,.15)}
button{width:100%;margin-top:24px;padding:12px;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  border:none;border-radius:10px;
  color:#fff;font-size:14px;font-weight:600;letter-spacing:.3px;
  cursor:pointer;transition:opacity .2s,transform .1s}
button:hover{opacity:.9}
button:active{transform:scale(.98)}
.footer{margin-top:28px;text-align:center;font-size:11px;color:#3f3f46}
.footer span{color:#52525b}
</style>
</head>
<body>
  <div class="card">
    <div class="icon">\u{1F6E1}</div>
    <h1>OIDC Service</h1>
    <p class="subtitle">Enter your access key to continue</p>
    <form method="POST">
      <input type="hidden" name="redirect_uri" value="${escapeHtml(redirect_uri)}">
      <input type="hidden" name="state" value="${escapeHtml(state || '')}">
      <input type="hidden" name="client_id" value="${escapeHtml(clientId)}">
      <input type="hidden" name="nonce" value="${escapeHtml(nonce)}">
      <input type="hidden" name="code_challenge" value="${escapeHtml(codeChallenge)}">
      <input type="hidden" name="code_challenge_method" value="${escapeHtml(codeChallengeMethod)}">
      <!-- hidden username anchors the credential to this origin so password managers
           offer to save/autofill the single-secret access key (W122). -->
      <input type="text" name="username" value="access-key" autocomplete="username" hidden readonly>
      <label for="cdk">Access Key</label>
      <input type="password" id="cdk" name="cdk" placeholder="Paste your key here" required spellcheck="false" autocomplete="current-password" autocapitalize="off">
      <button type="submit">Sign In</button>
    </form>
    <div class="footer">Secured by <span>OpenID Connect</span></div>
  </div>
</body>
</html>`;
      return new Response(html, { headers: LOGIN_HEADERS });
    }

    // --- /authorize POST ---
    if (url.pathname === '/authorize' && request.method === 'POST') {
      const formData = await request.formData();
      const redirect_uri = formData.get('redirect_uri');
      const state = formData.get('state');
      const clientId = formData.get('client_id');
      const cdk = (formData.get('cdk') || '').trim();
      const nonce = formData.get('nonce') || '';
      const codeChallenge = formData.get('code_challenge');
      const codeChallengeMethod = formData.get('code_challenge_method') || (allowPlainPkce ? 'plain' : 'S256');

      // Re-validate everything on POST — the GET checks are advisory; an attacker
      // can POST directly, so these params are not trusted just because a form rendered.
      const err = validateAuthorizeParams({
        redirect_uri, clientId, codeChallenge, codeChallengeMethod,
        allowedRedirects, allowedClients, supportedPkceMethods,
      });
      if (err) return new Response(err, { status: 400 });

      if (!cdk) return new Response("CDK 不能为空", { status: 400 });
      if (cdk.length > MAX_CDK_LEN) return new Response("CDK 过长", { status: 400 });

      const cdkRow = await env.MY_DB.prepare("SELECT * FROM cdks WHERE cdk = ?").bind(cdk).first();
      if (!cdkRow) return new Response("CDK 不存在", { status: 403 });
      if (cdkRow.status !== 'valid') return new Response("CDK 已失效", { status: 403 });

      const role = cdkRow.role || 'user';
      const isAdmin = role === 'admin';
      const reusable = cdkRow.reusable === 1;
      const nowMs = Date.now();

      const authCode = crypto.randomUUID();
      const expiresAt = nowMs + AUTH_CODE_TTL_MS;

      const insertStmt = env.MY_DB.prepare(`
        INSERT INTO auth_codes (
          code, client_id, cdk, is_admin, expires_at, redirect_uri, nonce,
          code_challenge, code_challenge_method
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).bind(
        authCode, clientId, cdk, isAdmin ? 1 : 0, expiresAt, redirect_uri, nonce,
        codeChallenge, codeChallengeMethod
      );

      // Admin and reusable CDKs survive; a normal CDK is burned on first use.
      if (isAdmin || reusable) {
        await insertStmt.run();
      } else {
        await env.MY_DB.batch([
          env.MY_DB.prepare("UPDATE cdks SET status = 'disabled', expires_at = ? WHERE cdk = ?").bind(nowMs, cdk),
          insertStmt
        ]);
      }

      const redirectUrl = new URL(redirect_uri);
      redirectUrl.searchParams.set('code', authCode);
      if (state) redirectUrl.searchParams.set('state', state);

      return new Response(null, {
        status: 302,
        headers: { Location: redirectUrl.toString(), "Cache-Control": "no-store" }
      });
    }

    // --- /token POST ---
    if (url.pathname === '/token' && request.method === 'POST') {
      const formData = await request.formData();
      const code = formData.get('code');
      const reqClientId = formData.get('client_id');
      const reqRedirectUri = formData.get('redirect_uri');
      const codeVerifier = formData.get('code_verifier');

      const row = await env.MY_DB.prepare("SELECT * FROM auth_codes WHERE code = ?").bind(code).first();
      if (!row || Date.now() > row.expires_at) {
        return json({ error: "invalid_grant" }, 400, { "Cache-Control": "no-store" });
      }

      if (row.client_id !== reqClientId || row.redirect_uri !== reqRedirectUri) {
        return json({ error: "invalid_grant", error_description: "client or redirect mismatch" },
          400, { "Cache-Control": "no-store" });
      }

      // PKCE — the code is useless without the verifier that hashes to its challenge.
      if (!row.code_challenge) {
        return json({ error: "invalid_grant", error_description: "missing code_challenge" },
          400, { "Cache-Control": "no-store" });
      }
      if (!codeVerifier) {
        return json({ error: "invalid_request", error_description: "missing code_verifier" },
          400, { "Cache-Control": "no-store" });
      }

      let pkceValid = false;
      if (row.code_challenge_method === 'S256') {
        pkceValid = (await sha256Base64Url(codeVerifier)) === row.code_challenge;
      } else if (row.code_challenge_method === 'plain' && allowPlainPkce) {
        pkceValid = codeVerifier === row.code_challenge;
      }
      if (!pkceValid) {
        return json({ error: "invalid_grant", error_description: "PKCE verification failed" },
          400, { "Cache-Control": "no-store" });
      }

      // Atomically consume the code: if another request already deleted it,
      // changes === 0 and we refuse — prevents code replay / double redemption.
      const consumed = await env.MY_DB.prepare("DELETE FROM auth_codes WHERE code = ?").bind(code).run();
      if (!consumed.meta || consumed.meta.changes === 0) {
        return json({ error: "invalid_grant", error_description: "code already used" },
          400, { "Cache-Control": "no-store" });
      }

      const isAdmin = row.is_admin === 1;
      const now = Math.floor(Date.now() / 1000);

      const subject = `cdk:${row.cdk}`;
      const username = row.cdk;
      const userEmail = `${row.cdk}@local.example`;
      const displayName = isAdmin ? `admin:${row.cdk}` : `user:${row.cdk}`;
      const groups = isAdmin ? ["admin", "user"] : ["user"];
      const roles = isAdmin ? ["admin", "user"] : ["user"];

      const payload = {
        // JWT / OIDC 标准字段
        iss: ISSUER_URL,
        sub: subject,
        aud: row.client_id,
        iat: now,
        nbf: now,
        exp: now + ID_TOKEN_TTL_S,
        jti: crypto.randomUUID(),
        typ: "Bearer",

        // OIDC 认证流程相关
        ...(row.nonce ? { nonce: row.nonce } : {}),

        // 标准身份 claims
        name: displayName,
        preferred_username: username,
        email: userEmail,
        email_verified: true,

        // 授权相关 claims
        groups,
        roles,

        // 业务扩展 claims
        cdk: row.cdk,
        client_id: row.client_id,
        is_admin: isAdmin
      };

      if (!env.PRIVATE_JWK) return new Response("Missing PRIVATE_JWK", { status: 500 });

      try {
        const id_token = await createRS256JWT(payload, env.PRIVATE_JWK);
        return json({
          access_token: crypto.randomUUID(),
          token_type: "Bearer",
          expires_in: payload.exp - now,
          id_token
        }, 200, { "Cache-Control": "no-store", "Pragma": "no-cache" });
      } catch (e) {
        return json({ error: "server_error" }, 500, { "Cache-Control": "no-store" });
      }
    }

    if (url.pathname === '/logout') {
      const postLogoutRedirectUri = url.searchParams.get('post_logout_redirect_uri');
      if (!postLogoutRedirectUri) {
        return new Response("缺少 post_logout_redirect_uri", { status: 400 });
      }
      // Same allowlist as redirect_uri — an unvalidated logout redirect is an open redirector.
      if (!isAllowedRedirectUri(postLogoutRedirectUri, allowedRedirects)) {
        return new Response("post_logout_redirect_uri 不被允许", { status: 400 });
      }

      const callbackUrl = new URL(postLogoutRedirectUri);
      const state = url.searchParams.get('state');
      if (state) callbackUrl.searchParams.set('state', state);

      return new Response(null, {
        status: 302,
        headers: { Location: callbackUrl.toString(), "Cache-Control": "no-store" }
      });
    }

    // --- OpenAPI Spec ---
    if (url.pathname === '/openapi.json') {
      const spec = {
        openapi: "3.0.3",
        info: {
          title: "OIDC Service",
          description: "Lightweight OpenID Connect Provider with Access Key authentication, built on Cloudflare Workers + D1.",
          version: "1.1.0"
        },
        servers: [{ url: ISSUER_URL }],
        paths: {
          "/.well-known/openid-configuration": {
            get: {
              summary: "OIDC Discovery",
              description: "Returns the OpenID Connect discovery document.",
              operationId: "getDiscovery",
              tags: ["OIDC"],
              responses: {
                "200": {
                  description: "Discovery document",
                  content: { "application/json": { schema: { "$ref": "#/components/schemas/DiscoveryDocument" } } }
                }
              }
            }
          },
          "/keys": {
            get: {
              summary: "JWKS",
              description: "Returns the JSON Web Key Set used to verify ID tokens.",
              operationId: "getJwks",
              tags: ["OIDC"],
              responses: {
                "200": {
                  description: "JWKS",
                  content: { "application/json": { schema: { "$ref": "#/components/schemas/JWKS" } } }
                }
              }
            }
          },
          "/authorize": {
            get: {
              summary: "Authorization Page",
              description: "Renders the access key login form. Implements OIDC Authorization Code flow with PKCE.",
              operationId: "authorizeGet",
              tags: ["Auth"],
              parameters: [
                { name: "client_id", in: "query", required: true, schema: { type: "string" }, description: "Client identifier" },
                { name: "redirect_uri", in: "query", required: true, schema: { type: "string", format: "uri" }, description: "Callback URI after authorization (must be allowlisted if ALLOWED_REDIRECT_URIS is set)" },
                { name: "code_challenge", in: "query", required: true, schema: { type: "string" }, description: "PKCE code challenge" },
                { name: "code_challenge_method", in: "query", schema: { type: "string", enum: ["S256", "plain"], default: "S256" }, description: "PKCE challenge method (plain only if ALLOW_PLAIN_PKCE=true)" },
                { name: "state", in: "query", schema: { type: "string" }, description: "Opaque state value passed back to redirect_uri" },
                { name: "nonce", in: "query", schema: { type: "string" }, description: "Nonce for ID token replay protection" }
              ],
              responses: {
                "200": { description: "Login HTML page", content: { "text/html": {} } },
                "400": { description: "Missing or invalid parameters" }
              }
            },
            post: {
              summary: "Submit Authorization",
              description: "Validates the access key (CDK), generates an authorization code, and redirects to redirect_uri with the code.",
              operationId: "authorizePost",
              tags: ["Auth"],
              requestBody: {
                required: true,
                content: {
                  "application/x-www-form-urlencoded": {
                    schema: {
                      type: "object",
                      required: ["cdk", "redirect_uri", "client_id", "code_challenge"],
                      properties: {
                        cdk: { type: "string", description: "Access key" },
                        redirect_uri: { type: "string", format: "uri" },
                        client_id: { type: "string" },
                        state: { type: "string" },
                        nonce: { type: "string" },
                        code_challenge: { type: "string" },
                        code_challenge_method: { type: "string", enum: ["S256", "plain"] }
                      }
                    }
                  }
                }
              },
              responses: {
                "302": { description: "Redirect to redirect_uri with authorization code" },
                "400": { description: "Invalid request" },
                "403": { description: "Invalid or expired access key" }
              }
            }
          },
          "/token": {
            post: {
              summary: "Token Exchange",
              description: "Exchanges an authorization code for an ID token. Requires PKCE code_verifier.",
              operationId: "token",
              tags: ["Auth"],
              requestBody: {
                required: true,
                content: {
                  "application/x-www-form-urlencoded": {
                    schema: {
                      type: "object",
                      required: ["code", "client_id", "redirect_uri", "code_verifier"],
                      properties: {
                        code: { type: "string", description: "Authorization code from /authorize" },
                        client_id: { type: "string" },
                        redirect_uri: { type: "string", format: "uri" },
                        code_verifier: { type: "string", description: "PKCE code verifier" }
                      }
                    }
                  }
                }
              },
              responses: {
                "200": {
                  description: "Token response",
                  content: {
                    "application/json": {
                      schema: { "$ref": "#/components/schemas/TokenResponse" }
                    }
                  }
                },
                "400": {
                  description: "Invalid grant or PKCE failure",
                  content: {
                    "application/json": {
                      schema: { "$ref": "#/components/schemas/ErrorResponse" }
                    }
                  }
                }
              }
            }
          },
          "/logout": {
            get: {
              summary: "Logout",
              description: "RP-Initiated logout. Redirects to post_logout_redirect_uri (must be allowlisted if ALLOWED_REDIRECT_URIS is set).",
              operationId: "logout",
              tags: ["Auth"],
              parameters: [
                { name: "post_logout_redirect_uri", in: "query", required: true, schema: { type: "string", format: "uri" }, description: "URI to redirect after logout" },
                { name: "state", in: "query", schema: { type: "string" }, description: "Opaque state value" }
              ],
              responses: {
                "302": { description: "Redirect to post_logout_redirect_uri" },
                "400": { description: "Missing or disallowed post_logout_redirect_uri" }
              }
            }
          }
        },
        components: {
          schemas: {
            DiscoveryDocument: {
              type: "object",
              properties: {
                issuer: { type: "string", format: "uri" },
                authorization_endpoint: { type: "string", format: "uri" },
                token_endpoint: { type: "string", format: "uri" },
                jwks_uri: { type: "string", format: "uri" },
                end_session_endpoint: { type: "string", format: "uri" },
                response_types_supported: { type: "array", items: { type: "string" } },
                subject_types_supported: { type: "array", items: { type: "string" } },
                id_token_signing_alg_values_supported: { type: "array", items: { type: "string" } },
                code_challenge_methods_supported: { type: "array", items: { type: "string" } }
              }
            },
            JWKS: {
              type: "object",
              properties: {
                keys: { type: "array", items: { type: "object" } }
              }
            },
            TokenResponse: {
              type: "object",
              properties: {
                access_token: { type: "string" },
                token_type: { type: "string", example: "Bearer" },
                expires_in: { type: "integer", example: 3600 },
                id_token: { type: "string", description: "RS256-signed JWT" }
              }
            },
            ErrorResponse: {
              type: "object",
              properties: {
                error: { type: "string", example: "invalid_grant" },
                error_description: { type: "string" }
              }
            }
          }
        }
      };
      return new Response(JSON.stringify(spec, null, 2), {
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
      });
    }

    // --- Swagger UI ---
    if (url.pathname === '/docs') {
      const docsHtml = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Docs — OIDC Service</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
<style>
body { margin: 0 }
.swagger-ui .topbar { display: none }
</style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({
  url: '${ISSUER_URL}/openapi.json',
  dom_id: '#swagger-ui',
  deepLinking: true,
  presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
  layout: 'BaseLayout'
});
</script>
</body>
</html>`;
      return new Response(docsHtml, { headers: DOCS_HEADERS });
    }

    return new Response("Not Found", { status: 404 });
  }
};

// Shared validation for /authorize GET + POST. Returns an error string (caller
// replies 400) or null when OK.
function validateAuthorizeParams({ redirect_uri, clientId, codeChallenge,
  codeChallengeMethod, allowedRedirects, allowedClients, supportedPkceMethods }) {
  if (!redirect_uri || !clientId) return "缺少 redirect_uri 或 client_id";
  if (!isAllowedRedirectUri(redirect_uri, allowedRedirects)) return "redirect_uri 不被允许";
  if (allowedClients.length > 0 && !allowedClients.includes(clientId)) return "client_id 不被允许";
  if (!codeChallenge) return "缺少 code_challenge";
  if (!supportedPkceMethods.includes(codeChallengeMethod)) return "不支持的 code_challenge_method";
  return null;
}

export const __test__ = {
  escapeHtml,
  parseAllowlist,
  isAllowedRedirectUri,
  validateAuthorizeParams,
  sha256Base64Url,
  base64UrlEncode,
  createRS256JWT,
};
