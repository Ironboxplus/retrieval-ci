// Pure-function tests for the OIDC Worker — no D1, no network.
//   npm test     (node --test test/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';

if (!globalThis.crypto) globalThis.crypto = webcrypto;

const { __test__ } = await import('../main.js');
const { escapeHtml, parseAllowlist, isAllowedRedirectUri, validateAuthorizeParams,
        sha256Base64Url, createRS256JWT } = __test__;

function b64urlToBytes(s) {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(s.length / 4) * 4, '=');
  return Uint8Array.from(Buffer.from(b64, 'base64'));
}
function decodeSeg(seg) {
  return JSON.parse(Buffer.from(seg.replace(/-/g, '+').replace(/_/g, '/'), 'base64').toString('utf8'));
}

// --- escapeHtml --------------------------------------------------------------
test('escapeHtml neutralizes markup', () => {
  assert.equal(escapeHtml(`<script>"&'`), '&lt;script&gt;&quot;&amp;&#39;');
  assert.equal(escapeHtml(null), '');
});

// --- parseAllowlist ----------------------------------------------------------
test('parseAllowlist splits on space/comma/newline and trims', () => {
  assert.deepEqual(parseAllowlist(''), []);
  assert.deepEqual(parseAllowlist(undefined), []);
  assert.deepEqual(parseAllowlist('a b,c\n d '), ['a', 'b', 'c', 'd']);
});

// --- isAllowedRedirectUri ----------------------------------------------------
test('redirect: empty allowlist accepts any https, rejects non-http(s)', () => {
  assert.equal(isAllowedRedirectUri('https://app.example/cb', []), true);
  assert.equal(isAllowedRedirectUri('http://localhost:3000/cb', []), true);
  assert.equal(isAllowedRedirectUri('http://127.0.0.1/cb', []), true);
  assert.equal(isAllowedRedirectUri('http://evil.example/cb', []), false); // non-loopback http
  assert.equal(isAllowedRedirectUri('javascript:alert(1)', []), false);
  assert.equal(isAllowedRedirectUri('https://app.example/cb#frag', []), false); // fragment banned
  assert.equal(isAllowedRedirectUri('not a url', []), false);
  assert.equal(isAllowedRedirectUri('', []), false);
});

test('redirect: non-empty allowlist matches exact URI or origin', () => {
  const allow = ['https://retrieval.rnarket.com', 'https://other.example/exact/cb'];
  assert.equal(isAllowedRedirectUri('https://retrieval.rnarket.com/any/path', allow), true); // origin match
  assert.equal(isAllowedRedirectUri('https://other.example/exact/cb', allow), true);          // exact match
  assert.equal(isAllowedRedirectUri('https://other.example/different', allow), false);         // path not allowed
  assert.equal(isAllowedRedirectUri('https://attacker.example/cb', allow), false);
});

// --- validateAuthorizeParams -------------------------------------------------
test('validateAuthorizeParams enforces required + allowlists + PKCE method', () => {
  const base = {
    redirect_uri: 'https://app.example/cb',
    clientId: 'web',
    codeChallenge: 'abc',
    codeChallengeMethod: 'S256',
    allowedRedirects: [],
    allowedClients: [],
    supportedPkceMethods: ['S256'],
  };
  assert.equal(validateAuthorizeParams(base), null); // happy path
  assert.match(validateAuthorizeParams({ ...base, redirect_uri: '' }), /redirect_uri/);
  assert.match(validateAuthorizeParams({ ...base, clientId: '' }), /client_id/);
  assert.match(validateAuthorizeParams({ ...base, redirect_uri: 'http://evil.example/cb' }), /redirect_uri/);
  assert.match(validateAuthorizeParams({ ...base, codeChallenge: '' }), /code_challenge/);
  assert.match(validateAuthorizeParams({ ...base, codeChallengeMethod: 'plain' }), /code_challenge_method/);
  assert.equal(validateAuthorizeParams({ ...base, allowedClients: ['web'] }), null);
  assert.match(validateAuthorizeParams({ ...base, allowedClients: ['only-other'] }), /client_id/);
  // plain allowed only when listed as supported
  assert.equal(validateAuthorizeParams({ ...base, codeChallengeMethod: 'plain', supportedPkceMethods: ['S256', 'plain'] }), null);
});

// --- PKCE S256 (RFC 7636 test vector) ---------------------------------------
test('sha256Base64Url matches the RFC 7636 worked example', async () => {
  const verifier = 'dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk';
  assert.equal(await sha256Base64Url(verifier), 'E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM');
});

// --- RS256 JWT roundtrip -----------------------------------------------------
test('createRS256JWT produces a verifiable RS256 token with the kid header', async () => {
  const { publicKey, privateKey } = await crypto.subtle.generateKey(
    { name: 'RSASSA-PKCS1-v1_5', modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: 'SHA-256' },
    true, ['sign', 'verify']);
  const privJwk = await crypto.subtle.exportKey('jwk', privateKey);
  privJwk.kid = 'test-kid';

  const token = await createRS256JWT({ sub: 'cdk:abc', is_admin: true }, privJwk);
  const [h, p, sig] = token.split('.');
  assert.ok(h && p && sig);

  const header = decodeSeg(h);
  assert.equal(header.alg, 'RS256');
  assert.equal(header.kid, 'test-kid');
  assert.equal(decodeSeg(p).sub, 'cdk:abc');

  const verifyKey = await crypto.subtle.importKey(
    'jwk', await crypto.subtle.exportKey('jwk', publicKey),
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify']);
  const ok = await crypto.subtle.verify(
    'RSASSA-PKCS1-v1_5', verifyKey, b64urlToBytes(sig), new TextEncoder().encode(`${h}.${p}`));
  assert.equal(ok, true);
});
