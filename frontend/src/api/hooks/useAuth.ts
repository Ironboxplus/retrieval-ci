import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import api from '../client';
import { PUBLIC_MODE } from '../../constants';
import type { Me, Identity, ApiKeyInfo, ApiKeyCreated, TopupResult } from '../../types';

/**
 * Auth/account hooks for the paid public tier (BWH gateway).
 *
 * `/auth/*` lives at the SITE ROOT, not under `/api` — the shared `api` client
 * is `baseURL=/api`, so auth calls use this root-relative instance. Everything
 * is same-origin (retrieval.rnarket.com → traefik → gateway), so the gateway
 * session cookie rides along automatically; no `withCredentials` needed.
 *
 * All queries are `enabled: PUBLIC_MODE` — the internal 4090 build has no
 * gateway and no `/auth`, so they must never fire there.
 */
const authApi = axios.create({ baseURL: '/', headers: { 'Content-Type': 'application/json' } });

export const AUTH_ME_KEY = ['auth', 'me'] as const;
const KEYS_KEY = ['auth', 'keys'] as const;

export function canQueryAuthenticatedApi(
  publicMode: boolean,
  authenticationResolved: boolean,
  user: unknown,
): boolean {
  return !publicMode || (authenticationResolved && user != null);
}

/** Begin CDK (access-key) login: full-page redirect to the gateway → cf-oidc Worker. */
export function login(): void {
  window.location.assign('/auth/login');
}

/** Begin linux.do Connect login (full-page redirect → connect.linux.do). */
export function loginLinuxDo(): void {
  window.location.assign('/api/linuxdooid/login');
}

/** Begin Google login (full-page redirect → accounts.google.com). */
export function loginGoogle(): void {
  window.location.assign('/api/google/login');
}

/** Bind another login method to the CURRENT account (must be logged in). */
export function bindIdentity(provider: 'linuxdo' | 'google'): void {
  const seg = provider === 'linuxdo' ? 'linuxdooid' : 'google';
  window.location.assign(`/api/${seg}/link`);
}

/**
 * Current user, or `null` when anonymous. A 401 (not logged in) resolves to
 * `null` — it's an expected state, not an error — so consumers can branch on
 * `data == null` without try/catch.
 */
export function useMe() {
  return useQuery<Me | null>({
    queryKey: AUTH_ME_KEY,
    enabled: PUBLIC_MODE,
    staleTime: 30_000,
    retry: false,
    queryFn: async () => {
      try {
        const r = await authApi.get<Me>('/auth/me');
        return r.data;
      } catch (e) {
        if (axios.isAxiosError(e) && e.response?.status === 401) return null;
        throw e;
      }
    },
  });
}

/** Gate public API queries that require a gateway-authenticated user. */
export function useAuthenticatedApiEnabled(): boolean {
  const { data: user, isFetched } = useMe();
  return canQueryAuthenticatedApi(PUBLIC_MODE, isFetched, user);
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => authApi.post('/auth/logout').then((r) => r.data),
    onSuccess: () => {
      qc.setQueryData(AUTH_ME_KEY, null);
      qc.invalidateQueries({ queryKey: AUTH_ME_KEY });
    },
  });
}

/** First-login: claim a username. Gateway sets the session on success. */
export function useRegister() {
  const qc = useQueryClient();
  return useMutation<{ id: number; username: string; tier: string }, unknown, string>({
    mutationFn: (username) =>
      authApi.post('/auth/register', { username }).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: AUTH_ME_KEY }),
  });
}

export interface AuthProviders { cdk: boolean; linuxdo: boolean; google: boolean }

/** Which login methods the gateway has enabled. Public (works while anonymous). */
export function useAuthProviders() {
  return useQuery<AuthProviders>({
    queryKey: ['auth', 'providers'],
    enabled: PUBLIC_MODE,
    staleTime: 5 * 60_000,
    queryFn: () => authApi.get<AuthProviders>('/auth/providers').then((r) => r.data),
  });
}

const IDENTITIES_KEY = ['auth', 'identities'] as const;

/** Login methods bound to the current account (账号绑定). */
export function useIdentities() {
  return useQuery<Identity[]>({
    queryKey: IDENTITIES_KEY,
    enabled: PUBLIC_MODE,
    queryFn: () => authApi.get<Identity[]>('/auth/identities').then((r) => r.data),
  });
}

/** Unlink a bound login method. The gateway refuses to remove the last one. */
export function useUnlinkIdentity() {
  const qc = useQueryClient();
  return useMutation<unknown, unknown, number>({
    mutationFn: (id) => authApi.delete(`/auth/identities/${id}`).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: IDENTITIES_KEY });
      qc.invalidateQueries({ queryKey: AUTH_ME_KEY });
    },
  });
}

export function useApiKeys() {
  return useQuery<ApiKeyInfo[]>({
    queryKey: KEYS_KEY,
    enabled: PUBLIC_MODE,
    queryFn: () => authApi.get<ApiKeyInfo[]>('/auth/keys').then((r) => r.data),
  });
}

export function useCreateKey() {
  const qc = useQueryClient();
  return useMutation<ApiKeyCreated, unknown, void>({
    mutationFn: () => authApi.post<ApiKeyCreated>('/auth/keys').then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS_KEY }),
  });
}

export function useRevokeKey() {
  const qc = useQueryClient();
  return useMutation<unknown, unknown, number>({
    mutationFn: (id) => authApi.delete(`/auth/keys/${id}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: KEYS_KEY }),
  });
}

/** Topup lives under `/api` → shared client. Returns the provider pay_url. */
export function useTopup() {
  const qc = useQueryClient();
  return useMutation<TopupResult, unknown, { credits: number; price: string }>({
    mutationFn: (body) => api.post<TopupResult>('/topup', body).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: AUTH_ME_KEY }),
  });
}
