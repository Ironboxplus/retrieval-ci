/**
 * Pure logic for /search?mode=code (F2-c: code search merged into the main search
 * page). Kept JSX-free and framework-free so URL round-tripping, request assembly,
 * and the Gantt/distribution visualizations are unit-testable without mounting
 * antd/framer-motion.
 */

export type CodeRetrievalMode = 'hybrid' | 'dense' | 'keyword';

export interface CodeUIState {
  query: string;
  /** Federated-search `projects[]` scope strings (absolute indexed-dir paths).
   * Empty = every scope the caller owns. */
  projects: string[];
  language: string;
  pathGlob: string;
  maxPerFile: number | null;
  topK: number;
  mode: CodeRetrievalMode;
  rerank: boolean;
  /** Admin-only. Ignored server-side (422) unless true AND the caller is admin —
   * the UI itself only shows the toggle to admins (ConsoleGate-equivalent). */
  includePublic: boolean;
  sourceIds: string[];
  domains: string[];
}

export const DEFAULT_CODE_UI: CodeUIState = {
  query: '', projects: [], language: '', pathGlob: '', maxPerFile: null,
  topK: 10, mode: 'hybrid', rerank: false, includePublic: false,
  sourceIds: [], domains: [],
};

function splitCsv(v: string | null): string[] {
  return v ? v.split(',').filter(Boolean) : [];
}

/** Reads the code-mode URL params. Shares `q` with paper mode (one query box,
 * one `?q=`); every other param is `c`-prefixed so switching `?mode=` never
 * collides with paper mode's own `venues`/`years`/etc. */
export function readCodeUI(sp: URLSearchParams): CodeUIState {
  const maxPerFileRaw = sp.get('cmax');
  const maxPerFile = maxPerFileRaw ? Number(maxPerFileRaw) : null;
  const topKRaw = sp.get('ctopk');
  const topK = topKRaw && Number.isFinite(Number(topKRaw)) ? Number(topKRaw) : DEFAULT_CODE_UI.topK;
  const modeRaw = sp.get('cmode');
  const mode: CodeRetrievalMode =
    modeRaw === 'dense' || modeRaw === 'keyword' ? modeRaw : 'hybrid';
  return {
    query: sp.get('q') ?? '',
    projects: splitCsv(sp.get('cproj')),
    language: sp.get('clang') ?? '',
    pathGlob: sp.get('cpath') ?? '',
    maxPerFile: maxPerFile != null && Number.isFinite(maxPerFile) ? maxPerFile : null,
    topK,
    mode,
    rerank: sp.get('crerank') === '1',
    includePublic: sp.get('cpublic') === '1',
    sourceIds: splitCsv(sp.get('csrc')),
    domains: splitCsv(sp.get('cdom')),
  };
}

export function writeCodeUI(ui: CodeUIState): URLSearchParams {
  const sp = new URLSearchParams();
  if (ui.query) sp.set('q', ui.query);
  if (ui.projects.length) sp.set('cproj', ui.projects.join(','));
  if (ui.language) sp.set('clang', ui.language);
  if (ui.pathGlob) sp.set('cpath', ui.pathGlob);
  if (ui.maxPerFile != null) sp.set('cmax', String(ui.maxPerFile));
  if (ui.topK !== DEFAULT_CODE_UI.topK) sp.set('ctopk', String(ui.topK));
  if (ui.mode !== DEFAULT_CODE_UI.mode) sp.set('cmode', ui.mode);
  if (ui.rerank) sp.set('crerank', '1');
  if (ui.includePublic) sp.set('cpublic', '1');
  if (ui.sourceIds.length) sp.set('csrc', ui.sourceIds.join(','));
  if (ui.domains.length) sp.set('cdom', ui.domains.join(','));
  return sp;
}

export interface FederatedCodeSearchRequestBody {
  query: string;
  top_k: number;
  mode: CodeRetrievalMode;
  rerank: boolean;
  language: string | null;
  include_public: boolean;
  projects: string[];
  path_glob: string | null;
  max_per_file: number | null;
  source_ids: string[];
  domains: string[];
}

/** Maps UI state to the federated-search request body. Mirrors the backend's own
 * fail-closed rule (FederatedCodeSearchRequest._public_only_filters_require_include_public):
 * source_ids/domains are only sent when includePublic is true, so an admin who
 * picked public sources and then flips the toggle off doesn't send a request the
 * backend would 422. */
export function buildFederatedSearchRequest(ui: CodeUIState): FederatedCodeSearchRequestBody {
  return {
    query: ui.query,
    top_k: ui.topK,
    mode: ui.mode,
    rerank: ui.rerank,
    language: ui.language.trim() || null,
    include_public: ui.includePublic,
    projects: ui.projects,
    path_glob: ui.pathGlob.trim() || null,
    max_per_file: ui.maxPerFile,
    source_ids: ui.includePublic ? ui.sourceIds : [],
    domains: ui.includePublic ? ui.domains : [],
  };
}

/** Backend hard cap: FederatedCodeSearchRequest.projects has max_length=50
 * (app/services/code_kb/schemas.py). An owner can have far more indexed
 * projects than that (round-3 review: 243 on the account this was verified
 * against) — round 2's "everything shown checked when the selection is
 * empty" broke the very first uncheck, submitting 242 projects and getting
 * a 422 back. Selection is real again (empty = literally nothing ticked,
 * which still means "search everything owned" server-side); what changed is
 * how "search everything" is SHOWN — via bar color/opacity in
 * CodeScopePanel, not phantom checkmarks — and a cap so a project or host
 * checkbox that would push the total over 50 is disabled before the click,
 * not after a 422. */
export const MAX_SELECTED_PROJECTS = 50;

/** Add/remove one project from the selection — the pure form of
 * CodeSearchView's/CodeDistributionPanel's click-to-toggle-a-project
 * handler. A no-op past the cap (defense in depth; the UI is expected to
 * disable the control before this is ever called at the cap). */
export function toggleProjectSelection(
  current: string[], project: string, cap = MAX_SELECTED_PROJECTS,
): string[] {
  if (current.includes(project)) return current.filter((p) => p !== project);
  if (current.length >= cap) return current;
  return [...current, project];
}

/** Whether an individual (unselected) project checkbox should be disabled —
 * true once the cap is already reached by OTHER selections. An already-
 * selected project is never disabled by this (unticking always works, even
 * exactly at the cap). */
export function isProjectCapped(
  selectedCount: number, alreadySelected: boolean, cap = MAX_SELECTED_PROJECTS,
): boolean {
  return !alreadySelected && selectedCount >= cap;
}

/** Whether a host's own checkbox should be disabled: its subtree alone
 * exceeds the cap (ticking it could never produce a valid request, however
 * few other projects are selected elsewhere), or ticking its not-yet-
 * selected projects would push the running total over the cap. A host that
 * fits under the cap given the CURRENT selection stays enabled — the same
 * host can flip to disabled later purely because selections elsewhere grew,
 * which is intentional (the cap is global, not per-host). */
export function isHostCapped(
  hostProjectCount: number, selectedCount: number, hostSelectedCount: number, cap = MAX_SELECTED_PROJECTS,
): boolean {
  if (hostProjectCount > cap) return true;
  const newlySelected = hostProjectCount - hostSelectedCount;
  return selectedCount + newlySelected > cap;
}

export interface CodeSearchTimingsMsLike {
  embed?: number | null;
  retrieve?: number | null;
  fuse?: number | null;
  rerank?: number | null;
  total?: number | null;
}

export interface TimingSegment {
  key: 'embed' | 'retrieve' | 'fuse' | 'rerank';
  label: string;
  ms: number;
  startPct: number;
  widthPct: number;
}

const TIMING_STAGE_LABEL: Record<TimingSegment['key'], string> = {
  embed: '编码', retrieve: '检索', fuse: '融合', rerank: '重排',
};

/** Gantt-bar segments: each present, positive phase placed end-to-end along a
 * timeline whose full width is `total`. A phase that never ran (null) or the
 * backend's own leftover overhead (concurrent legs, network, serialization)
 * simply doesn't get a segment — the bar is never force-stretched to 100%. */
export function timingsSegments(t: CodeSearchTimingsMsLike): TimingSegment[] {
  const total = t.total ?? 0;
  if (!(total > 0)) return [];
  const order: Array<{ key: TimingSegment['key']; ms: number | null | undefined }> = [
    { key: 'embed', ms: t.embed },
    { key: 'retrieve', ms: t.retrieve },
    { key: 'fuse', ms: t.fuse },
    { key: 'rerank', ms: t.rerank },
  ];
  const segments: TimingSegment[] = [];
  let cursor = 0;
  for (const stage of order) {
    if (stage.ms == null || stage.ms <= 0) continue;
    const rawWidth = (stage.ms / total) * 100;
    const widthPct = Math.max(0, Math.min(100 - cursor, rawWidth));
    if (widthPct <= 0) continue;
    segments.push({
      key: stage.key, label: TIMING_STAGE_LABEL[stage.key], ms: stage.ms,
      startPct: cursor, widthPct,
    });
    cursor += widthPct;
  }
  return segments;
}

export interface FederatedHitLike {
  origin: 'private' | 'public';
  project?: string | null;
  host?: string | null;
  source_id?: string | null;
  revision?: string | null;
  domain?: string | null;
}

/** Short revision for display — 10 chars (matches PublicCodeConsole's own
 * shortRevision convention), or null so callers OMIT the field rather than
 * showing an empty placeholder (F1 item 16: omit, don't dash-fill). */
export function shortRevision(revision: string | null | undefined): string | null {
  return revision ? revision.slice(0, 10) : null;
}

/** One hit's source marker (design addendum: "CodeHitCard 加来源标记 — 私有项目名 /
 * 公共源 + revision 短 hash"). No bilingual "private · 私有" — one label only. */
export function originLabel(hit: FederatedHitLike): string {
  if (hit.origin === 'private') return hit.project || '私有项目';
  const rev = shortRevision(hit.revision);
  return rev ? `${hit.source_id ?? '公共源'} · ${rev}` : (hit.source_id ?? '公共源');
}

/** Last `segments` path components, marked with a leading "…/" when the path
 * had more than that — the project tree row (design fix: "long paths wrap to
 * two lines... make each row one line") shows this instead of the raw path,
 * with the full path left for a `title` tooltip. Not true mid-string
 * ellipsis (CSS can't do that): showing only the tail is what actually
 * identifies a project when paths share a long common prefix (e.g. every
 * host's "…/tilelang_work/repos/X" reads distinctly; a head-truncated
 * "C:/Users/.../repos/X" would too, but drops the very host/drive context a
 * reader would want first if two projects happen to share a leaf name). */
export function pathTail(path: string, segments = 2): string {
  const parts = path.split(/[/\\]/).filter(Boolean);
  if (parts.length <= segments) return path;
  return `…/${parts.slice(-segments).join('/')}`;
}

export interface DistributionSegment {
  key: string;
  label: string;
  count: number;
  pct: number;
  origin?: 'private' | 'public';
}

/** origin (private/public) split — the top segmented bar in the code results
 * board (design addendum: "私有 / 公共分段条"). Stable order: private first. */
export function originDistribution(hits: FederatedHitLike[]): DistributionSegment[] {
  if (hits.length === 0) return [];
  const counts = { private: 0, public: 0 };
  for (const h of hits) counts[h.origin] += 1;
  const total = hits.length;
  return (['private', 'public'] as const)
    .filter((key) => counts[key] > 0)
    .map((key) => ({
      key, label: key === 'private' ? '私有' : '公共',
      count: counts[key], pct: (counts[key] / total) * 100,
    }));
}

/** Per-scope bar (design addendum: "按项目的条形图，点击任一段即加入筛选") — private hits
 * group by project, public hits group by source_id, sorted by count descending so
 * the biggest contributor reads first. */
export function scopeDistribution(hits: FederatedHitLike[]): DistributionSegment[] {
  if (hits.length === 0) return [];
  // Bucketed by "origin:key" so a private project and a public source_id that
  // happen to share the same name (e.g. both called "aiter") never merge into
  // one bar — origin is carried through separately so a caller's click-to-
  // filter can route a private segment to the project filter and a public
  // one to the source filter instead of conflating the two scope kinds.
  const counts = new Map<string, { key: string; origin: 'private' | 'public'; count: number }>();
  for (const h of hits) {
    const key = h.origin === 'private' ? (h.project || '私有项目') : (h.source_id || '公共源');
    const bucket = `${h.origin}:${key}`;
    const entry = counts.get(bucket);
    if (entry) entry.count += 1;
    else counts.set(bucket, { key, origin: h.origin, count: 1 });
  }
  const total = hits.length;
  return [...counts.values()]
    .sort((a, b) => b.count - a.count)
    .map(({ key, origin, count }) => ({ key, label: key, count, pct: (count / total) * 100, origin }));
}
