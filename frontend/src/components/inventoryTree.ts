import type { InventoryFile, InventoryResponse } from '../api/hooks/useAdminConsole';

/**
 * Pure user → host → project → dir → file tree shaping, extracted from
 * JournalConsole's former local `buildInventoryTree` (frontend refactor F1,
 * §2.2) so the same shape can drive both the Journal console's file-level
 * Tree and the code search project TreeSelect (which only needs to stop at
 * the project level — see `projectScopeOptions`).
 *
 * This module stays JSX-free on purpose: nodes carry plain data (label,
 * counts, kind) and the page that renders them decides the icon/Tag markup.
 */

export type InventoryNodeType = 'user' | 'host' | 'project' | 'dir' | 'file';

export interface InventoryTreeNode {
  key: string;
  type: InventoryNodeType;
  label: string;
  fileCount?: number;
  chunkCount?: number;
  rootDir?: string | null;
  language?: string | null;
  kind?: string;
  collection?: string | null;
  isDoc?: boolean;
  selectable: boolean;
  isLeaf?: boolean;
  children?: InventoryTreeNode[];
  /** Project nodes only — a running/queued code index job's id for this
   * scope, or null/undefined when there isn't one (F2-c's CodeScopePanel
   * shows a progress ring instead of the usual size bar while this is set). */
  codeActiveIndexJobId?: string | null;
}

export interface FileSel { file: InventoryFile; owner: string; host: string; project: string }
export interface ProjectSel { owner: string; host: string; project: string }

export interface InventoryTreeResult {
  nodes: InventoryTreeNode[];
  fileMap: Map<string, FileSel>;
  projectMap: Map<string, ProjectSel>;
}

/** Basename that tolerates both "/" and "\" — legacy inventory rows from
 * Windows clients stored backslash paths, so splitting on "/" alone left the
 * whole path as the "basename". */
export function baseName(p: string): string {
  return p.split(/[/\\]/).pop() || '';
}

/**
 * Build the full user → host → project → dir → file tree plus the leaf-key
 * lookup maps, as pure output of one pass (the maps are never mutated by a
 * separate render step).
 */
export function buildInventoryTree(inv: InventoryResponse | undefined): InventoryTreeResult {
  const fileMap = new Map<string, FileSel>();
  const projectMap = new Map<string, ProjectSel>();
  if (!inv) return { nodes: [], fileMap, projectMap };

  const nodes: InventoryTreeNode[] = inv.users.map((u) => ({
    key: `u:${u.owner_user_id}`,
    type: 'user',
    label: u.owner_user_id,
    fileCount: u.file_count,
    chunkCount: u.chunk_count,
    selectable: false,
    children: u.hosts.map((h) => ({
      key: `u:${u.owner_user_id}|h:${h.host}`,
      type: 'host',
      label: h.host,
      fileCount: h.file_count,
      selectable: false,
      children: h.projects.map((p) => {
        const projectKey = `${u.owner_user_id}|${h.host}|p:${p.project}`;
        projectMap.set(projectKey, { owner: u.owner_user_id, host: h.host, project: p.project });
        return {
          key: projectKey,
          type: 'project' as const,
          label: p.project,
          rootDir: p.root_dir,
          fileCount: p.file_count,
          chunkCount: p.chunk_count,
          codeActiveIndexJobId: p.code_active_index_job_id,
          selectable: p.dirs.length === 0,
          children: p.dirs.map((d) => ({
            key: `${u.owner_user_id}|${h.host}|${p.project}|d:${d.dir}`,
            type: 'dir' as const,
            label: `${d.dir}/`,
            fileCount: d.file_count,
            chunkCount: d.chunk_count,
            selectable: false,
            children: d.files.map((f) => {
              const key = `${u.owner_user_id}|${h.host}|${p.project}|${d.dir}|f:${f.rel_path}`;
              fileMap.set(key, { file: f, owner: u.owner_user_id, host: h.host, project: p.project });
              const isDoc = f.kind === 'doc' || f.collection === 'journal_chunks';
              return {
                key,
                type: 'file' as const,
                label: baseName(f.rel_path),
                chunkCount: f.chunk_count,
                language: f.language,
                kind: f.kind,
                collection: f.collection,
                isDoc,
                selectable: true,
                isLeaf: true,
              };
            }),
          })),
        };
      }),
    })),
  }));

  return { nodes, fileMap, projectMap };
}

/**
 * Largest `chunkCount` among all project-type nodes in the tree, at any
 * depth — the scale a project row's proportional mini-bar is drawn against
 * (design addendum §1.1: "project tree = indentation + a proportional
 * mini-bar per project instead of 'N chunks' text"). One global max keeps
 * every project's bar on the same scale, so bars are comparable across the
 * whole tree, not just among siblings.
 */
export function maxProjectChunkCount(nodes: InventoryTreeNode[]): number {
  let max = 0;
  for (const n of nodes) {
    if (n.type === 'project') max = Math.max(max, n.chunkCount ?? 0);
    if (n.children) max = Math.max(max, maxProjectChunkCount(n.children));
  }
  return max;
}

export interface ProjectScopeOption {
  /** Federated code search `projects[]` scope string: "owner|host|project". */
  value: string;
  label: string;
  owner: string;
  host: string;
  project: string;
}

/**
 * Project-level-only view of the same inventory, for the code search project
 * TreeSelect (F2): user/host nodes group and batch-select; only project
 * leaves are individually selectable, and their value is the scope string
 * the federated search `projects[]` filter expects.
 */
export function projectScopeOptions(inv: InventoryResponse | undefined): ProjectScopeOption[] {
  const { projectMap } = buildInventoryTree(inv);
  return [...projectMap.entries()].map(([key, sel]) => ({
    value: key.replace(/\|p:/, '|'),
    label: sel.project,
    ...sel,
  }));
}
