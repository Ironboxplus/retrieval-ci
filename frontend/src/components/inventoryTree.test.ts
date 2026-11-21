import { describe, expect, it } from 'vitest';

import { buildInventoryTree, baseName, projectScopeOptions, maxProjectChunkCount } from './inventoryTree';
import type { InventoryResponse } from '../api/hooks/useAdminConsole';

function file(relPath: string, overrides: Partial<InventoryResponse['users'][number]['hosts'][number]['projects'][number]['dirs'][number]['files'][number]> = {}) {
  return {
    rel_path: relPath,
    kind: 'code',
    collection: 'code_chunks',
    language: 'ts',
    chunk_count: 3,
    indexed_at: null,
    ...overrides,
  };
}

describe('baseName', () => {
  it('splits on forward slash', () => {
    expect(baseName('src/components/Foo.tsx')).toBe('Foo.tsx');
  });
  it('splits on backslash (legacy Windows client rows)', () => {
    expect(baseName('src\\components\\Foo.tsx')).toBe('Foo.tsx');
  });
  it('returns empty string for empty input', () => {
    expect(baseName('')).toBe('');
  });
});

describe('buildInventoryTree', () => {
  it('returns empty result for undefined inventory', () => {
    const { nodes, fileMap, projectMap } = buildInventoryTree(undefined);
    expect(nodes).toEqual([]);
    expect(fileMap.size).toBe(0);
    expect(projectMap.size).toBe(0);
  });

  it('returns empty result for an inventory with no users', () => {
    const inv: InventoryResponse = { users: [], totals: {} };
    const { nodes, fileMap, projectMap } = buildInventoryTree(inv);
    expect(nodes).toEqual([]);
    expect(fileMap.size).toBe(0);
    expect(projectMap.size).toBe(0);
  });

  it('builds a full user -> host -> project -> dir -> file tree for multiple users/hosts', () => {
    const inv: InventoryResponse = {
      totals: { users: 2 },
      users: [
        {
          owner_user_id: 'alice', file_count: 2, chunk_count: 5,
          hosts: [
            {
              host: 'laptop', file_count: 2, chunk_count: 5,
              projects: [
                {
                  project: 'retrival', host: 'laptop', root_dir: 'C:/Prd/retrival',
                  file_count: 2, chunk_count: 5,
                  dirs: [
                    {
                      dir: 'src', file_count: 2, chunk_count: 5,
                      files: [file('src/a.ts'), file('src/b.ts', { chunk_count: 2 })],
                    },
                  ],
                },
              ],
            },
          ],
        },
        {
          owner_user_id: 'bob', file_count: 1, chunk_count: 1,
          hosts: [
            {
              host: 'server', file_count: 1, chunk_count: 1,
              projects: [
                {
                  project: 'other', host: 'server', root_dir: '/srv/other',
                  file_count: 1, chunk_count: 1,
                  dirs: [{ dir: 'lib', file_count: 1, chunk_count: 1, files: [file('lib/c.ts')] }],
                },
              ],
            },
          ],
        },
      ],
    };

    const { nodes, fileMap, projectMap } = buildInventoryTree(inv);

    expect(nodes).toHaveLength(2);
    expect(nodes[0].key).toBe('u:alice');
    expect(nodes[0].type).toBe('user');
    expect(nodes[1].key).toBe('u:bob');

    const aliceHost = nodes[0].children![0];
    expect(aliceHost.key).toBe('u:alice|h:laptop');
    expect(aliceHost.type).toBe('host');

    const aliceProject = aliceHost.children![0];
    expect(aliceProject.key).toBe('alice|laptop|p:retrival');
    expect(aliceProject.type).toBe('project');
    expect(aliceProject.selectable).toBe(false); // has dirs -> group node, not a leaf

    const aliceDir = aliceProject.children![0];
    expect(aliceDir.label).toBe('src/');
    expect(aliceDir.children).toHaveLength(2);

    const fileNode = aliceDir.children![0];
    expect(fileNode.isLeaf).toBe(true);
    expect(fileNode.label).toBe('a.ts');
    expect(fileNode.key).toBe('alice|laptop|retrival|src|f:src/a.ts');

    // fileMap and projectMap are populated as pure output of the same pass
    expect(fileMap.size).toBe(3);
    expect(fileMap.get('alice|laptop|retrival|src|f:src/a.ts')).toEqual({
      file: file('src/a.ts'), owner: 'alice', host: 'laptop', project: 'retrival',
    });
    expect(projectMap.size).toBe(2);
    expect(projectMap.get('alice|laptop|p:retrival')).toEqual({ owner: 'alice', host: 'laptop', project: 'retrival' });
  });

  it('keeps same-named projects on different hosts distinct', () => {
    const projectShape = (host: string) => ({
      project: 'shared-name', host, root_dir: `/root/${host}`,
      file_count: 1, chunk_count: 1,
      dirs: [{ dir: 'src', file_count: 1, chunk_count: 1, files: [file('src/x.ts')] }],
    });
    const inv: InventoryResponse = {
      totals: {},
      users: [
        {
          owner_user_id: 'alice', file_count: 2, chunk_count: 2,
          hosts: [
            { host: 'host-a', file_count: 1, chunk_count: 1, projects: [projectShape('host-a')] },
            { host: 'host-b', file_count: 1, chunk_count: 1, projects: [projectShape('host-b')] },
          ],
        },
      ],
    };

    const { projectMap, fileMap } = buildInventoryTree(inv);

    expect(projectMap.size).toBe(2);
    expect(projectMap.has('alice|host-a|p:shared-name')).toBe(true);
    expect(projectMap.has('alice|host-b|p:shared-name')).toBe(true);
    // file keys must also stay distinct per host despite identical project/dir/file names
    expect(fileMap.size).toBe(2);
  });

  it('a project leaf (no dirs) is selectable', () => {
    const inv: InventoryResponse = {
      totals: {},
      users: [{
        owner_user_id: 'alice', file_count: 0, chunk_count: 0,
        hosts: [{ host: 'h', file_count: 0, chunk_count: 0, projects: [
          { project: 'empty-project', host: 'h', root_dir: null, file_count: 0, chunk_count: 0, dirs: [] },
        ] }],
      }],
    };
    const { nodes } = buildInventoryTree(inv);
    const project = nodes[0].children![0].children![0];
    expect(project.selectable).toBe(true);
    expect(project.children).toEqual([]);
  });
});

describe('maxProjectChunkCount', () => {
  it('returns the largest chunkCount among all project-type nodes, at any depth', () => {
    const inv: InventoryResponse = {
      totals: {},
      users: [{
        owner_user_id: 'alice', file_count: 0, chunk_count: 0,
        hosts: [{
          host: 'laptop', file_count: 0, chunk_count: 0,
          projects: [
            { project: 'small', host: 'laptop', root_dir: null, file_count: 1, chunk_count: 5, dirs: [] },
            { project: 'big', host: 'laptop', root_dir: null, file_count: 10, chunk_count: 500, dirs: [] },
          ],
        }],
      }],
    };
    const { nodes } = buildInventoryTree(inv);
    expect(maxProjectChunkCount(nodes)).toBe(500);
  });

  it('is 0 for an empty tree', () => {
    expect(maxProjectChunkCount([])).toBe(0);
  });
});

describe('projectScopeOptions', () => {
  it('flattens to owner|host|project scope strings for the code search TreeSelect', () => {
    const inv: InventoryResponse = {
      totals: {},
      users: [{
        owner_user_id: 'alice', file_count: 0, chunk_count: 0,
        hosts: [{ host: 'laptop', file_count: 0, chunk_count: 0, projects: [
          { project: 'retrival', host: 'laptop', root_dir: null, file_count: 0, chunk_count: 0, dirs: [] },
        ] }],
      }],
    };
    const options = projectScopeOptions(inv);
    expect(options).toEqual([
      { value: 'alice|laptop|retrival', label: 'retrival', owner: 'alice', host: 'laptop', project: 'retrival' },
    ]);
  });

  it('returns an empty list for an empty inventory', () => {
    expect(projectScopeOptions(undefined)).toEqual([]);
  });
});
