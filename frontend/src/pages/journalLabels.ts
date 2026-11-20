/**
 * Chinese labels for the journal entry's free-form `source_kind` and
 * `client` fields (app/services/journal/schemas.py — "如
 * memory/session_summary/handoff/manual_note" and "如 http/mcp/codex/claude")
 * so the all-Chinese Settings›记忆 UI never shows a bare lowercase English
 * tag. Both fields are open text on the backend, so unrecognized values fall
 * back to the raw string rather than throwing.
 */

const SOURCE_KIND_LABEL: Record<string, string> = {
  memory: '记忆',
  session_summary: '会话摘要',
  handoff: '交接',
  manual_note: '手动笔记',
  doc: '文档',
};

export function sourceKindLabel(kind: string): string {
  return SOURCE_KIND_LABEL[kind] ?? kind;
}

// `client` names a product/protocol, not a translatable word — display-case
// it instead of inventing a Chinese term that doesn't exist for "MCP".
const CLIENT_LABEL: Record<string, string> = {
  mcp: 'MCP',
  http: 'HTTP',
  codex: 'Codex',
  claude: 'Claude',
};

export function clientLabel(client: string): string {
  return CLIENT_LABEL[client] ?? client;
}
