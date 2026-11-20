# Retrieval Frontend

React 19 + TypeScript + Ant Design 6 + Vite frontend for paper search, paper details, Markdown rendering, document matching, landscape analysis, documents, Code KB and Journal administration.

Architecture: [../overview.md](../overview.md). API contracts: [../docs/API.md](../docs/API.md).

## Builds

```powershell
# Internal/admin build
npm run build

# Public build
$env:VITE_PUBLIC_MODE='1'
npm run build
```

The builds are separate release artifacts. Public-mode route visibility is not an authorization boundary; gateway/compute APIs must still enforce every permission.

## Important Paths

| Path | Responsibility |
|---|---|
| `src/App.tsx` | route tree and public/internal page selection |
| `src/api/client.ts` | shared relative `/api` HTTP client |
| `src/api/hooks/` | query/mutation contracts |
| `src/pages/Search.tsx` | paper search |
| `src/pages/PaperDetail.tsx` | paper metadata, chunks and artifacts |
| `src/components/MarkdownDocument.tsx` | rendered Markdown and source copy |
| `src/pages/MatchDoc.tsx` | document matching |
| `src/pages/Analysis.tsx` | paper landscape |
| `src/pages/JournalConsole.tsx` | Journal/Index/Playbook administration |

## Verification

Run the build plus a real browser smoke. Check route rendering, API 401/403 handling, session refresh, Markdown rendering, Range downloads, responsive layout and browser console errors.
