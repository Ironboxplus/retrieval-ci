/** Keep source numbers in a separate, nonselectable gutter so code stays copyable. */
export default function NumberedCodeSnippet({ code, startLine }: { code: string; startLine: number }) {
  const source = code.replace(/\r\n/g, '\n');
  const numbered = Number.isInteger(startLine) && startLine > 0;
  const numbers = source.split('\n').map((_, offset) => startLine + offset).join('\n');

  return (
    <div className="flex gap-3 overflow-x-auto rounded-lg bg-muted/50 px-3 py-2.5 font-mono text-[12.5px] leading-relaxed">
      {numbered && (
        <pre aria-hidden="true" className="m-0 shrink-0 select-none border-r border-border pr-3 text-right tabular-nums text-muted-foreground">
          {numbers}
        </pre>
      )}
      <pre className="m-0 min-w-max whitespace-pre"><code>{source}</code></pre>
    </div>
  );
}
