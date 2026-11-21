import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import NumberedCodeSnippet from './NumberedCodeSnippet';

describe('source code line numbers', () => {
  it('starts at the source line, counts blank lines, and keeps numbers outside code', () => {
    const code = 'def forward(x):\n\n    return x < 2';
    const html = renderToStaticMarkup(<NumberedCodeSnippet code={code} startLine={455} />);
    expect(html).toContain('455\n456\n457');
    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain('select-none');
    expect(html).toContain('<code>def forward(x):\n\n    return x &lt; 2</code>');
  });

  it('renders literal fences and HTML as code instead of Markdown', () => {
    const html = renderToStaticMarkup(
      <NumberedCodeSnippet code={'```\n<script>example</script>'} startLine={9} />,
    );
    expect(html).toContain('9\n10');
    expect(html).toContain('&lt;script&gt;');
    expect(html).not.toContain('<script>');
  });

  it('does not fabricate numbers when the source location is invalid', () => {
    const html = renderToStaticMarkup(<NumberedCodeSnippet code="raw code" startLine={0} />);
    expect(html).toContain('<code>raw code</code>');
    expect(html).not.toContain('aria-hidden');
  });
});
