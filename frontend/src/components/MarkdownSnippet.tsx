import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { clipMarkdown, displayMath, rehypePlugins, remarkPlugins } from '../lib/markdown';

const components: Components = {
  p: ({ children }) => <p style={{ margin: '2px 0' }}>{children}</p>,
  img: ({ alt }) => <span>[{alt || 'figure'}]</span>,
};

/** Short Markdown + LaTeX preview (abstracts, chunk previews): clipped
 *  without breaking a formula, paragraphs kept tight. */
export default function MarkdownSnippet({ source, max }: { source: string; max?: number }) {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
      {displayMath(max ? clipMarkdown(source, max) : source)}
    </ReactMarkdown>
  );
}
