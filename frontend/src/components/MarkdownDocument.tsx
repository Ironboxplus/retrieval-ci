import { useMemo } from 'react';
import { Tag, Typography, theme } from 'antd';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { displayMath, rehypePlugins, remarkPlugins } from '../lib/markdown';

const { Text } = Typography;

interface MarkdownDocumentProps {
  source: string;
  maxHeight?: string | number;
}

/** Compact, readable Markdown rendering shared by long-form document surfaces. */
export default function MarkdownDocument({ source, maxHeight = '76vh' }: MarkdownDocumentProps) {
  const { token } = theme.useToken();
  const components = useMemo<Components>(() => {
    const cellStyle = {
      border: `1px solid ${token.colorBorderSecondary}`,
      padding: '5px 9px',
      verticalAlign: 'top',
    } as const;
    const heading = (fontSize: number, marginTop: number) => ({ children }: { children?: React.ReactNode }) => (
      <Text strong style={{ display: 'block', fontSize, lineHeight: 1.35, margin: `${marginTop}px 0 5px` }}>
        {children}
      </Text>
    );

    return {
      h1: heading(21, 18),
      h2: heading(18, 16),
      h3: heading(16, 13),
      h4: heading(14, 11),
      h5: heading(13, 9),
      h6: heading(12, 8),
      p: ({ children }) => <p style={{ margin: '5px 0', lineHeight: 1.65 }}>{children}</p>,
      ul: ({ children }) => <ul style={{ margin: '5px 0', paddingLeft: 24 }}>{children}</ul>,
      ol: ({ children }) => <ol style={{ margin: '5px 0', paddingLeft: 24 }}>{children}</ol>,
      li: ({ children }) => <li style={{ margin: '2px 0', lineHeight: 1.55 }}>{children}</li>,
      blockquote: ({ children }) => (
        <blockquote style={{
          margin: '7px 0',
          padding: '5px 11px',
          borderLeft: `3px solid ${token.colorPrimaryBorder}`,
          background: token.colorFillQuaternary,
          color: token.colorTextSecondary,
        }}>
          {children}
        </blockquote>
      ),
      pre: ({ children }) => (
        <pre style={{
          margin: '8px 0',
          padding: '10px 12px',
          overflowX: 'auto',
          border: `1px solid ${token.colorBorderSecondary}`,
          borderRadius: token.borderRadius,
          background: token.colorFillTertiary,
          fontSize: 12,
          lineHeight: 1.5,
        }}>
          {children}
        </pre>
      ),
      code: ({ children, className }) => (
        <code className={className} style={{
          padding: className ? 0 : '1px 4px',
          borderRadius: 4,
          background: className ? 'transparent' : token.colorFillTertiary,
          fontSize: '0.92em',
        }}>
          {children}
        </code>
      ),
      table: ({ children }) => (
        <div style={{ margin: '8px 0', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
            {children}
          </table>
        </div>
      ),
      th: ({ children }) => <th style={{ ...cellStyle, background: token.colorFillQuaternary, textAlign: 'left' }}>{children}</th>,
      td: ({ children }) => <td style={cellStyle}>{children}</td>,
      a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
      img: ({ alt, src }) => (
        <Tag title={src} style={{ margin: '3px 0', whiteSpace: 'normal' }}>
          Figure: {alt || src || 'image reference'}
        </Tag>
      ),
      hr: () => <hr style={{ border: 0, borderTop: `1px solid ${token.colorBorderSecondary}`, margin: '12px 0' }} />,
    };
  }, [token]);

  return (
    <div style={{
      maxHeight,
      overflow: 'auto',
      padding: '14px 20px 22px',
      color: token.colorText,
      fontSize: 13.5,
      background: token.colorBgContainer,
    }}>
      <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
        {displayMath(source)}
      </ReactMarkdown>
    </div>
  );
}
