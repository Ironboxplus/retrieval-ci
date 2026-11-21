import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';

import MarkdownSnippet from '../components/MarkdownSnippet';
import { clipMarkdown } from './markdown';

// A real arXiv chunk shape (Mixtral 2401.04088, converted by arxiv_html_convert):
// inline and display math, a GFM table, emphasis.
const CHUNK = [
  'The output of the expert layer is $\\sum_{i=0}^{n-1}G(x)_{i}\\cdot E_{i}(x)$ where **G** is the gate.',
  '',
  '$$\\text{Softmax}(\\text{TopK}(x\\cdot W_{g})).$$',
  '',
  '| Model | Active Params | MMLU |',
  '| --- | --- | --- |',
  '| Mixtral 8x7B | 13B | 70.6% |',
].join('\n');

describe('paper Markdown rendering', () => {
  it('typesets inline and display LaTeX with KaTeX and keeps GFM', () => {
    const html = renderToStaticMarkup(<MarkdownSnippet source={CHUNK} />);
    expect(html).toContain('class="katex"');
    expect(html).toContain('class="katex-display"');
    expect(html).not.toContain('$\\sum');
    expect(html).toContain('<strong>G</strong>');
    expect(html).toContain('<td>Mixtral 8x7B</td>');
  });

  it('renders a numbered one-line equation as a display block with its tag', () => {
    // arXiv 2606.32000, equation (5), as the converter wrote it.
    const line = '$$\\mathbb{E}\\left[L_{\\mathrm{norm}}\\right]\\approx\\frac{1}{d}\\,\\mathrm{Tr}(W\\,\\Sigma_{x}\\,W^{T})$$ (5)';
    const html = renderToStaticMarkup(<MarkdownSnippet source={`Before.\n\n${line}\n\nAfter.`} />);
    expect(html).toContain('class="katex-display"');
    expect(html).toContain('class="tag"');
    expect(html).not.toContain('$$');
  });

  it('renders malformed TeX as text instead of throwing', () => {
    const html = renderToStaticMarkup(<MarkdownSnippet source={'bad $\\frac{1}{$ formula'} />);
    expect(html).toContain('formula');
  });

  it('clips without leaving a formula open', () => {
    const text = 'Loss is $\\mathcal{L}=\\sum_i \\ell(x_i)$ over the batch.';
    const cut = clipMarkdown(text, 20);
    expect(cut).toBe('Loss is…');
    expect(clipMarkdown(text, 200)).toBe(text);
    expect(clipMarkdown('price \\$5 and more text here', 12)).toBe('price \\$5 an…');
  });
});
