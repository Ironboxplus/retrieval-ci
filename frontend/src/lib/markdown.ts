import type { Options } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

// One plugin set for every paper/journal Markdown surface: GFM tables plus
// $inline$ / $$display$$ LaTeX (paper Markdown from marker and the arXiv
// converter both write math that way). Bad TeX renders as red source text
// instead of throwing, so one malformed formula never blanks a result.
export const remarkPlugins: Options['remarkPlugins'] = [remarkGfm, remarkMath];
export const rehypePlugins: Options['rehypePlugins'] = [
  [rehypeKatex, { throwOnError: false, strict: 'ignore' }],
];

/** A line that is only `$$ ... $$`, optionally followed by its number as in
 *  `$$ ... $$ (5)`, is a display equation (how the arXiv converter and marker
 *  write them), but remark-math reads single-line `$$` as inline math; give
 *  it the fenced form so it renders as a block, the number as `\tag`. */
export function displayMath(text: string): string {
  return text.replace(
    /^[ \t]*\$\$([^\n$]+?)\$\$[ \t]*(?:\(([\w.]{1,8})\))?[ \t]*$/gm,
    (_m, body: string, tag?: string) => `$$\n${body.trim()}${tag ? ` \\tag{${tag}}` : ''}\n$$`,
  );
}

/** Cut Markdown to at most `max` characters without leaving a `$` formula
 *  open (an unpaired `$` would swallow the rest of the preview as math). */
export function clipMarkdown(text: string, max: number): string {
  if (text.length <= max) return text;
  let cut = text.slice(0, max);
  const dollars = cut.match(/(?<!\\)\$/g)?.length ?? 0;
  if (dollars % 2 === 1) {
    const open = cut.search(/(?<!\\)\$+[^$]*$/);
    if (open >= 0) cut = cut.slice(0, open);
  }
  return `${cut.trimEnd()}…`;
}
