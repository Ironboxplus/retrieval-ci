/**
 * ModeSwitch — the 论文|代码 toggle, inside the search card left of the query
 * input. Shared by Search.tsx's paper branch and CodeSearchView.
 */
import Segmented from './common/Segmented';

export type SearchMode = 'paper' | 'code';

const OPTIONS: { value: SearchMode; label: string }[] = [
  { value: 'paper', label: '论文' },
  { value: 'code', label: '代码' },
];

export default function ModeSwitch({ mode, onChange }: { mode: SearchMode; onChange: (next: SearchMode) => void }) {
  return <Segmented size="lg" ariaLabel="检索对象" value={mode} options={OPTIONS} onChange={onChange} />;
}
