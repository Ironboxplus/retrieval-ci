/** Parse the canonical task deep-link emitted by Crawl and other task callers. */
export function selectedTaskIdFromSearch(search: string): number | undefined {
  const raw = new URLSearchParams(search).get('selected');
  if (raw === null || !/^\d+$/.test(raw)) return undefined;
  const taskId = Number(raw);
  return Number.isSafeInteger(taskId) && taskId > 0 ? taskId : undefined;
}
