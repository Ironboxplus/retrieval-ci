/** Copy text with a browser-compatible fallback for non-secure contexts. */
export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Sandboxed/non-secure browsers may expose the API but reject the write.
      // Fall through to the legacy selection-based copy path.
    }
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  let copied = false;
  try {
    textarea.select();
    copied = document.execCommand('copy');
  } finally {
    document.body.removeChild(textarea);
  }
  if (!copied) throw new Error('Browser copy command failed');
}
