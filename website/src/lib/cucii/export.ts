import type { GeneratedCuciiPrompt } from './types';

export function exportFilename(prompt: GeneratedCuciiPrompt, extension: 'txt' | 'md') {
  return `${prompt.filenameStem}.${extension}`.replace(/[^a-z0-9._-]/gi, '-');
}

export function downloadCuciiPrompt(prompt: GeneratedCuciiPrompt, extension: 'txt' | 'md') {
  const content = extension === 'md' ? `# CUCII Prompt\n\n${prompt.text}\n` : `${prompt.text}\n`;
  const blob = new Blob([content], { type: extension === 'md' ? 'text/markdown;charset=utf-8' : 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = exportFilename(prompt, extension);
  anchor.click();
  URL.revokeObjectURL(url);
}
