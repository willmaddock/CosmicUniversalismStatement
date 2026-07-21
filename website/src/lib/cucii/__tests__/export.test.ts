import { describe, expect, it } from 'vitest';
import { exportFilename } from '../export';

describe('CUCII exports', () => {
  it('creates safe, typed filenames', () => {
    const prompt = { text: 'prompt', filenameStem: 'cucii-chatgpt-framework-prompt-v1.0', platform: 'chatgpt' as const, version: '1.0' as const, generatedAt: 'now', menuItemCount: 0, warnings: [] };
    expect(exportFilename(prompt, 'txt')).toBe('cucii-chatgpt-framework-prompt-v1.0.txt');
    expect(exportFilename({ ...prompt, filenameStem: 'unsafe name' }, 'md')).toBe('unsafe-name.md');
  });
});
