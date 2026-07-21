import { describe, expect, it } from 'vitest';
import { validateCuciiConfig } from '../validation';
import type { CuciiPromptConfig } from '../types';

const config: CuciiPromptConfig = { platform: 'chatgpt', promptDepth: 'standard', purposePreset: 'framework', mode: 'analytical', persona: 'none', menuMode: 'none', menuItems: [], principles: [], responseStyle: [], outputFormat: 'Organized sections', returnToMenu: false };

describe('CUCII validation', () => {
  it('rejects missing custom purpose, invalid menu counts, and duplicates', () => {
    expect(validateCuciiConfig({ ...config, purposePreset: 'custom' }).ok).toBe(false);
    expect(validateCuciiConfig({ ...config, menuMode: 'custom', menuItems: [] }).ok).toBe(false);
    expect(validateCuciiConfig({ ...config, menuMode: 'custom', menuItems: [{ id: 'a', label: 'Same', description: 'One' }, { id: 'b', label: ' same ', description: 'Two' }] }).ok).toBe(false);
  });

  it('requires a persona for role-play and normalizes control characters', () => {
    const invalid = validateCuciiConfig({ ...config, mode: 'role-play' });
    expect(invalid.ok).toBe(false);
    const valid = validateCuciiConfig({ ...config, userContext: '  Safe\u0000 context  ' });
    expect(valid.ok && valid.value.userContext).toBe('Safe context');
  });

  it('rejects custom menus with more than 12 choices without discarding them', () => {
    const menuItems = Array.from({ length: 13 }, (_, index) => ({ id: String(index), label: `Choice ${index}`, description: 'A valid choice' }));
    const result = validateCuciiConfig({ ...config, menuMode: 'custom', menuItems });
    expect(result.ok).toBe(false);
    expect(result.ok ? [] : result.errors).toContainEqual({ code: 'MENU_COUNT', field: 'menuItems', message: 'Maximum of 12 menu choices reached.' });
    expect(result.ok ? 0 : result.errors.length).toBeGreaterThan(0);
  });

  it('accepts exactly twelve custom menu choices', () => {
    const menuItems = Array.from({ length: 12 }, (_, index) => ({ id: String(index), label: `Choice ${index}`, description: `A distinct choice ${index}` }));
    expect(validateCuciiConfig({ ...config, menuMode: 'custom', menuItems }).ok).toBe(true);
  });

  it('requires a supported prompt depth', () => {
    expect(validateCuciiConfig({ ...config, promptDepth: 'unsupported' as CuciiPromptConfig['promptDepth'] }).ok).toBe(false);
  });
});
