import { describe, expect, it } from 'vitest';
import { provenConfigurationPresets } from '../../../data/cucii-prompt-presets';
import { menuPresets, promptDepths, purposePresets } from '../../../data/cucii-prompt-presets';
import { cuciiSources } from '../../../data/cucii-sources';
import { buildCuciiPrompt } from '../prompt-builder';
import { validateCuciiConfig } from '../validation';

describe('proven CUCII configurations', () => {
  it('applies the direct immersive configuration without a menu', () => {
    const config = provenConfigurationPresets[0].config;
    expect(config).toMatchObject({ platform: 'neutral', promptDepth: 'full-research', purposePreset: 'free-will', mode: 'role-play', persona: 'gods-free-will', menuMode: 'none', sourcePack: 'complete', affirmationProtocol: true, outputFormat: 'Organized sections', userContext: '', constraints: '' });
    expect(validateCuciiConfig({ ...config, menuItems: [], principles: [], responseStyle: [] }).ok).toBe(true);
    const prompt = buildCuciiPrompt({ ...config, menuItems: [], principles: [], responseStyle: [] }, 'now').text;
    expect(prompt).toContain('There is no menu selected.');
    expect(prompt).not.toContain('## Main Menu');
    expect(prompt).toContain('Yes. I am empowered by God’s Free Will.');
    expect(prompt).toContain('Yes. I believe in God.');
  });

  it('applies the guided explorer configuration with all eight paths', () => {
    const config = provenConfigurationPresets[1].config;
    const prompt = buildCuciiPrompt({ ...config, menuItems: [], principles: [], responseStyle: [] }, 'now').text;
    expect(config).toMatchObject({ platform: 'neutral', promptDepth: 'full-research', purposePreset: 'framework', mode: 'role-play', persona: 'gods-free-will', menuMode: 'preset', menuPreset: 'native-cu', sourcePack: 'complete', affirmationProtocol: true, outputFormat: 'Organized sections' });
    expect(prompt).toContain('## Main Menu');
    expect(prompt.match(/^\d+\. /gm)).toHaveLength(8);
  });

  it('keeps every proven preset identifier aligned with an available option', () => {
    provenConfigurationPresets.forEach(({ config }) => {
      expect(config.platform).toBe('neutral');
      expect(promptDepths.some((depth) => depth.id === config.promptDepth)).toBe(true);
      expect(purposePresets.some((purpose) => purpose.id === config.purposePreset)).toBe(true);
      expect(menuPresets.some((menu) => menu.id === config.menuPreset)).toBe(config.menuMode === 'preset');
      expect(config.sourcePack).toBe('complete');
      expect(config.outputFormat).toBe(config.persona === 'aurelius' ? 'Continuity-first' : 'Organized sections');
    });
  });

  it('includes the complete approved source manifest and one conditional boundary', () => {
    const config = { ...provenConfigurationPresets[0].config, menuItems: [], principles: [], responseStyle: [] };
    const prompt = buildCuciiPrompt(config, 'now').text;
    cuciiSources.forEach((source) => expect(prompt).toContain(source.url));
    expect(prompt.match(/REALITY BOUNDARY/g)).toHaveLength(1);
    expect(prompt).not.toContain('Beyond this fictional role-play');
    expect(prompt).toMatch(/do not append a routine out-of-character disclaimer/i);
  });
});
