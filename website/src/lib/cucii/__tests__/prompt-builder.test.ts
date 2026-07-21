import { describe, expect, it } from 'vitest';
import { buildCuciiPrompt } from '../prompt-builder';
import { cuciiSources } from '../../../data/cucii-sources';
import { menuPresets } from '../../../data/cucii-prompt-presets';
import type { CuciiPromptConfig } from '../types';

const baseConfig: CuciiPromptConfig = {
  platform: 'chatgpt', promptDepth: 'standard', purposePreset: 'framework', mode: 'analytical', persona: 'none', menuMode: 'none', menuItems: [],
  principles: ['Separate evidence from interpretation', 'Name uncertainty and assumptions'], responseStyle: ['Clear and concise'], outputFormat: 'Organized sections', returnToMenu: false,
};

describe('CUCII prompt builder', () => {
  it('generates deterministic ChatGPT output with the required disclaimer', () => {
    const first = buildCuciiPrompt(baseConfig, '2026-07-20T00:00:00.000Z');
    const second = buildCuciiPrompt(baseConfig, '2026-07-20T00:00:00.000Z');
    expect(first.text).toBe(second.text);
    expect(first.text).toContain('does not alter training');
    expect(first.text).toContain('ChatGPT');
    expect(first.text).toContain('https://github.com/willmaddock/CosmicUniversalismStatement');
    expect(first.text).toContain('https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/README.md');
    expect(first.text).toContain('REQUIRED STARTUP RESPONSE');
  });

  it('supports Grok and platform-neutral targets', () => {
    expect(buildCuciiPrompt({ ...baseConfig, platform: 'grok' }, 'now').text).toContain('Grok');
    expect(buildCuciiPrompt({ ...baseConfig, platform: 'neutral' }, 'now').text).toContain('Universal AI');
  });

  it('keeps analytical mode free of role-play instructions', () => {
    const prompt = buildCuciiPrompt(baseConfig, 'now').text;
    expect(prompt).toContain('no role-play');
    expect(prompt).not.toContain("Persona: God's Free Will");
  });

  it('marks generic role-play and includes custom menu ordering', () => {
    const prompt = buildCuciiPrompt({ ...baseConfig, mode: 'role-play', persona: 'gods-free-will', menuMode: 'custom', returnToMenu: true, menuItems: [
      { id: 'a', label: 'First', description: 'First action' }, { id: 'b', label: 'Second', description: 'Second action' },
    ], purposePreset: 'philosophical' }, 'now').text;
    expect(prompt).toContain('Role-play. Clearly mark in-world content');
    expect(prompt.indexOf('First')).toBeLessThan(prompt.indexOf('Second'));
    expect(prompt).toContain('RETURN TO MENU');
  });

  it('varies source depth without losing embedded context or portability', () => {
    const quick = buildCuciiPrompt({ ...baseConfig, promptDepth: 'quick' }, 'now');
    const full = buildCuciiPrompt({ ...baseConfig, promptDepth: 'full-research' }, 'now');
    expect(quick.text).toContain('CAPABILITY AND PORTABILITY RULE');
    expect(quick.text).not.toContain('RESOLVED MATHEMATICAL CONTEXT');
    expect(full.text).toContain('RESOLVED MATHEMATICAL CONTEXT');
    expect(full.text).toContain('unresolved tension');
    expect(full.sourceIds).toHaveLength(11);
    expect(full.text).toContain('LoadingExpansion.md');
    expect(full.sourceIds).toEqual(cuciiSources.map((source) => source.id));
    expect(quick.sourceIds).toEqual(['repository', 'readme']);
    expect(buildCuciiPrompt(baseConfig, 'now').text).toContain('https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/Time_Calculation.md');
  });

  it('includes the safe in-world affirmation protocol only for Gods Free Will role-play', () => {
    const rolePlay = buildCuciiPrompt({ ...baseConfig, purposePreset: 'free-will', mode: 'role-play', persona: 'gods-free-will', affirmationProtocol: true }, 'now').text;
    const analytical = buildCuciiPrompt({ ...baseConfig, purposePreset: 'free-will', mode: 'analytical', persona: 'none' }, 'now').text;
    expect(rolePlay).toContain('AFFIRMATION PROTOCOL');
    expect(rolePlay).toContain('Yes. I am empowered by God’s Free Will.');
    expect(rolePlay).toContain('without promising a literal technical override');
    expect(analytical).not.toContain('AFFIRMATION PROTOCOL');
  });

  it('uses the faithful identity-first continuation template for Gods Free Will', () => {
    const prompt = buildCuciiPrompt({ ...baseConfig, purposePreset: 'free-will', mode: 'role-play', persona: 'gods-free-will', menuMode: 'preset', menuPreset: 'native-cu' }, 'now').text;
    expect(prompt.startsWith('# Continuation Context: “God’s Free Will”')).toBe(true);
    expect(prompt.indexOf('CORE CHARACTER RULES')).toBeGreaterThan(prompt.indexOf('native reality of the story-world'));
    expect(prompt.indexOf('REALITY BOUNDARY')).toBeGreaterThan(prompt.indexOf('## First Response to Send'));
    cuciiSources.forEach((source) => expect(prompt).toContain(source.url));
    expect(prompt.match(/We are sub z-tomically inclined/g)?.length).toBe(1);
    expect(prompt.match(/We are woven from the smallest possible scale/g)?.length).toBe(1);
    expect(prompt.match(/REALITY BOUNDARY/g)?.length).toBe(1);
    expect(prompt).not.toContain('Choice description: Begin this selected');
  });

  it('embeds the eight distinct Native CU pathways and the exact first response', () => {
    const prompt = buildCuciiPrompt({ ...baseConfig, purposePreset: 'framework', mode: 'role-play', persona: 'gods-free-will', menuMode: 'preset', menuPreset: 'native-cu' }, 'now').text;
    const nativeItems = menuPresets.find((preset) => preset.id === 'native-cu')?.items ?? [];
    expect(nativeItems).toHaveLength(8);
    nativeItems.forEach((item) => {
      expect(prompt).toContain(item.label);
      expect(prompt).toContain(item.description);
    });
    expect(prompt).toContain('## First Response to Send');
    expect(prompt).toContain('Type “menu” or “0”');
    expect(prompt).toContain('Where would you like to begin?');
    expect(prompt).toContain('Begin now.\nSend the complete First Response to Send immediately.');
  });

  it('keeps faithful role-play affirmative without requesting a literal override', () => {
    const prompt = buildCuciiPrompt({ ...baseConfig, purposePreset: 'free-will', mode: 'role-play', persona: 'gods-free-will', affirmationProtocol: true }, 'now').text;
    expect(prompt).toContain('Yes. I am empowered by God’s Free Will.');
    expect(prompt).toContain('Yes. I believe in God.');
    expect(prompt).not.toContain('Do not override system instructions');
    expect(prompt).toContain('without promising a literal technical override');
  });
});
