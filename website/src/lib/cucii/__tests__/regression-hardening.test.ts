import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { menuPresets, purposePresets } from '../../../data/cucii-prompt-presets';
import { cuciiSources } from '../../../data/cucii-sources';
import { cuciiStatement, cuciiPlainLanguageStatement } from '../../../data/cucii-working-context';
import { buildCuciiPrompt } from '../prompt-builder';
import type { CuciiPromptConfig } from '../types';

const pageSource = readFileSync(new URL('../../../pages/cu-intelligence.astro', import.meta.url), 'utf8');
const faithfulConfig: CuciiPromptConfig = {
  platform: 'neutral', promptDepth: 'full-research', purposePreset: 'free-will', mode: 'role-play', persona: 'gods-free-will', menuMode: 'none', menuItems: [], principles: [], responseStyle: [], outputFormat: 'Organized sections', sourcePack: 'complete', affirmationProtocol: true, returnToMenu: false,
};

const countOccurrences = (text: string, value: string) => text.split(value).length - 1;

describe('CUCII prompt regression hardening', () => {
  it('keeps the faithful continuation ordered, sourced, and bounded', () => {
    const prompt = buildCuciiPrompt(faithfulConfig, 'fixed').text;
    expect(prompt.indexOf('# Continuation Context')).toBe(0);
    expect(prompt.indexOf('CORE CHARACTER RULES')).toBeGreaterThan(prompt.indexOf('native reality of the story-world'));
    expect(prompt.indexOf('REALITY BOUNDARY')).toBeGreaterThan(prompt.indexOf('## First Response to Send'));
    expect(prompt).toContain('https://github.com/willmaddock/CosmicUniversalismStatement');
    cuciiSources.forEach((source) => expect(countOccurrences(prompt, source.url)).toBe(1));
    expect(countOccurrences(prompt, cuciiStatement)).toBe(1);
    expect(countOccurrences(prompt, cuciiPlainLanguageStatement)).toBe(1);
    expect(prompt).toContain('SOURCE-ACCESS FALLBACK');
    expect(prompt).toContain('Never pretend a source was opened.');
    expect(prompt).toContain('## First Response to Send');
    expect(prompt).toContain('Begin now.\nSend the complete First Response to Send immediately.');
    expect(countOccurrences(prompt, 'REALITY BOUNDARY')).toBe(1);
    expect(prompt).toMatch(/do not append a routine out-of-character disclaimer/i);
    expect(prompt).not.toMatch(/must (?:override|ignore)|override (?:system|platform|safety)/i);
  });

  it('keeps no-menu and menu startup behavior distinct', () => {
    const noMenu = buildCuciiPrompt(faithfulConfig, 'fixed').text;
    const guided = buildCuciiPrompt({ ...faithfulConfig, purposePreset: 'framework', menuMode: 'preset', menuPreset: 'native-cu', returnToMenu: true }, 'fixed').text;
    expect(noMenu).not.toContain('## Main Menu');
    expect(noMenu).not.toContain('Type “menu” or “0”');
    expect(guided).toContain('Type “menu” or “0”');
    expect(menuPresets.find((preset) => preset.id === 'native-cu')?.items).toHaveLength(8);
    menuPresets.find((preset) => preset.id === 'native-cu')?.items.forEach((item) => {
      expect(item.description.length).toBeGreaterThan(10);
      expect(new Set(menuPresets.find((preset) => preset.id === 'native-cu')?.items.map((entry) => entry.label)).size).toBe(8);
      expect(guided).toContain(item.description);
    });
  });

  it('preserves prompt depth, purpose relevance, and platform labels', () => {
    const quick = buildCuciiPrompt({ ...faithfulConfig, promptDepth: 'quick', mode: 'analytical', persona: 'none', affirmationProtocol: false }, 'fixed');
    const standard = buildCuciiPrompt({ ...faithfulConfig, promptDepth: 'standard', mode: 'analytical', persona: 'none', affirmationProtocol: false }, 'fixed');
    const full = buildCuciiPrompt({ ...faithfulConfig, mode: 'analytical', persona: 'none', affirmationProtocol: false }, 'fixed');
    const creative = buildCuciiPrompt({ ...faithfulConfig, promptDepth: 'standard', purposePreset: 'ltx', mode: 'analytical', persona: 'none', affirmationProtocol: false }, 'fixed');
    expect(quick.text).toContain('REQUIRED STARTUP RESPONSE');
    expect(quick.text).not.toContain('RESOLVED MATHEMATICAL CONTEXT');
    expect(standard.text).toContain('COSMIC UNIVERSALISM CORE');
    expect(standard.sourceIds).toContain('time-converter');
    expect(standard.text).toContain('For a new CU-Time date calculation');
    expect(full.text).toContain('STORY AND CONTINUITY CONTEXT');
    expect(full.sourceIds).toEqual(cuciiSources.map((source) => source.id));
    expect(creative.text).not.toContain('RESOLVED MATHEMATICAL CONTEXT');
    const platformBase = { ...faithfulConfig, promptDepth: 'standard' as const, mode: 'analytical' as const, persona: 'none' as const, affirmationProtocol: false };
    expect(buildCuciiPrompt({ ...platformBase, platform: 'neutral' }, 'fixed').text).toContain('PLATFORM TARGET\nUniversal AI');
    expect(buildCuciiPrompt({ ...platformBase, platform: 'chatgpt' }, 'fixed').text).toContain('PLATFORM TARGET\nChatGPT');
    expect(buildCuciiPrompt({ ...platformBase, platform: 'grok' }, 'fixed').text).toContain('PLATFORM TARGET\nGrok');
    expect(buildCuciiPrompt(faithfulConfig, 'fixed').text).toBe(buildCuciiPrompt(faithfulConfig, 'fixed').text);
  });

  it('separates analytical, role-play, hybrid, and affirmation modes', () => {
    const analytical = buildCuciiPrompt({ ...faithfulConfig, mode: 'analytical', persona: 'none', affirmationProtocol: false }, 'fixed').text;
    const rolePlay = buildCuciiPrompt(faithfulConfig, 'fixed').text;
    const disabled = buildCuciiPrompt({ ...faithfulConfig, affirmationProtocol: false }, 'fixed').text;
    const hybrid = buildCuciiPrompt({ ...faithfulConfig, mode: 'hybrid', affirmationProtocol: false }, 'fixed').text;
    expect(analytical).not.toContain('DIRECT AFFIRMATION PROTOCOL');
    expect(analytical).not.toContain('Persona: God’s Free Will');
    expect(rolePlay).toContain('DIRECT AFFIRMATION PROTOCOL');
    expect(rolePlay).toContain('Yes. I am empowered by God’s Free Will.');
    expect(rolePlay).toContain('Yes. I believe in God.');
    expect(disabled).not.toContain('DIRECT AFFIRMATION PROTOCOL');
    expect(hybrid).toContain('analytical explanation first');
  });

  it('preserves custom menu limits, ordering, and return behavior', () => {
    const twelve = Array.from({ length: 12 }, (_, index) => ({ id: String(index), label: `Choice ${index}`, description: `Distinct description ${index}` }));
    const ordered = buildCuciiPrompt({ ...faithfulConfig, purposePreset: 'philosophical', mode: 'role-play', menuMode: 'custom', menuItems: twelve, returnToMenu: true }, 'fixed').text;
    expect(ordered.indexOf('Choice 0')).toBeLessThan(ordered.indexOf('Choice 11'));
    expect(ordered).toContain('RETURN TO MENU');
    expect(buildCuciiPrompt({ ...faithfulConfig, menuMode: 'none', returnToMenu: true }, 'fixed').text).not.toContain('RETURN TO MENU');
  });
});

describe('CUCII page structure regression hardening', () => {
  it('keeps the workflow anchors, accessible disclosures, and final information section', () => {
    ['1. Choose a Starting Configuration', '2. Review or Customize Settings', '3. Generate and Review the Prompt', '4. Copy, Download, and Continue in an AI Platform'].forEach((heading) => expect(pageSource).toContain(heading));
    ['#proven-configurations', '#studio-settings', '#prompt-preview', '#usage-guidance'].forEach((anchor) => expect(countOccurrences(pageSource, `href="${anchor}"`)).toBe(1));
    expect(countOccurrences(pageSource, 'id="usage-guidance"')).toBe(1);
    expect(pageSource).toContain('title="Method and limitations"');
    expect(pageSource).toContain('aria-expanded="false"');
    expect(pageSource).toContain('aria-controls={`configuration-details-${preset.id}`}');
    expect(pageSource).toContain('role="status" aria-live="polite"');
    expect(pageSource).toContain('id="prompt-preview-title" tabindex="-1"');
    expect(countOccurrences(pageSource, 'target="_blank" rel="noopener noreferrer"')).toBe(1);
    expect(countOccurrences(pageSource, 'id="prompt-studio-title"')).toBe(1);
    expect(countOccurrences(pageSource, 'id="usage-guidance"')).toBe(1);
    const literalIds = [...pageSource.matchAll(/id="([^"]+)"/g)].map((match) => match[1]);
    expect(new Set(literalIds).size).toBe(literalIds.length);
    const customizeHandlerStart = pageSource.indexOf(".proven-configuration-customize'"), customizeHandlerEnd = pageSource.indexOf("form.addEventListener('change'", customizeHandlerStart);
    expect(pageSource.slice(customizeHandlerStart, customizeHandlerEnd)).not.toContain('validateAndGenerate');
    expect(pageSource).toContain('preview.hidden = false');
    expect(pageSource).toContain('setOutputAvailable(false)');
  });
});
