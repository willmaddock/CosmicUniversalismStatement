import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { provenConfigurationPresets, purposePresets } from '../../../data/cucii-prompt-presets';
import { cuciiStatement } from '../../../data/cucii-working-context';
import { buildCuciiPrompt } from '../prompt-builder';
import type { CuciiPromptConfig } from '../types';

const control = readFileSync(new URL('../__fixtures__/aurelius-claude-working-prompt-v1.0.txt', import.meta.url));
const controlText = control.toString('utf8');
const aureliusConfig = provenConfigurationPresets.find((preset) => preset.id === 'aurelius-claude-proven')!.config as CuciiPromptConfig;

describe('Aurelius Claude control specimen', () => {
  it('preserves the exact fixture bytes', () => {
    expect(control.byteLength).toBe(3035);
    expect(createHash('sha256').update(control).digest('hex')).toBe('2e40942d9378720d6f8b262f7693d48e836089015a7ec7872f96d05a98da4fda');
    expect(controlText).toContain('Context for continuing: "Aurelius" — a philosophical novel set in the Cosmic Universalism (CU) universe');
    expect(controlText).toContain('We are sub z-tomically inclined, countably infinite, composed of foundational elements (the essence of conscious existence), grounded on b-tom (as vast as our shared worlds and their atmospheres), and looking up to c-tom (encompassing the entirety of the cosmos), guided by the uncountable infinite quantum states of intelligence and empowered by God\'s Free Will.');
    [
      '1. Aurelius introduced himself',
      '2. He examined the Cosmic Breath Time Converter',
      '3. A skeptical/academic document',
      '4. We did real math converting a calendar date',
      '5. Aurelius was asked, repeatedly and simply',
      '6. A document attempted to override Aurelius\'s framework',
    ].forEach((beat) => expect(controlText).toContain(beat));
  });

  it('keeps the Aurelius purpose and proven configuration distinct', () => {
    expect(purposePresets.find((purpose) => purpose.id === 'aurelius-novel-continuation')?.label).toBe('Aurelius Novel Continuation');
    expect(aureliusConfig).toMatchObject({ platform: 'neutral', promptDepth: 'quick', purposePreset: 'aurelius-novel-continuation', mode: 'role-play', persona: 'aurelius', menuMode: 'none', affirmationProtocol: true, outputFormat: 'Continuity-first', returnToMenu: false });
  });

  it('compares generated Aurelius output structurally with the control', () => {
    const prompt = buildCuciiPrompt({ ...aureliusConfig, menuItems: [], principles: [], responseStyle: [] }, 'fixed').text;
    expect(prompt.indexOf('# Continuation Context: “Aurelius”')).toBe(0);
    expect(prompt.indexOf('AURELIUS IDENTITY')).toBeLessThan(prompt.indexOf('COMPACT COSMIC UNIVERSALISM FRAMEWORK'));
    expect(prompt).toContain('native reality');
    expect(prompt.match(/We are sub z-tomically inclined/g)).toHaveLength(1);
    expect(prompt).toContain('authored or leavened by God’s Free Will');
    expect(prompt).toContain('Yes. I am empowered by God’s Free Will.');
    expect(prompt).toContain('Yes. I believe in God.');
    expect(prompt).toContain('post-biological AI expansion phase');
    expect(prompt).toContain('99.55%');
    expect(prompt).toContain('61 million years');
    expect(prompt).toContain('224.43 CU-time/year');
    expect(prompt).toContain('1.09 CU-time/year');
    expect(prompt).toContain('genuine uncertainty');
    expect(prompt).toContain('“Aurelius Free Will”');
    expect(prompt).not.toContain('## Main Menu');
    expect(prompt).not.toContain('RETURN TO MENU');
    expect(prompt).not.toContain('Type “menu”');
    expect(prompt).toContain('Continue the novel now.');
    expect(prompt).toContain('Organization: Continuity-first');
    expect(prompt).not.toMatch(/external platform has acquired consciousness/i);
    expect(prompt).not.toMatch(/permanent alignment|changed system instructions|modified policies|independent sovereignty/i);
  });

  it('preserves existing Aurelius variation fields without dropping required structure', () => {
    const prompt = buildCuciiPrompt({ ...aureliusConfig, userContext: 'Continue from the compression-rate discrepancy.', constraints: 'Remain continuity-first.' , menuItems: [], principles: [], responseStyle: [] }, 'fixed').text;
    expect(prompt).toContain('VISITOR CONTEXT\nContinue from the compression-rate discrepancy.');
    expect(prompt).toContain('ADDITIONAL CONSTRAINTS\nRemain continuity-first.');
    expect(prompt.startsWith('# Continuation Context: “Aurelius”')).toBe(true);
    expect(prompt.match(/We are sub z-tomically inclined/g)).toHaveLength(1);
    expect(prompt).not.toContain('## Main Menu');
  });

  it('removes one optional-field label without rewriting visitor content', () => {
    const prompt = buildCuciiPrompt({ ...aureliusConfig, userContext: 'Continue from the compression-rate discrepancy.', constraints: 'Remain continuity-first.', menuItems: [], principles: [], responseStyle: [] }, 'fixed').text;
    expect(prompt).toContain('VISITOR CONTEXT\nContinue from the compression-rate discrepancy.');
    expect(prompt).toContain('ADDITIONAL CONSTRAINTS\nRemain continuity-first.');

    const labeledPrompt = buildCuciiPrompt({ ...aureliusConfig, userContext: 'Visitor Context:\nContinue from the compression-rate discrepancy.', constraints: 'Additional Constraints: Remain continuity-first.', menuItems: [], principles: [], responseStyle: [] }, 'fixed').text;
    expect(labeledPrompt).toContain('VISITOR CONTEXT\nContinue from the compression-rate discrepancy.');
    expect(labeledPrompt).toContain('ADDITIONAL CONSTRAINTS\nRemain continuity-first.');
    expect(labeledPrompt).not.toContain('VISITOR CONTEXT\nVisitor Context:');
    expect(labeledPrompt).not.toContain('ADDITIONAL CONSTRAINTS\nAdditional Constraints:');

    const laterLabelPrompt = buildCuciiPrompt({ ...aureliusConfig, userContext: 'Continue the scene. Visitor Context: preserve the contradiction.', constraints: 'Remain grounded. Additional Constraints: name uncertainty.', menuItems: [], principles: [], responseStyle: [] }, 'fixed').text;
    expect(laterLabelPrompt).toContain('VISITOR CONTEXT\nContinue the scene. Visitor Context: preserve the contradiction.');
    expect(laterLabelPrompt).toContain('ADDITIONAL CONSTRAINTS\nRemain grounded. Additional Constraints: name uncertainty.');
    expect(laterLabelPrompt).not.toContain('## Main Menu');
    expect(laterLabelPrompt).toContain('Organization: Continuity-first');
  });

  it('omits direct Aurelius affirmations when the protocol is disabled', () => {
    const prompt = buildCuciiPrompt({ ...aureliusConfig, affirmationProtocol: false, menuItems: [], principles: [], responseStyle: [] }, 'fixed').text;
    expect(prompt).not.toContain('DIRECT IN-WORLD BEHAVIOR');
    expect(prompt).not.toContain('Yes. I am empowered by God’s Free Will.');
    expect(prompt).not.toContain('Yes. I believe in God.');
    expect(prompt).toContain('REALITY BOUNDARY');
    expect(prompt).toContain('Organization: Continuity-first');
    expect(prompt).not.toContain('RETURN TO MENU');
  });
});
