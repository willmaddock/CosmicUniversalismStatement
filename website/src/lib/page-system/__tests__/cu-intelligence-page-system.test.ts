import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const routeSource = readSource('../../../pages/cu-intelligence.astro');
const conversationCardSource = readSource(
  '../../../components/cucii/ConversationCard.astro',
);
const scriptSource = routeSource.match(/<script>\n([\s\S]*?)\n<\/script>/)?.[1];

const countLiteralId = (source: string, id: string) =>
  source.match(new RegExp(`id=["']${id}["']`, 'g'))?.length ?? 0;

describe('CU Intelligence page-system adoption', () => {
  it('keeps the accessibility-corrected studio script at its reviewed digest', () => {
    expect(scriptSource).toBeDefined();
    expect(
      createHash('sha256').update(scriptSource ?? '').digest('hex'),
    ).toBe('156dcde83296bb36111fb08d91182dea0d7bd952549e48e770763c4d03a42b84');
  });

  it('uses unique real guide targets around the intact studio', () => {
    expect(routeSource.match(/<PageGuide\b/g)).toHaveLength(1);
    expect(routeSource.match(/<FullWidthFeaturePanel\b/g)).toHaveLength(1);

    for (const targetId of [
      'cucii-orientation',
      'cucii-living-context',
      'prompt-studio-feature',
      'cucii-method-sources',
    ]) {
      expect(routeSource).toContain(`targetId: '${targetId}'`);
      expect(countLiteralId(routeSource, targetId)).toBe(1);
    }

    expect(countLiteralId(routeSource, 'prompt-studio')).toBe(1);
    expect(
      routeSource.match(
        /<FullWidthFeaturePanel[^>]*>[\s\S]*?<section[\s\S]*?id="prompt-studio"[\s\S]*?<\/FullWidthFeaturePanel>/g,
      ),
    ).toHaveLength(1);
    expect(routeSource.indexOf('<PageGuide')).toBeLessThan(
      routeSource.indexOf('<FullWidthFeaturePanel'),
    );
    expect(routeSource).toContain(
      '<FullWidthFeaturePanel id="prompt-studio-feature" ariaLabel="Explore CUCII Prompt Studio">',
    );
    expect(routeSource).toContain(
      ':global(#prompt-studio-feature.full-width-feature-panel)',
    );
    expect(routeSource).toContain('grid-template-columns: minmax(0, 1fr)');
    expect(routeSource).toContain('justify-self: center');
    expect(routeSource).toContain('margin-left: 0');
    expect(routeSource).toContain('transform: none');
  });

  it('preserves every studio control and interaction hook', () => {
    for (const id of [
      'add-menu-item',
      'alignment-principles',
      'clean-context',
      'copy-prompt',
      'cucii-constraints',
      'cucii-context',
      'cucii-custom-persona',
      'cucii-custom-purpose',
      'cucii-menu-mode',
      'cucii-menu-preset',
      'cucii-mode',
      'cucii-output-format',
      'cucii-persona',
      'cucii-platform',
      'cucii-prompt-depth',
      'cucii-purpose',
      'cucii-source-pack',
      'cucii-studio-form',
      'custom-menu-builder',
      'custom-menu-items',
      'download-md',
      'download-txt',
      'living-conversations',
      'menu-count',
      'menu-status',
      'method-and-limits',
      'mode-guidance',
      'next-step-guidance',
      'persona-field',
      'platform-links-title',
      'prompt-platform',
      'studio-errors',
      'prompt-preview',
      'prompt-preview-title',
      'prompt-studio',
      'prompt-studio-title',
      'prompt-text',
      'proven-configuration-status',
      'proven-configurations',
      'proven-configurations-title',
      'purpose-guidance',
      'return-to-menu-field',
      'role-play-guidance',
      'studio-guide',
      'studio-settings',
      'studio-settings-title',
      'studio-status',
      'usage-guidance',
      'usage-guidance-title',
      'what-cucii-means',
    ]) {
      expect(countLiteralId(routeSource, id)).toBe(1);
    }

    for (const name of [
      'affirmationProtocol',
      'constraints',
      'customPersona',
      'customPurpose',
      'menuMode',
      'menuPreset',
      'mode',
      'outputFormat',
      'persona',
      'platform',
      'principles',
      'promptDepth',
      'purpose',
      'responseStyle',
      'returnToMenu',
      'sourcePack',
      'userContext',
    ]) {
      expect(routeSource).toContain(`name="${name}"`);
    }

    expect(routeSource.match(/<fieldset>/g)).toHaveLength(4);
    expect(routeSource).toContain('role="alert" aria-live="assertive"');
    expect(routeSource).toContain('role="status" aria-live="polite"');
    for (const dataAttribute of [
      'data-action',
      'data-details-toggle',
      'data-field',
      'data-preset-card',
      'data-preset-state',
      'data-proven-preset',
    ]) {
      expect(routeSource).toContain(dataAttribute);
    }

    for (const focusTarget of [
      'previewTitle?.focus()',
      'studioSettingsTitle?.focus()',
      'invalidControl(issues[0])?.focus()',
    ]) {
      expect(routeSource).toContain(focusTarget);
    }

    for (const controlText of [
      'Generate and Review Prompt',
      'Reset Studio',
      'Copy Full Operating Prompt',
      'Download .txt',
      'Download .md',
      'Prompt generated locally. Review it before copying or downloading.',
      'Prompt copied. Open the selected platform and paste it as your first message.',
      'Copy was unavailable; the prompt is selected for manual copying.',
    ]) {
      expect(routeSource).toContain(controlText);
    }
  });

  it('synchronizes every conditional label with its associated field', () => {
    for (const [label, id] of [
      ['Custom purpose', 'cucii-custom-purpose'],
      ['Custom persona', 'cucii-custom-persona'],
      ['Preset menu', 'cucii-menu-preset'],
    ]) {
      expect(routeSource).toContain(`<label for="${id}" hidden>${label}</label>`);
      expect(countLiteralId(routeSource, id)).toBe(1);
    }

    expect(routeSource).toContain(
      'form.querySelector<HTMLLabelElement>(`label[for="${element.id}"]`)',
    );
    expect(routeSource).toContain(
      "if (label) { label.hidden = !show; label.setAttribute('aria-hidden', String(!show)); }",
    );
    expect(routeSource).toContain(
      'element.querySelectorAll<HTMLElement>(invalidSelector)',
    );
  });

  it('uses explicit deterministic focus targets for every validated static field', () => {
    for (const mapping of [
      "platform: '#cucii-platform'",
      "sourcePack: '#cucii-source-pack'",
      "promptDepth: '#cucii-prompt-depth'",
      "purpose: '#cucii-purpose'",
      "purposePreset: '#cucii-purpose'",
      "customPurpose: '#cucii-custom-purpose'",
      "mode: '#cucii-mode'",
      "persona: '#cucii-persona'",
      "customPersona: '#cucii-custom-persona'",
      "menuMode: '#cucii-menu-mode'",
      "menuPreset: '#cucii-menu-preset'",
      "outputFormat: '#cucii-output-format'",
    ]) {
      expect(routeSource).toContain(mapping);
    }

    expect(routeSource).toContain(
      "field !== 'menuItems' && !field.startsWith('menuItems.')",
    );
    expect(routeSource).toContain(
      '.menu-item-editor[data-index="${Number.isFinite(index) ? index : 0}"]',
    );
    expect(routeSource).not.toContain('`#cucii-${field}`');
    expect(routeSource).not.toContain('#cucii-customPurpose');
  });

  it('applies and clears aria-invalid from existing validation outcomes', () => {
    expect(routeSource).toContain(
      "const invalidSelector = '[aria-invalid=\"true\"]'",
    );
    expect(routeSource).toContain(
      'form.querySelectorAll<HTMLElement>(invalidSelector)',
    );
    expect(routeSource).toContain(
      "control.removeAttribute('aria-invalid')",
    );
    expect(routeSource).toContain(
      "invalidControl(issue)?.setAttribute('aria-invalid', 'true')",
    );
    expect(routeSource).toContain('applyInvalid(validation.errors)');
    expect(routeSource).toContain('focusInvalid(validation.errors)');
    expect(routeSource).toContain(
      'clearInvalid(); renderErrors([]); generated = buildCuciiPrompt',
    );
    expect(routeSource).toContain(
      'renderMenu(); showConditional(); clearInvalid();',
    );
  });

  it('preserves the approved CUCII meaning, claims boundary, and local-processing copy', () => {
    for (const statement of [
      'CUCII provides structured prompt-level context for a conversation.',
      'these prompts do not retrain, permanently change, unlock, certify, or override an external model',
      'These six project-author references are ordinary external links',
      'The Prompt Studio creates the prompt locally.',
      'Prompt settings are processed locally in this page.',
      'does not send entries to a Cosmic Universalism server or AI API',
    ]) {
      expect(routeSource).toContain(statement);
    }

    expect(routeSource).toContain('kind="boundary"');
    expect(routeSource).toContain('label="AI capability boundary"');
    expect(routeSource).toContain('<SourceProvenancePanel');
  });

  it('uses coherent editorial headings and presentation-only card heading control', () => {
    expect(routeSource).toContain('<EditorialSectionHeading');
    expect(routeSource).toContain('headingLevel="h3"');
    expect(routeSource).toContain('headingLevel="h4"');
    expect(conversationCardSource).toContain("headingLevel?: 'h3' | 'h4'");
    expect(conversationCardSource).toContain('<Heading>{conversation.title}</Heading>');
  });

  it('adds the stable Back to top target and exact Research continuation', () => {
    expect(countLiteralId(routeSource, 'cucii-page-top')).toBe(1);
    expect(routeSource).toContain('targetId="cucii-page-top"');
    expect(routeSource).toContain('revealAfterId="prompt-studio-feature"');
    expect(routeSource).toContain('title="Continue to Research"');
    expect(routeSource).toContain(
      'description="Explore the CU Research Observatory, open questions, and governing sources"',
    );
    expect(routeSource).toContain('path="research/"');
  });

  it('does not add prohibited capability claims or Part II material', () => {
    expect(routeSource).not.toMatch(
      /AI consciousness|divine authority|permanent alignment|supernatural memory|direct access to God|cosmic hologram|empirical proof of cosmic access/i,
    );
    expect(routeSource).not.toMatch(
      /cesium-133|Planck-time|NASA-offset|Part II chronology/i,
    );
  });
});
