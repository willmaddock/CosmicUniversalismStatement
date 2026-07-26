import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const frameworkSource = readSource('../../../pages/framework.astro');
const protectedRouteSources = [
  '../../../pages/index.astro',
  '../../../pages/cosmic-breath.astro',
  '../../../pages/media.astro',
  '../../../pages/about.astro',
].map(readSource);

describe('Framework page-system demonstration', () => {
  it('preserves the existing categories and substantive wording', () => {
    for (const category of [
      'Philosophical inquiry',
      'Computational expression',
      'Research discipline',
      'Empirical references',
      'mathematical models',
      'theoretical propositions',
      'open questions',
    ]) {
      expect(frameworkSource).toContain(category);
    }
    expect(frameworkSource).toContain(
      'classification="philosophical-interpretation"',
    );
    expect(frameworkSource).toContain(
      'classification="cu-mathematical-model"',
    );
  });

  it('uses the approved primitives with semantic heading order', () => {
    for (const component of [
      'EditorialSectionHeading',
      'EpistemicCallout',
      'SourceProvenancePanel',
      'ContinuationNavigation',
    ]) {
      expect(frameworkSource).toContain(`<${component}`);
    }
    expect(frameworkSource).toContain(
      'aria-labelledby="framework-categories-title"',
    );
    expect(frameworkSource).toContain(
      'headingId="framework-categories-title"',
    );
    expect(frameworkSource).toContain('headingLevel="h3"');
  });

  it('contains only the approved Framework continuation', () => {
    expect(frameworkSource).toContain(
      'title="Continue to Cosmic Breath"',
    );
    expect(frameworkSource).toContain(
      'description="Explore Cosmic Structure and the TOM framework"',
    );
    expect(frameworkSource).toContain('path="cosmic-breath/"');
    expect(frameworkSource).not.toContain('PageGuide');
    expect(frameworkSource).not.toMatch(/Back[- ]to[- ]top/i);
  });

  it('adds no citation, version, date, script, or Part II material', () => {
    expect(frameworkSource).not.toMatch(
      /\bcitation\b|\bversion\s+\d|\b(?:19|20)\d{2}\b/i,
    );
    expect(frameworkSource).not.toContain('<script');
    expect(frameworkSource).not.toContain('client:');
    expect(frameworkSource).not.toMatch(
      /\babstract\b|expandable preview|document metadata/i,
    );
  });

  it('keeps page-system adoption out of routes not authorized for harmonization', () => {
    protectedRouteSources.forEach((source) => {
      expect(source).not.toContain('/page-system/');
      expect(source).not.toContain('ContinuationNavigation');
      expect(source).not.toContain('EditorialSectionHeading');
      expect(source).not.toContain('EpistemicCallout');
      expect(source).not.toContain('SourceProvenancePanel');
    });
  });
});
