import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const routeSource = readSource('../../../pages/cu-time.astro');
const converterSource = readSource('../../../components/CUTimeConverter.astro');

const routeTargetIds = [
  'cu-time-converter-feature',
  'cu-time-understand-result',
  'cu-time-method-limits',
];

const countLiteralId = (source: string, id: string) =>
  source.match(new RegExp(`id=["']${id}["']`, 'g'))?.length ?? 0;

describe('CU-Time page-system adoption', () => {
  it('keeps the existing converter and its structural IDs exactly once', () => {
    expect(routeSource.match(/<CUTimeConverter \/>/g)).toHaveLength(1);

    for (const id of [
      'cu-time-converter-title',
      'gregorian-to-cu',
      'cu-to-gregorian',
      'gregorian-month',
      'gregorian-error',
      'cu-time-input',
      'cu-time-error',
      'gregorian-result-title',
      'cu-time-result-title',
    ]) {
      expect(countLiteralId(converterSource, id)).toBe(1);
    }
  });

  it('uses unique real guide targets without duplicating converter navigation', () => {
    for (const targetId of routeTargetIds) {
      expect(routeSource).toContain(`targetId: '${targetId}'`);
      expect(countLiteralId(routeSource, targetId)).toBe(1);
    }

    expect(new Set(routeTargetIds).size).toBe(routeTargetIds.length);
    expect(routeSource).not.toContain("targetId: 'gregorian-to-cu'");
    expect(routeSource).not.toContain("targetId: 'cu-to-gregorian'");
    expect(routeSource.indexOf('<PageGuide')).toBeLessThan(
      routeSource.indexOf('<FullWidthFeaturePanel'),
    );
  });

  it('keeps the stable Back to top target unique and route-local', () => {
    expect(countLiteralId(routeSource, 'cu-time-page-top')).toBe(1);
    expect(routeSource).toContain('targetId="cu-time-page-top"');
    expect(routeSource).toContain('revealAfterId="cu-time-converter-feature"');
  });

  it('preserves every substantive explanatory statement', () => {
    for (const statement of [
      'CU-Time connects familiar chronology with the framework’s larger sequence of',
      'This tool implements the documented mathematical constants and calendar rules.',
      'CU-Time is intended to provide consistent notation for positions within the',
      'Gregorian inputs use explicit civil CE/BCE years and UTC.',
      'CU-Time is the framework’s full Cosmic Breath coordinate, spanning from the sub-ZTOM seed boundary',
      'The Observable-Universe–Aligned CU Coordinate is a subordinate CU mathematical reference aligned with',
      'The Observable Universe marker appears between B-TOM and C-TOM because the governing CU Statement',
      'The converter uses deterministic Decimal arithmetic and documented calendar/model assumptions.',
    ]) {
      expect(routeSource).toContain(statement);
    }
  });

  it('uses the accepted editorial, boundary, provenance, and continuation primitives', () => {
    for (const primitive of [
      'EditorialSectionHeading',
      'EpistemicCallout',
      'SourceProvenancePanel',
      'ContinuationNavigation',
    ]) {
      expect(routeSource).toContain(`<${primitive}`);
    }

    expect(routeSource).toContain('kind="boundary"');
    expect(routeSource).toContain('id="cu-time-precision"');
    expect(routeSource).toContain('title="Continue to CU Intelligence"');
    expect(routeSource).toContain(
      'description="Explore CUCII, alignment, and recursive intelligence"',
    );
    expect(routeSource).toContain('path="cu-intelligence/"');
  });

  it('adds no route script, citation metadata, chronology constants, or Part II material', () => {
    expect(routeSource).not.toContain('<script');
    expect(routeSource).not.toContain('client:');
    expect(routeSource).not.toMatch(/\\bcitation\\b|\\bversion\\s+\\d|\\b(?:19|20)\\d{2}\\b/i);
    expect(routeSource).not.toMatch(
      /anchorJdn|daysPerYear|baseCuNasa|nasaUniverseAge|abstract|expandable preview/i,
    );
  });
});
