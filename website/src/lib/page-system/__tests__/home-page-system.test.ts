import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const digest = (source: string) =>
  createHash('sha256').update(source).digest('hex');

const routeSource = readSource('../../../pages/index.astro');
const heroSource = readSource('../../../components/Hero.astro');
const statementSource = readSource('../../../components/StatementPreview.astro');
const researchSource = readSource('../../../components/ResearchPreview.astro');
const timeSource = readSource('../../../components/CUTimeInvitation.astro');
const questionsSource = readSource('../../../components/OpenQuestionsPreview.astro');

const protectedDigests = new Map([
  [
    '../../../components/CosmicBreathDiagram.astro',
    '2750ad7f57cdfa875de17562e7d7c065b5812b9b930899d5b9d943625d7c88b1',
  ],
  [
    '../../../components/BreathSummary.astro',
    'f03395571ba444b80446f585778ad230149de505547e78d37c7f571eac01c074',
  ],
  [
    '../../../components/ResearchLegend.astro',
    'cab598962c437a226a99b1c1fe1d3355919c8571a7e56d3ea7623474d51530a2',
  ],
] as const);

describe('Home page-system adoption', () => {
  it('preserves the distinctive Home hero, lead, and established actions', () => {
    expect(routeSource.match(/<Hero \/>/g)).toHaveLength(1);
    expect(heroSource.match(/<h1\b/g)).toHaveLength(1);
    expect(heroSource).toContain('<span class="hero__title-line">COSMIC</span>');
    expect(heroSource).toContain(
      '<span class="hero__title-line">UNIVERSALISM</span>',
    );
    expect(heroSource).toContain(
      'A philosophical and computational framework exploring consciousness, time,',
    );
    expect(heroSource).toContain('href={sitePath("framework")}');
    expect(heroSource).toContain('Explore the Framework');
    expect(heroSource).toContain(
      'href="https://github.com/willmaddock/CosmicUniversalismStatement/blob/main/README.md"',
    );
    expect(heroSource).toContain('Open the CU Statement');
    expect(heroSource).toContain('target="_blank"');
    expect(heroSource).toContain('rel="noopener noreferrer"');
  });

  it('keeps one complete server-rendered editorial journey in semantic source order', () => {
    for (const component of [
      'StatementPreview',
      'ResearchPreview',
      'CUTimeInvitation',
      'OpenQuestionsPreview',
    ]) {
      expect(routeSource.match(new RegExp(`<${component} \\/>`, 'g'))).toHaveLength(1);
    }

    const positions = [
      routeSource.indexOf('<Hero />'),
      routeSource.indexOf('<StatementPreview />'),
      routeSource.indexOf('<ResearchPreview />'),
      routeSource.indexOf('<CUTimeInvitation />'),
      routeSource.indexOf('<OpenQuestionsPreview />'),
      routeSource.indexOf('<ContinuationNavigation'),
    ];
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(positions.every((position) => position >= 0)).toBe(true);

    expect(routeSource).not.toContain('<script');
    expect(routeSource).not.toContain('client:');
    expect(routeSource).not.toContain('<PageGuide');
    expect(routeSource).not.toContain('<BackToTop');
  });

  it('keeps a valid heading hierarchy and all visible section headings', () => {
    expect(heroSource.match(/<h1\b/g)).toHaveLength(1);
    for (const source of [
      statementSource,
      researchSource,
      timeSource,
      questionsSource,
    ]) {
      expect(source).toContain('<h2');
      expect(source).not.toContain('<h1');
    }
    expect(researchSource.match(/<h3\b/g)).toHaveLength(2);
  });

  it('uses exactly one intact principal visual feature without false controls', () => {
    expect(heroSource.match(/<FullWidthFeaturePanel\b/g)).toHaveLength(1);
    expect(heroSource.match(/<CosmicBreathDiagram \/>/g)).toHaveLength(1);
    expect(heroSource.match(/<BreathSummary \/>/g)).toHaveLength(1);
    expect(heroSource.match(/<ResearchLegend \/>/g)).toHaveLength(1);
    expect(heroSource).toContain(
      'ariaLabel="Cosmic Breath structural overview"',
    );
    expect(heroSource).not.toMatch(/<button\b|role=["']button["']/);
  });

  it('preserves the diagram, summary, and classification vocabulary byte-for-byte', () => {
    for (const [relativePath, expectedDigest] of protectedDigests) {
      expect(digest(readSource(relativePath))).toBe(expectedDigest);
    }
  });

  it('reuses the established epistemic distinction with a textual label', () => {
    expect(heroSource.match(/<EpistemicCallout\b/g)).toHaveLength(1);
    expect(heroSource).toContain('kind="boundary"');
    expect(heroSource).toContain('label="Research boundary"');
    expect(heroSource).toContain(
      'Scientific references distinguished from CU theoretical propositions.',
    );
  });

  it('preserves all established Home destinations and base-path helpers', () => {
    expect(heroSource).toContain('href={sitePath("framework")}');
    expect(statementSource).toContain('href={sitePath("framework")}');
    expect(researchSource).toContain('href={sitePath("research")}');
    expect(timeSource).toContain('href={sitePath("cu-time")}');
    expect(questionsSource).toContain(
      'href={sitePath("research#open-questions")}',
    );
    expect(routeSource).toContain('path="framework/"');
  });

  it('adds one exact same-tab continuation to Framework', () => {
    expect(routeSource.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    expect(routeSource).toContain('ariaLabel="Continue from Home"');
    expect(routeSource).toContain('title="Continue to Framework"');
    expect(routeSource).toContain(
      'description="Explore the philosophical, mathematical, and empirical foundations"',
    );
    expect(routeSource).toContain('path="framework/"');
  });

  it('retains responsive stacking, focus treatment, and complete no-script content', () => {
    expect(heroSource).toContain('.hero__link:focus-visible');
    expect(heroSource).toContain('@media (max-width: 36rem)');
    expect(routeSource).toContain('@media (max-width: 36rem)');
    expect(routeSource).toContain('@media (prefers-reduced-motion: reduce)');
    expect(researchSource).toContain('grid-template-columns: 1fr');
    expect(questionsSource).toContain('grid-template-columns: 1fr');
    expect(routeSource).not.toMatch(/overflow-x:\s*(?:auto|scroll)/);
    expect(routeSource).not.toMatch(/carousel/i);
  });

  it('adds no publication metadata, authority claim, or Part II content', () => {
    const combinedSource = `${routeSource}\n${heroSource}`;
    expect(combinedSource).not.toMatch(
      /\bcitation\b|\bversion\s+\d|\b(?:19|20)\d{2}\b/i,
    );
    expect(combinedSource).not.toMatch(
      /\bnew research finding\b|\bscientific conclusion\b|\bPart II\b/i,
    );
    expect(combinedSource).not.toMatch(/PageGuide|BackToTop|SourceProvenancePanel/);
  });
});
