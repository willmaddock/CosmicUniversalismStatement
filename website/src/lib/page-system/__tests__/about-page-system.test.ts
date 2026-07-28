import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const routeSource = readSource('../../../pages/about.astro');
const introSource = readSource('../../../components/PageIntro.astro');
const continuationSource = readSource(
  '../../../components/page-system/ContinuationNavigation.astro',
);
const provenanceSource = readSource(
  '../../../components/page-system/SourceProvenancePanel.astro',
);

const countLiteralId = (source: string, id: string) =>
  source.match(new RegExp(`id=["']${id}["']`, 'g'))?.length ?? 0;

describe('About page-system adoption', () => {
  it('keeps one PageIntro H1 and a valid visible H2 hierarchy', () => {
    expect(routeSource.match(/<ContentPageLayout\b/g)).toHaveLength(1);
    expect(introSource.match(/<h1\b/g)).toHaveLength(1);
    expect(routeSource.match(/<EditorialSectionHeading\b/g)).toHaveLength(2);
    expect(routeSource.match(/<SourceProvenancePanel\b/g)).toHaveLength(1);
    expect(routeSource).not.toContain('headingLevel="h3"');

    for (const [id, title] of [
      ['about-purpose-title', 'Purpose'],
      ['about-public-documentation-title', 'Public documentation'],
      ['about-revision-critique-title', 'Revision and critique'],
    ] as const) {
      expect(countLiteralId(routeSource, id)).toBe(0);
      expect(routeSource).toContain(`headingId="${id}"`);
      expect(routeSource).toContain(`title="${title}"`);
    }

    expect(routeSource).toContain('aria-labelledby="about-purpose-title"');
    expect(routeSource).toContain(
      'aria-labelledby="about-public-documentation-title"',
    );
    expect(provenanceSource).toContain('aria-labelledby={headingId}');
  });

  it('preserves every existing substantive sentence and route metadata', () => {
    for (const statement of [
      'About the Cosmic Universalism Computational Intelligence Initiative.',
      'An open framework built for examination',
      'The Cosmic Universalism Computational Intelligence Initiative develops and documents the framework in public, inviting careful reading, critique, and continued research.',
      'The project explores how philosophical ideas, computational formalisms, and',
      'cosmological questions can be expressed together without obscuring the',
      'difference between evidence and speculation.',
      'The website is a guide to that work: what the framework proposes, what it',
      'references, how its models are described, and which questions remain open.',
      'The initiative presents the framework for examination rather than as a completed',
      'scientific account. Public pages should make assumptions, classifications,',
      'limitations, and unresolved questions easier to locate.',
      'Critique is part of the documentation process. When an interpretation changes,',
      'the project should preserve enough provenance to show what changed, why it changed,',
      'and whether earlier material is historical, superseded, or archived.',
    ]) {
      expect(routeSource).toContain(statement);
    }

    expect(routeSource).toContain('metaTitle="About | Cosmic Universalism"');
    expect(routeSource).toContain('eyebrow="About"');
  });

  it('keeps the established project identity without inventing personal authorship', () => {
    expect(routeSource).toContain('eyebrow="Project identity"');
    expect(routeSource).toContain(
      'The Cosmic Universalism Computational Intelligence Initiative',
    );
    expect(routeSource).not.toMatch(
      /\b(?:degree|PhD|professor|employment|employer|affiliation|endorsed by|team member|staff|funder|sponsor|collaborator|legal entity)\b/i,
    );
    expect(routeSource).not.toMatch(
      /<img\b|<picture\b|portrait|avatar|résumé|biography|curriculum vitae/i,
    );
  });

  it('keeps qualified commitments visible and does not turn them into guarantees', () => {
    expect(routeSource).toContain('eyebrow="Public commitment"');
    expect(routeSource).toContain('kind="boundary" label="Framework boundary"');
    expect(routeSource).toContain('rather than as a completed');
    expect(routeSource).toContain('Public pages should make');
    expect(routeSource).toContain('the project should preserve');
    expect(routeSource).not.toMatch(
      /\bguarantee(?:d|s)?\b|\bensure(?:d|s)?\b|\balways\b|\bpermanent(?:ly)?\b/i,
    );
    expect(routeSource).not.toMatch(
      /\bprivacy policy\b|\bsecurity policy\b|\bmoderation policy\b|\bcompliance\b|\baudit(?:ed|ing)?\b|\bbinding governance\b/i,
    );
  });

  it('uses only established revision and provenance content with no invented source action', () => {
    expect(routeSource).toContain('eyebrow="Revision record"');
    expect(routeSource).toContain('enough provenance to show what changed');
    expect(routeSource).not.toContain('links={');
    expect(provenanceSource).toContain('links = []');
    expect(routeSource).not.toMatch(
      /repository version|release date|canonical digest|publication history|review record|contributor list|authority date|citation system/i,
    );
  });

  it('preserves the established section IDs and semantic source order', () => {
    const positions = [
      routeSource.indexOf('id="about-purpose"'),
      routeSource.indexOf('id="about-public-documentation"'),
      routeSource.indexOf('id="about-revision-critique"'),
      routeSource.indexOf('<ContinuationNavigation'),
    ];

    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(routeSource.match(/id="about-purpose"/g)).toHaveLength(1);
    expect(routeSource.match(/id="about-public-documentation"/g)).toHaveLength(1);
    expect(routeSource.match(/id="about-revision-critique"/g)).toHaveLength(1);
  });

  it('adds exactly one deliberate semantic, base-path-safe return to Home', () => {
    expect(routeSource.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    expect(routeSource).toContain('ariaLabel="Return from About to Home"');
    expect(routeSource).toContain('title="Return to Home"');
    expect(routeSource).toContain(
      'description="Revisit the Cosmic Universalism overview and choose another path"',
    );
    expect(routeSource).toContain('path="/"');
    expect(continuationSource).toContain('<a href={sitePath(path)}>');
    expect(continuationSource).not.toContain('target=');
    expect(routeSource).not.toMatch(/<nav\b/);
  });

  it('keeps all core content server-rendered without duplicate route controls', () => {
    expect(routeSource).not.toContain('<PageGuide');
    expect(routeSource).not.toContain('<BackToTop');
    expect(routeSource).not.toContain('<script');
    expect(routeSource).not.toContain('client:');
    expect(routeSource).not.toMatch(/aria-live|role=["']status["']/);
    expect(routeSource).not.toMatch(/<button\b|<form\b/);
  });

  it('retains phone stacking, contained text, and visible descriptive actions', () => {
    expect(routeSource).toContain('@media (max-width: 48rem)');
    expect(routeSource).toContain('@media (max-width: 36rem)');
    expect(routeSource).toContain('grid-template-columns: minmax(0, 1fr)');
    expect(routeSource).toContain('min-width: 0');
    expect(provenanceSource).toContain('overflow-wrap: anywhere');
    expect(routeSource).not.toMatch(/overflow-x:\s*(?:auto|scroll)/);
    expect(routeSource).not.toMatch(/carousel|position:\s*sticky/i);
  });

  it('adds no contact, promotional, institutional, authority, or Part II material', () => {
    expect(routeSource).not.toMatch(
      /\bcontact\b|\bemail\b|\bnewsletter\b|\bdonation\b|\bdonate\b|\bsocial account\b|\btestimonial\b|\brating\b|\baward\b/i,
    );
    expect(routeSource).not.toMatch(
      /\bpeer review\b|\bscientific consensus\b|\binstitutional approval\b|\blegal certification\b|\bPart II\b/i,
    );
    expect(routeSource).not.toMatch(
      /\bcitation\b|\bversion\s+\d|\b(?:19|20)\d{2}\b|\bpublication record\b|\breview status\b/i,
    );
    expect(routeSource).not.toMatch(
      /\bcesium-133\b|\bPlanck-time\b|\bNASA offset\b|\bnew research conclusion\b/i,
    );
  });
});
