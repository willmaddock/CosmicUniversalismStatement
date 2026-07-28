import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const routeSource = readSource('../../../pages/media.astro');
const introSource = readSource('../../../components/PageIntro.astro');
const continuationSource = readSource(
  '../../../components/page-system/ContinuationNavigation.astro',
);

const countLiteralId = (source: string, id: string) =>
  source.match(new RegExp(`id=["']${id}["']`, 'g'))?.length ?? 0;

describe('Media page-system adoption', () => {
  it('keeps one PageIntro H1 and three visible editorial H2 sections', () => {
    expect(routeSource.match(/<ContentPageLayout\b/g)).toHaveLength(1);
    expect(introSource.match(/<h1\b/g)).toHaveLength(1);
    expect(routeSource.match(/<EditorialSectionHeading\b/g)).toHaveLength(3);
    expect(routeSource).not.toContain('headingLevel="h3"');

    for (const [id, title] of [
      ['media-clarity-title', 'Clarity before volume'],
      ['media-planned-formats-title', 'Planned formats'],
      ['media-publication-standard-title', 'Publication standard'],
    ] as const) {
      expect(countLiteralId(routeSource, id)).toBe(0);
      expect(routeSource).toContain(`aria-labelledby="${id}"`);
      expect(routeSource).toContain(`headingId="${id}"`);
      expect(routeSource).toContain(`title="${title}"`);
    }
  });

  it('preserves every existing substantive sentence and route metadata', () => {
    for (const statement of [
      'An introduction to future Cosmic Universalism media and explanatory resources.',
      'Different ways into a large framework',
      'The media library will gather accessible explanations, visual narratives, and conversations about Cosmic Universalism.',
      'Future media will identify whether it explains source science, CU mathematics,',
      'a theoretical proposition, or an open question. This foundation keeps those',
      'categories visible before the library grows.',
      'Future entries may include written explainers, diagram walkthroughs, recorded',
      'conversations, and other educational formats. Each entry should state its subject,',
      'classification, and source basis before asking the audience to evaluate its',
      'conclusions.',
      'Media that discusses established scientific material should identify the external',
      'source separately from any CU interpretation. Media that presents CU mathematics',
      'or theoretical propositions should name that classification directly. Unresolved',
      'issues should remain visibly labeled as open questions.',
    ]) {
      expect(routeSource).toContain(statement);
    }

    expect(routeSource).toContain('metaTitle="Media | Cosmic Universalism"');
    expect(routeSource).toContain('eyebrow="Media"');
  });

  it('states the honest forthcoming and foundation status in visible text', () => {
    expect(routeSource.match(/eyebrow="Foundation"/g)).toHaveLength(2);
    expect(routeSource.match(/Forthcoming/g)).toHaveLength(2);
    expect(routeSource.match(/<EpistemicCallout\b/g)).toHaveLength(1);
    expect(routeSource).toContain('kind="context" label="Forthcoming"');
    expect(routeSource).toContain('Future media will identify');
    expect(routeSource).toContain('Future entries may include');
    expect(routeSource).not.toContain('Available now');
    expect(routeSource).not.toContain('In development');
  });

  it('renders no fabricated catalog, item, media control, or unavailable action', () => {
    expect(routeSource).not.toMatch(
      /<article\b|<img\b|<picture\b|<video\b|<audio\b|<button\b|<form\b/,
    );
    expect(routeSource).not.toMatch(
      /catalog|thumbnail|cover art|subscribe|newsletter|feed|download|release date|issue number|progress|countdown/i,
    );
    expect(routeSource).not.toMatch(/disabled|aria-disabled|role=["']button["']/);
    expect(routeSource).not.toContain('<FullWidthFeaturePanel');
    expect(routeSource).not.toContain('<SourceProvenancePanel');
  });

  it('adds one exact semantic, base-path-safe About continuation', () => {
    expect(routeSource.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    expect(routeSource).toContain('ariaLabel="Continue from Media"');
    expect(routeSource).toContain('title="Continue to About"');
    expect(routeSource).toContain(
      'description="Learn about the project, its authorship, and its public commitments"',
    );
    expect(routeSource).toContain('path="about/"');
    expect(continuationSource).toContain('<a href={sitePath(path)}>');
    expect(continuationSource).not.toContain('target=');
    expect(continuationSource).toContain('Forward →');
  });

  it('keeps the established section order and continuation last', () => {
    const positions = [
      routeSource.indexOf('id="media-clarity"'),
      routeSource.indexOf('id="media-planned-formats"'),
      routeSource.indexOf('id="media-publication-standard"'),
      routeSource.indexOf('<ContinuationNavigation'),
    ];

    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });

  it('keeps core content server-rendered without guide, Back to top, or route script', () => {
    expect(routeSource).not.toContain('<PageGuide');
    expect(routeSource).not.toContain('<BackToTop');
    expect(routeSource).not.toContain('<script');
    expect(routeSource).not.toContain('client:');
    expect(routeSource).not.toMatch(/aria-live|role=["']status["']/);
  });

  it('retains phone stacking, contained panels, and descriptive actions', () => {
    expect(routeSource).toContain('@media (max-width: 48rem)');
    expect(routeSource).toContain('@media (max-width: 36rem)');
    expect(routeSource).toContain(
      'grid-template-columns: minmax(0, 1fr)',
    );
    expect(routeSource).toContain('min-width: 0');
    expect(routeSource).not.toMatch(/overflow-x:\s*(?:auto|scroll)/);
    expect(routeSource).not.toMatch(/carousel|position:\s*sticky/i);
  });

  it('adds no invented publication, authority, or Part II material', () => {
    expect(routeSource).not.toMatch(
      /\bpodcast\b|\binterview\b|\bpress coverage\b|\bmedia partner\b|\bpeer review\b|\binstitutional affiliation\b/i,
    );
    expect(routeSource).not.toMatch(
      /\bcitation\b|\bversion\s+\d|\b(?:19|20)\d{2}\b|\bPart II\b/i,
    );
    expect(routeSource).not.toMatch(
      /\bnew research finding\b|\bscientific conclusion\b|\bauthority claim\b/i,
    );
  });
});
