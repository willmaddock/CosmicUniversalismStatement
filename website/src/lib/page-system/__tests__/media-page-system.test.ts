import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { mediaLibrary } from '../../../data/media/media-library';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const routeSource = readSource('../../../pages/media.astro');
const cardSource = readSource('../../../components/media/MediaCard.astro');
const introSource = readSource('../../../components/PageIntro.astro');
const continuationSource = readSource(
  '../../../components/page-system/ContinuationNavigation.astro',
);

const countLiteralId = (source: string, id: string) =>
  source.match(new RegExp(`id=["']${id}["']`, 'g'))?.length ?? 0;

describe('Media landing-page library', () => {
  it('keeps one PageIntro H1 and three logical editorial H2 sections', () => {
    expect(routeSource.match(/<ContentPageLayout\b/g)).toHaveLength(1);
    expect(introSource.match(/<h1\b/g)).toHaveLength(1);
    expect(routeSource.match(/<EditorialSectionHeading\b/g)).toHaveLength(3);

    for (const [id, title] of [
      ['media-clarity-title', 'Clarity before volume'],
      ['media-library-title', 'Owner-reviewed Media publications'],
      ['media-publication-standard-title', 'Publication standard'],
    ] as const) {
      expect(countLiteralId(routeSource, id)).toBe(0);
      expect(routeSource).toContain(`aria-labelledby="${id}"`);
      expect(routeSource).toContain(`headingId="${id}"`);
      expect(routeSource).toContain(`title="${title}"`);
    }
  });

  it('renders both sealed records through one registry-backed MediaCard map', () => {
    expect(mediaLibrary).toHaveLength(2);
    expect(routeSource).toContain("import { mediaLibrary } from '../data/media/media-library'");
    expect(routeSource).toContain("import MediaCard from '../components/media/MediaCard.astro'");
    expect(routeSource).toContain('mediaLibrary.map((entry) => <MediaCard entry={entry} />)');
    expect(routeSource.match(/<MediaCard\b/g)).toHaveLength(1);

    for (const entry of mediaLibrary) {
      expect(entry.publicationStatus).toBe('published');
      expect(routeSource).not.toContain(entry.youtubeId);
      expect(routeSource).not.toContain(entry.slug);
      expect(routeSource).not.toContain(entry.title);
    }

    expect(mediaLibrary.map(({ classifications }) => classifications.length)).toEqual([
      4,
      2,
    ]);
  });

  it('derives each accessible card from MediaEntry and links to the permanent CU route', () => {
    expect(cardSource).toContain('entry: MediaEntry');
    expect(cardSource).toContain('<article class="media-card">');
    expect(cardSource).toContain('src={sitePath(entry.posterPath)}');
    expect(cardSource).toContain('alt={entry.posterAlt}');
    expect(cardSource).toContain('<h3>{entry.title}</h3>');
    expect(cardSource).toContain('{entry.summary}');
    expect(cardSource).toContain('<ClassificationBadge classification={classification} />');
    expect(cardSource).toContain('href={sitePath(`media/${entry.slug}/`)}');
    expect(cardSource.match(/<a\b/g)).toHaveLength(1);
    expect(cardSource).not.toContain('entry.youtubeUrl');
    expect(cardSource).not.toMatch(/<iframe\b|<video\b|<script\b|client:/);
  });

  it('shows the published records truthfully without unsupported release claims', () => {
    expect(routeSource).toContain('Owner-reviewed Media publications');
    expect(routeSource).toContain(
      'These owner-reviewed publications are part of the Cosmic Universalism Media library. Each entry links to a permanent CU detail page with classification, transcript, accessibility resources, source provenance, and publication boundaries.',
    );
    expect(routeSource).toContain('Publication remains deliberate');
    expect(routeSource).not.toMatch(
      /feature branch|under review|staging status|staging records|staging is not release/i,
    );
    expect(cardSource).toContain("entry.publicationStatus === 'staging'");
    expect(cardSource).toContain('Staging review');
    expect(cardSource).toContain('not publicly released');
    expect(`${routeSource}\n${cardSource}`).not.toMatch(
      /available now|now published|latest|newest|new release|officially verified|view count|views\b/i,
    );
  });

  it('preserves deliberate publication and the empirical/CU distinction', () => {
    expect(routeSource).toContain('label="Deliberate publication"');
    expect(routeSource).toContain('YouTube delivers video; the CU website retains');
    expect(routeSource).toContain(
      'Established scientific material is sourced separately from CU interpretation.',
    );
    expect(routeSource).toContain(
      'CU mathematics and theoretical propositions retain their canonical classifications',
    );
    expect(routeSource).toContain('visibly labeled as open questions');
  });

  it('keeps the exact base-path-safe About continuation last', () => {
    expect(routeSource.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    expect(routeSource).toContain('ariaLabel="Continue from Media"');
    expect(routeSource).toContain('title="Continue to About"');
    expect(routeSource).toContain('path="about/"');
    expect(continuationSource).toContain('<a href={sitePath(path)}>');

    const positions = [
      routeSource.indexOf('id="media-clarity"'),
      routeSource.indexOf('id="media-library"'),
      routeSource.indexOf('id="media-publication-standard"'),
      routeSource.indexOf('<ContinuationNavigation'),
    ];
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });

  it('remains server-rendered without landing-page media or navigation scripts', () => {
    expect(routeSource).not.toContain('<script');
    expect(routeSource).not.toContain('client:');
    expect(routeSource).not.toContain('<BackToTop');
    expect(routeSource).not.toContain('<PageGuide');
    expect(routeSource).not.toContain('<FullWidthFeaturePanel');
    expect(routeSource).not.toMatch(/<iframe\b|<video\b|youtube-nocookie|youtube\.com\/embed/i);
    expect(`${routeSource}\n${cardSource}`).not.toMatch(
      /fetch\(|XMLHttpRequest|youtube.*api|synchroni[sz]|metadata cache|polling/i,
    );
  });

  it('uses a contained responsive grid with visible focus and no carousel architecture', () => {
    expect(routeSource).toContain('grid-template-columns: repeat(2, minmax(0, 1fr))');
    expect(routeSource).toContain('@media (max-width: 48rem)');
    expect(routeSource).toContain('@media (max-width: 36rem)');
    expect(routeSource).toContain('grid-template-columns: minmax(0, 1fr)');
    expect(cardSource).toContain('min-width: 0');
    expect(cardSource).toContain('width: 100%');
    expect(cardSource).toContain('.media-card__detail-link:focus-visible');
    expect(`${routeSource}\n${cardSource}`).not.toMatch(
      /overflow-x:\s*(?:auto|scroll)|position:\s*sticky|carousel|type=["']search["']|searchParams|data-filter|filter-control/i,
    );
  });

  it('adds no fabricated provider, release, popularity, or chronology metadata', () => {
    expect(`${routeSource}\n${cardSource}`).not.toMatch(
      /view count|views\b|subscriber|popularity|trending|upload date|publication date|release date|issue number|countdown/i,
    );
    expect(cardSource).not.toMatch(/Date\(|new Date|sort\(/);
  });
});
