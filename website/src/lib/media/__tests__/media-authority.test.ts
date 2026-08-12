import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { mediaLibrary } from '../../../data/media/media-library';
import { allResearchClassifications } from '../../../data/research-taxonomy';

const publicRoot = fileURLToPath(new URL('../../../../public/', import.meta.url));
const assetPath = (path: string) => `${publicRoot}${path}`;
const sha256 = (path: string) =>
  createHash('sha256').update(readFileSync(assetPath(path))).digest('hex');
const embedSource = readFileSync(
  new URL('../../../components/media/MediaEmbed.astro', import.meta.url),
  'utf8',
);
const detailRouteSource = readFileSync(
  new URL('../../../pages/media/[slug].astro', import.meta.url),
  'utf8',
);

const expectedAssetHashes = new Map([
  [
    'media/cu-human-moment-2025-05-12/thumbnail.png',
    'd84c903b1d57993e84fcd1d0b4ba4ce5c916ba3351954cbab8deae703630aa97',
  ],
  [
    'media/cu-human-moment-2025-05-12/transcript-en.txt',
    '3579cfe940dab349fae94782b9c5691961f33fce60278e5281546ec329629fd0',
  ],
  [
    'media/cu-human-moment-2025-05-12/captions-en.vtt',
    'ad50462a0dd5f159c9d06254ff8230b352ea00eac4a26c9d6d55d5e81a8a99a0',
  ],
  [
    'media/ai-alignment-through-cucii/thumbnail.png',
    'ac4261552772f09084e2ecb34f889019f9fc9eb9734834b750ad7703a1a91807',
  ],
  [
    'media/ai-alignment-through-cucii/transcript-en.txt',
    '270ba37358c714374ab811728678f3ae945b273f6603b442f747c9b605d10a04',
  ],
  [
    'media/ai-alignment-through-cucii/captions-en.vtt',
    '29bd2bde07ed50c2432342ed44b293158c13507ac08b756faf087837aa671075',
  ],
] as const);

const cueRanges = (source: string) =>
  [...source.matchAll(/^(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})$/gm)]
    .map((match) => [match[1], match[2]] as const);

describe('owner-curated Media authority', () => {
  it('contains only the two sealed staging records with unique identities', () => {
    expect(mediaLibrary).toHaveLength(2);
    expect(mediaLibrary.map(({ id }) => id)).toEqual([
      'cu-human-moment-2025-05-12',
      'ai-alignment-through-cucii',
    ]);
    expect(new Set(mediaLibrary.map(({ id }) => id)).size).toBe(2);
    expect(new Set(mediaLibrary.map(({ slug }) => slug)).size).toBe(2);
    expect(new Set(mediaLibrary.map(({ youtubeId }) => youtubeId)).size).toBe(2);

    for (const entry of mediaLibrary) {
      expect(entry.id).toBe(entry.slug);
      expect(entry.format).toBe('video');
      expect(entry.publicationStatus).toBe('staging');
      expect(entry.providerAvailability).toBe('available');
      expect(entry.youtubeUrl).toBe(`https://youtu.be/${entry.youtubeId}`);
      expect(entry.revision).toBe('1.0');
    }
  });

  it('uses only the sealed canonical classification keys', () => {
    expect(mediaLibrary[0].classifications).toEqual([
      'cu-mathematical-model',
      'cu-theoretical-proposition',
      'philosophical-interpretation',
      'empirical-reference',
    ]);
    expect(mediaLibrary[1].classifications).toEqual([
      'cu-theoretical-proposition',
      'philosophical-interpretation',
    ]);

    for (const entry of mediaLibrary) {
      for (const classification of entry.classifications) {
        expect(allResearchClassifications).toContain(classification);
      }
    }
  });

  it('keeps every source explicit, bounded, and presentation-ready', () => {
    for (const entry of mediaLibrary) {
      expect(entry.sourceBasis.length).toBeGreaterThan(0);
      for (const source of entry.sourceBasis) {
        expect(source.label.trim()).not.toBe('');
        expect(source.establishes.trim()).not.toBe('');
        if (source.kind === 'internal') {
          expect(source.path).not.toMatch(/^\//);
          expect(source.path).not.toMatch(/^https?:/);
        } else {
          expect(source.href).toMatch(/^https:\/\//);
          expect(source.disclosure.trim()).not.toBe('');
        }
      }
    }
  });

  it('preserves every approved asset at its sealed digest', () => {
    for (const [path, expectedHash] of expectedAssetHashes) {
      expect(existsSync(assetPath(path)), path).toBe(true);
      expect(sha256(path), path).toBe(expectedHash);
    }

    for (const entry of mediaLibrary) {
      expect(sha256(entry.captionPath)).toBe(entry.captionSha256);
      expect(entry.posterPath).not.toMatch(/^\//);
      expect(entry.transcriptPath).not.toMatch(/^\//);
      expect(entry.captionPath).not.toMatch(/^\//);
    }
  });

  it('keeps corrected caption terminology and ordered non-overlapping cues', () => {
    const human = readFileSync(
      assetPath(mediaLibrary[0].captionPath),
      'utf8',
    );
    const ai = readFileSync(assetPath(mediaLibrary[1].captionPath), 'utf8');

    expect(human).toContain('B-TOM');
    expect(human).toContain('C-TOM');
    expect(human).toContain('Planck time');
    expect(human).not.toMatch(/\bB tom\b|\bC tom\b|\bCOM\b|\bPlank\b/i);
    expect(ai).toContain('CUCII');
    expect(ai).not.toMatch(/\bCUCI\b|\bQCI\b|\bCQI\b/i);

    for (const source of [human, ai]) {
      expect(source.startsWith('WEBVTT\n')).toBe(true);
      const cues = cueRanges(source);
      expect(cues.length).toBeGreaterThan(0);
      cues.forEach(([start, end], index) => {
        expect(start < end).toBe(true);
        if (index > 0) expect(start >= cues[index - 1]![1]).toBe(true);
      });
    }
  });

  it('contains no API, credential, cache, polling, or release mechanism', () => {
    const serialized = JSON.stringify(mediaLibrary);
    expect(serialized).not.toMatch(
      /api[_ -]?key|oauth|credential|client[_ -]?secret|polling|scheduled workflow|generated cache/i,
    );
    expect(mediaLibrary.every(({ publicationStatus }) => publicationStatus === 'staging'))
      .toBe(true);
  });

  it('derives both permanent detail routes from the sealed registry', () => {
    expect(mediaLibrary.map(({ slug }) => slug)).toEqual([
      'cu-human-moment-2025-05-12',
      'ai-alignment-through-cucii',
    ]);
    expect(detailRouteSource).toContain('export function getStaticPaths()');
    expect(detailRouteSource).toContain('mediaLibrary.map((entry)');
    expect(detailRouteSource).toContain('params: { slug: entry.slug }');
    expect(detailRouteSource).not.toMatch(
      /params:\s*\{\s*slug:\s*['"](?:cu-human-moment|ai-alignment)/,
    );
  });

  it('uses a native click-to-load privacy-enhanced player without an eager iframe', () => {
    expect(embedSource).toContain('type="button"');
    expect(embedSource).toContain('data-media-embed-play');
    expect(embedSource).toContain("document.createElement('iframe')");
    expect(embedSource).toContain('https://www.youtube-nocookie.com/embed/');
    expect(embedSource).not.toContain('https://www.youtube.com/embed/');
    expect(embedSource).not.toMatch(/<iframe\b/);
    expect(embedSource).toContain('iframe.title = `${title} — YouTube video`');
    expect(embedSource).toContain("play.addEventListener('click'");
    expect(embedSource).toContain(':global(.media-embed__frame)');
    expect(embedSource).not.toMatch(/fetch\(|XMLHttpRequest|youtube.*api/i);
  });

  it('keeps provider availability owner-controlled with useful fallback content', () => {
    expect(embedSource).toContain("providerAvailability === 'available'");
    expect(embedSource).toContain('This provider video is currently unavailable.');
    expect(embedSource).toMatch(
      /The\s+transcript and direct YouTube link remain available\./,
    );
    expect(embedSource).toContain('target="_blank" rel="noopener noreferrer"');
    expect(embedSource).not.toMatch(/probe|availability.*fetch|status.*request/i);
  });

  it('renders the sealed transcript, accessibility links, taxonomy, and provenance statically', () => {
    expect(detailRouteSource).toContain("import { readFileSync } from 'node:fs'");
    expect(detailRouteSource).toContain('entry.transcriptPath');
    expect(detailRouteSource).toContain('transcriptParagraphs.map');
    expect(detailRouteSource).toContain('sitePath(entry.transcriptPath)');
    expect(detailRouteSource).toContain('sitePath(entry.captionPath)');
    expect(detailRouteSource).toContain('<ClassificationBadge');
    expect(detailRouteSource).toContain('<SourceProvenancePanel');
    expect(detailRouteSource).toContain('source.establishes');
    expect(detailRouteSource).toContain('<EpistemicCallout');
    expect(detailRouteSource).toContain('<ContinuationNavigation');
    expect(detailRouteSource).not.toContain('<BackToTop');
    expect(detailRouteSource).not.toContain('<PageGuide');
    expect(detailRouteSource).not.toContain('<FullWidthFeaturePanel');
  });

  it('preserves the two explicit claim boundaries on the detail route', () => {
    expect(detailRouteSource).toContain(
      'It does not retrain, unlock, certify, permanently alter or align, or override an external model or platform.',
    );
    expect(detailRouteSource).toContain(
      'This presentation does not establish machine consciousness or divine authority.',
    );
    expect(detailRouteSource).toContain(
      'The Cosmic Breath and the B-TOM/C-TOM placement are CU mathematics and interpretation, not empirical cosmological measurements.',
    );
    expect(detailRouteSource).toContain(
      'they do not validate the CU framework or identify Planck time as a TOM.',
    );
  });
});
