import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const digest = (source: string) =>
  createHash('sha256').update(source).digest('hex');

const guideSource = readSource(
  '../../../components/page-system/PageGuide.astro',
);
const continuationSource = readSource(
  '../../../components/page-system/ContinuationNavigation.astro',
);

const routeSources = {
  home: readSource('../../../pages/index.astro'),
  framework: readSource('../../../pages/framework.astro'),
  cosmicBreath: readSource('../../../pages/cosmic-breath.astro'),
  cuTime: readSource('../../../pages/cu-time.astro'),
  cuIntelligence: readSource('../../../pages/cu-intelligence.astro'),
  research: readSource('../../../pages/research/index.astro'),
  media: readSource('../../../pages/media.astro'),
  about: readSource('../../../pages/about.astro'),
} as const;

const protectedDigests = new Map([
  ['../../../components/page-system/BackToTop.astro', '96b25b7b8d604658af0fd7b48c6eebf6578487ec0ea776e536f697823ac3aa69'],
  ['../../../components/page-system/FullWidthFeaturePanel.astro', '278ec5d072f44f0aeafdd86c93f9204871f6bb97d6b1e21120e7150c51f56a7c'],
  ['../../../components/page-system/EditorialSectionHeading.astro', 'd411f8bf19241e7c8e7895cd1f66b2bb5792e3e7e091798dab7baa42790f0ab4'],
  ['../../../components/page-system/EpistemicCallout.astro', 'a2b4221d8b813d3b6247b8003b6e73e83017a0f439ca4cf0f0abcfc6a2fdc7c2'],
  ['../../../components/page-system/SourceProvenancePanel.astro', 'a6be773a5a77336a42e4366a5198dd04836682eadeffbe997c6c2a412f0fed43'],
  ['../../../components/Header.astro', '2b1e7cf1dfebfc1e98ba43c6539c5006fff5b96704a14856996994fa6909956f'],
  ['../../../components/Footer.astro', '685a88746cc80f6bfc97a405886be084ac8dae5b9c91f3637b8f0bfdf4244e63'],
  ['../../../layouts/BaseLayout.astro', 'acd647a19004cd5843047e11970a6d364f8a2e935c0d9302158c173820f25582'],
  ['../../../layouts/ContentPageLayout.astro', 'a76222c78baa158d9c1022142f24ad47207bcbca840efdd5a93ea947194ab149'],
  ['../../../styles/global.css', 'f5bc581e43533441fe2e35ced4420e80d1f59b8ae746a02f27d6d181f94e8e4a'],
  ['../../../styles/tokens.css', 'f8c2260dc066db1fdd05d05777494c180c74c54e9342879d3d8b5d37939ddc54'],
  ['../../../config/site.ts', 'b6aac8c9040b40ef8aa5e10ad9683eb4e37831e5ebb31d939bcf47bf6ac7e470'],
  ['../../../pages/index.astro', 'f78dc57be535dd5bd83eac1aaa98d8882d7b8d04f5df5a304435ce6e6b60fcff'],
  ['../../../pages/framework.astro', '16fcffe3ce0fb7632756626221d3a8a984f85e28c61df717aec3f850dbf3e8ac'],
  ['../../../pages/cosmic-breath.astro', 'a0ed45ef3aff558f297bff50893dc1b3ca82e75cc7756b16be050bf7a2a511bc'],
  ['../../../pages/cu-time.astro', 'dc557e235883cb18b33766d2d818139b58eb202fcd5d0be3daa1909bbae4c0d3'],
  ['../../../pages/cu-intelligence.astro', '6f371ba4beffa261e5183db7795b649354760f9939da323df2f29a280fdc90c7'],
  ['../../../pages/research/index.astro', '5c778a30758adb2114e020fc9db8d7dd2c2f55b98503a1ec52c86b21c87d5273'],
  ['../../../pages/media.astro', '00d48d13e4f36382d57620d169b3d265fe06dc76c75d52540149e409aecf28ff'],
  ['../../../pages/about.astro', '1de1e76e3e27c940af0a09ad0923771b9b1b7b5891060ea6c02022812c27b4d5'],
] as const);

describe('shared in-page navigation and closing parity', () => {
  it('keeps the PageGuide public contract and consumer-owned navigation intact', () => {
    for (const prop of ['ariaLabel', 'title', 'groups', 'eyebrow', 'id', 'headingId']) {
      expect(guideSource).toContain(prop);
    }

    expect(guideSource.match(/<nav\b/g)).toHaveLength(1);
    expect(guideSource).toContain('<nav aria-label={ariaLabel}>');
    expect(guideSource).toContain('<h2 id={headingId}>{title}</h2>');
    expect(guideSource).toContain('groups.map((group)');
    expect(guideSource).toContain('group.links.map((link)');
    expect(guideSource).toContain('href={`#${link.targetId}`}');
    expect(guideSource).not.toContain('<script');
  });

  it('adds structural non-color numbering without changing labels or fragments', () => {
    expect(guideSource).toContain('counter-reset: page-guide-group');
    expect(guideSource).toContain('counter-increment: page-guide-group');
    expect(guideSource).toContain(
      'content: counter(page-guide-group, decimal-leading-zero)',
    );
    expect(guideSource).toContain(
      '<span class="page-guide__group-number" aria-hidden="true"></span>',
    );
    expect(guideSource).not.toMatch(/targetId.*(?:index|number)|href=.*group-number/);
  });

  it('provides the stronger layered PageGuide frame without interaction debt', () => {
    expect(guideSource).toContain('width: 100%');
    expect(guideSource).toContain('var(--color-border-feature)');
    expect(guideSource).toContain('var(--color-border-section)');
    expect(guideSource).toContain('var(--color-panel-feature)');
    expect(guideSource).toContain('var(--color-panel-quiet)');
    expect(guideSource).toContain('color: var(--color-amber)');
    expect(guideSource).toContain('font-family: var(--font-family-serif)');
    expect(guideSource).toContain('repeat(auto-fit');
    expect(guideSource).toContain('max-width: none');
    expect(guideSource).toContain('@media (max-width: 36rem)');
    expect(guideSource).not.toContain('var(--space-5)');
    expect(guideSource).not.toMatch(/position:\s*sticky|overflow-x:\s*(?:auto|scroll)/);
    const minimumWidths = [...guideSource.matchAll(/min-width:\s*([^;]+)/g)]
      .map((match) => match[1]?.trim());
    expect(minimumWidths).toEqual(['0']);
    expect(guideSource).not.toMatch(/<details|<button|history\.|IntersectionObserver/);
  });

  it('keeps one semantic, base-path-safe continuation action', () => {
    expect(continuationSource.match(/<nav\b/g)).toHaveLength(1);
    expect(continuationSource.match(/<a\b/g)).toHaveLength(1);
    expect(continuationSource).toContain('aria-label={ariaLabel}');
    expect(continuationSource).toContain('<a href={sitePath(path)}>');
    expect(continuationSource).toContain('<strong>{title}</strong>');
    expect(continuationSource).toContain('<span>{description}</span>');
    expect(continuationSource).toContain('Forward →');
    expect(continuationSource).not.toContain('target=');
    expect(continuationSource).not.toMatch(/<a[\s\S]*<a|<button|<script/);
  });

  it('provides a broad layered continuation with clear responsive direction', () => {
    expect(continuationSource).toContain('width: 100%');
    expect(continuationSource).toContain('grid-template-columns: minmax(7rem');
    expect(continuationSource).toContain('var(--color-panel-feature)');
    expect(continuationSource).toContain('color: var(--color-amber)');
    expect(continuationSource).toContain('color: var(--color-cyan)');
    expect(continuationSource).toContain('font-family: var(--font-family-serif)');
    expect(continuationSource).toContain('a:focus-visible');
    expect(continuationSource).toContain('@media (max-width: 48rem)');
    expect(continuationSource).toContain('grid-template-columns: 1fr');
    expect(continuationSource).not.toContain('var(--space-5)');
    expect(continuationSource).not.toMatch(/position:\s*(?:fixed|sticky)|100vw|<script/);
  });

  it('preserves PageGuide and continuation consumer counts', () => {
    expect(routeSources.cuTime.match(/<PageGuide\b/g)).toHaveLength(1);
    expect(routeSources.cuIntelligence.match(/<PageGuide\b/g)).toHaveLength(1);
    expect(routeSources.research.match(/<PageGuide\b/g)).toHaveLength(1);

    for (const route of [
      routeSources.home,
      routeSources.framework,
      routeSources.cosmicBreath,
      routeSources.cuTime,
      routeSources.cuIntelligence,
      routeSources.research,
      routeSources.media,
      routeSources.about,
    ]) {
      expect(route.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    }

    for (const route of [
      routeSources.home,
      routeSources.media,
      routeSources.about,
      routeSources.cosmicBreath,
    ]) {
      expect(route).not.toContain('<PageGuide');
    }
  });

  it('keeps route-owned PageGuide destinations and continuation copy unchanged', () => {
    expect(routeSources.cuTime.match(/targetId:/g)).toHaveLength(3);
    expect(routeSources.cuIntelligence.match(/targetId:/g)).toHaveLength(4);
    expect(routeSources.research.match(/targetId:/g)).toHaveLength(5);

    const continuationContracts = [
      [routeSources.home, 'Continue to Framework', 'framework/'],
      [routeSources.framework, 'Continue to Cosmic Breath', 'cosmic-breath/'],
      [routeSources.cosmicBreath, 'Continue to CU-Time', 'cu-time/'],
      [routeSources.cuTime, 'Continue to CU Intelligence', 'cu-intelligence/'],
      [routeSources.cuIntelligence, 'Continue to Research', 'research/'],
      [routeSources.research, 'Continue to Media', 'media/'],
      [routeSources.media, 'Continue to About', 'about/'],
      [routeSources.about, 'Return to Home', '/'],
    ] as const;

    for (const [source, title, path] of continuationContracts) {
      expect(source).toContain(`title="${title}"`);
      expect(source).toContain(`path="${path}"`);
    }
  });

  it('keeps every protected route and dependency byte-identical', () => {
    for (const [relativePath, expectedDigest] of protectedDigests) {
      expect(digest(readSource(relativePath))).toBe(expectedDigest);
    }
  });
});
