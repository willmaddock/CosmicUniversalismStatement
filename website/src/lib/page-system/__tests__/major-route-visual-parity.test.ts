import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const repositoryRoot = fileURLToPath(new URL('../../../../../', import.meta.url));
const read = (path: string) => readFileSync(join(repositoryRoot, path), 'utf8');
const digest = (value: string | Buffer) =>
  createHash('sha256').update(value).digest('hex');

const walk = (path: string): string[] =>
  readdirSync(join(repositoryRoot, path), { withFileTypes: true }).flatMap(
    (entry) => {
      const child = join(path, entry.name);
      return entry.isDirectory() ? walk(child) : [child];
    },
  );

const aggregateDigest = (paths: string[]) => {
  const hash = createHash('sha256');
  [...new Set(paths)].sort().forEach((path) => {
    hash.update(path);
    hash.update('\0');
    hash.update(readFileSync(join(repositoryRoot, path)));
    hash.update('\0');
  });
  return hash.digest('hex');
};

const scriptBodies = (source: string) =>
  [...source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(
    (match) => match[1] ?? '',
  );

const normalizeRouteContent = (source: string) =>
  source
    .replace(/<style>[\s\S]*?<\/style>/g, '')
    .replace(
      /\s*<span class="route-chapter-marker" aria-hidden="true">\d{2}<\/span>/g,
      '',
    )
    .replace(/<div class="research-editorial-chapter">/g, '')
    .replace(/<\/div>/g, '')
    .replace(/sectionNumber="\d{2}"/g, 'sectionNumber="NN"')
    .replace(/\s+/g, ' ')
    .trim();

const cuTimeRoutePath = 'website/src/pages/cu-time.astro';
const cuIntelligenceRoutePath = 'website/src/pages/cu-intelligence.astro';
const researchRoutePath = 'website/src/pages/research/index.astro';
const cuTimeRoute = read(cuTimeRoutePath);
const cuIntelligenceRoute = read(cuIntelligenceRoutePath);
const researchRoute = read(researchRoutePath);
const cuTimeConverter = read('website/src/components/CUTimeConverter.astro');
const researchObservatory = read(
  'website/src/components/research/ResearchObservatory.astro',
);
const researchAuthority = JSON.parse(
  read('website/src/data/research/CU-RESEARCH-OBSERVATORY-1.0.json'),
);
const mobileHeaderTest = read(
  'website/src/lib/page-system/__tests__/mobile-header.test.ts',
);
const visualParityNavigationTest = read(
  'website/src/lib/page-system/__tests__/visual-parity-navigation.test.ts',
);

const routeDigestContracts = [
  {
    entry: '../../../pages/cu-time.astro',
    source: cuTimeRoute,
    baseline:
      '72180f351cf051066b96a771e7a542364a2858cf64c76afd629e65086cc800ee',
  },
  {
    entry: '../../../pages/cu-intelligence.astro',
    source: cuIntelligenceRoute,
    baseline:
      '26acbe37fb1a64d0bb44ce3c67b9f108d814ee66e50f9967b770bb972b6e49df',
  },
  {
    entry: '../../../pages/research/index.astro',
    source: researchRoute,
    baseline:
      'ecff1649b871789bf98ae7093482db989051a7303d1eec5abebb76d8ca53a214',
  },
] as const;

const normalizeAuthorizedRouteDigests = (source: string) => {
  let normalized = source;
  for (const contract of routeDigestContracts) {
    const escapedEntry = contract.entry.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    normalized = normalized.replace(
      new RegExp(`(\\['${escapedEntry}', ')[a-f0-9]{64}('])`),
      `$1${contract.baseline}$2`,
    );
  }
  return normalized;
};

const componentFiles = walk('website/src/components');
const dataFiles = walk('website/src/data');
const existingTestFiles = walk('website/src/lib').filter((path) =>
  path.includes('/__tests__/'),
);

const protectedGroups = {
  shared: [
    'website/src/components/page-system/PageGuide.astro',
    'website/src/components/page-system/ContinuationNavigation.astro',
    'website/src/components/page-system/FullWidthFeaturePanel.astro',
    'website/src/components/page-system/EditorialSectionHeading.astro',
    'website/src/components/page-system/EpistemicCallout.astro',
    'website/src/components/page-system/SourceProvenancePanel.astro',
    'website/src/components/page-system/BackToTop.astro',
    'website/src/components/Header.astro',
    'website/src/components/Footer.astro',
    'website/src/layouts/BaseLayout.astro',
    'website/src/layouts/ContentPageLayout.astro',
    'website/src/styles/global.css',
    'website/src/styles/tokens.css',
    'website/src/config/site.ts',
  ],
  cosmicBreath: [
    'website/src/pages/cosmic-breath.astro',
    ...componentFiles.filter((path) =>
      dirname(path) === 'website/src/components' &&
      relative(dirname(path), path).startsWith('CosmicBreath'),
    ),
    ...walk('website/src/data/cosmic-breath'),
    ...walk('website/src/lib/cosmicBreath'),
  ],
  cuTime: [
    'website/src/components/CUTimeConverter.astro',
    ...walk('website/src/lib/cuTime'),
    'website/src/lib/page-system/__tests__/cu-time-page-system.test.ts',
  ],
  cucii: [
    ...walk('website/src/lib/cucii'),
    ...dataFiles.filter((path) => path.split('/').at(-1)?.startsWith('cucii-')),
    ...walk('website/src/components/cucii'),
    'website/src/lib/page-system/__tests__/cu-intelligence-page-system.test.ts',
  ],
  research: [
    ...walk('website/src/lib/research'),
    ...dataFiles.filter(
      (path) =>
        path.includes('/research/') ||
        ['research-content.ts', 'research-taxonomy.ts'].includes(
          path.split('/').at(-1) ?? '',
        ),
    ),
    'website/src/components/research/ResearchObservatory.astro',
    'website/src/components/ResearchIndexSection.astro',
    'website/src/components/ResearchProvenanceGuide.astro',
    'website/src/components/ResearchStatusBadge.astro',
    'website/src/components/ClassificationBadge.astro',
    'website/src/lib/page-system/__tests__/research-page-system.test.ts',
  ],
  existingTests: existingTestFiles.filter(
    (path) =>
      !path.endsWith('mobile-header.test.ts') &&
      !path.endsWith('visual-parity-navigation.test.ts') &&
      !path.endsWith('major-route-visual-parity.test.ts'),
  ),
} as const;

const baselineGroupDigests = {
  shared: '5d699f46c7409fad040c1e6dc654bfe560b58f9c51dcc28aee7f7f2df4a5b5ac',
  cosmicBreath:
    '9d34a804e22cea2e0a66f01629eefafc41941a8b944f63fd7d646bde3c13954b',
  cuTime: '4be55635476f513a3fa1b93c2111c41cfd53cd2cf3a1377fc00eacde43abc5f9',
  cucii: '4fccddc9566ec4e753060e7179472318dbfc906dd3c632cbf3f5256810f7ac86',
  research:
    '1dbfccabf6e1bda9513151fab4596ca736d017105b344e9da55416b7cd0f0ec6',
  existingTests:
    '2d592f508a8aea1b1454fe16e78a86737e1bbaf1437d7f76a785b3e9469c7a8b',
} as const;

describe('major-route graphical parity', () => {
  it('keeps shared systems, Cosmic Breath, and independent logic byte-identical', () => {
    for (const [name, paths] of Object.entries(protectedGroups)) {
      expect(aggregateDigest([...paths]), name).toBe(
        baselineGroupDigests[name as keyof typeof baselineGroupDigests],
      );
    }
  });

  it('preserves substantive route content while adding presentation-only chapters', () => {
    expect(digest(normalizeRouteContent(cuTimeRoute))).toBe(
      '910c3a567f905c66768e0151b208f41b13dfad395c0c3ca64dc37474e96f3a5b',
    );
    expect(digest(normalizeRouteContent(cuIntelligenceRoute))).toBe(
      'df41cd60fa5d27a772ca8669342563f1873b592881dd0a5ec9233c6610764fe2',
    );
    expect(digest(normalizeRouteContent(researchRoute))).toBe(
      '31812a96ae67de2df79d3e51249a85b7cd7dca93b6ea0175ef55ee58c3c9dd35',
    );
  });

  it('preserves every protected browser script body by exact extraction', () => {
    expect(scriptBodies(cuTimeConverter)).toHaveLength(1);
    expect(digest(scriptBodies(cuTimeConverter)[0])).toBe(
      '2e892cdcbdefd969c0e99462c04f0244a806b301cccad7da8226bbc842f97361',
    );

    expect(scriptBodies(cuIntelligenceRoute)).toHaveLength(1);
    expect(digest(scriptBodies(cuIntelligenceRoute)[0])).toBe(
      '1abe5e35a4388fef0616d6c5031f1f897346fb84a7b38882d0fb62a22e0103e0',
    );

    expect(scriptBodies(researchObservatory)).toHaveLength(3);
    expect(scriptBodies(researchObservatory).map(digest)).toEqual([
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
      'e481cd8bfc6d458865cf8aa2df819f1ab5f0d785428bb1fa8b61875fe2c94225',
    ]);
  });

  it('retains CU-Time structure and separates conversion, result, and method surfaces', () => {
    expect(cuTimeRoute.match(/<CUTimeConverter\b/g)).toHaveLength(1);
    for (const id of [
      'cu-time-converter-feature',
      'cu-time-understand-result',
      'cu-time-method-limits',
      'cu-time-precision',
    ]) {
      expect(cuTimeRoute).toContain(`id="${id}"`);
    }
    for (const id of [
      'gregorian-month',
      'gregorian-day',
      'gregorian-year',
      'cu-time-input',
      'gregorian-error',
      'cu-time-error',
    ]) {
      expect(cuTimeConverter).toContain(`id="${id}"`);
    }
    expect(cuTimeRoute.match(/class="route-chapter-marker"/g)).toHaveLength(3);
    expect(cuTimeRoute).toContain('#cu-time-converter-feature .cu-time-results');
    expect(cuTimeRoute).toContain('title="Continue to CU Intelligence"');
    expect(cuTimeRoute).toContain('revealAfterId="cu-time-converter-feature"');
  });

  it('retains the CUCII studio contract and visually distinguishes its surfaces', () => {
    expect(cuIntelligenceRoute.match(/id="prompt-studio"/g)).toHaveLength(1);
    for (const contract of [
      'id="cucii-studio-form"',
      'id="studio-errors"',
      'id="studio-status"',
      'id="prompt-preview"',
      'id="prompt-text"',
      'data-proven-preset',
      'data-preset-state',
      'aria-live="assertive"',
      'aria-live="polite"',
    ]) {
      expect(cuIntelligenceRoute).toContain(contract);
    }
    for (const visualSurface of [
      '.conversation-grid :global(.conversation-card)',
      '.proven-configuration-card',
      '.studio-form',
      '.prompt-preview',
      '.studio-usage-guidance',
    ]) {
      expect(cuIntelligenceRoute).toContain(visualSurface);
    }
    expect(cuIntelligenceRoute).not.toMatch(
      /guaranteed alignment|permanent alignment|AI consciousness|divine access|supernatural memory/i,
    );
    expect(cuIntelligenceRoute).toContain('title="Continue to Research"');
  });

  it('retains the complete Observatory contract and adds lower-page hierarchy', () => {
    expect(researchAuthority.nodes).toHaveLength(11);
    expect(researchAuthority.filterGroups).toHaveLength(6);
    expect(researchAuthority.relationships).toHaveLength(33);
    expect(Object.keys(researchAuthority.keyboardNavigation)).toHaveLength(11);
    expect(
      researchAuthority.nodes.filter(
        (node: { id: string }) => node.id !== researchAuthority.overviewNodeId,
      ),
    ).toHaveLength(10);
    expect(researchObservatory.match(/<template data-detail-template=/g)).toHaveLength(2);
    expect(researchObservatory.match(/aria-live="polite"/g)).toHaveLength(3);
    expect(researchObservatory).toContain('data-view-control="cards" aria-pressed="true"');
    expect(researchObservatory).toContain('research-observatory__structured-map-note');
    expect(researchRoute.match(/class="route-chapter-marker"/g)).toHaveLength(5);
    expect(researchRoute.match(/class="research-editorial-chapter"/g)).toHaveLength(3);
    expect(researchRoute).toContain('title="Continue to Media"');
  });

  it('keeps decorative markers neutral, responsive, and outside interactive content', () => {
    for (const [route, count] of [
      [cuTimeRoute, 3],
      [cuIntelligenceRoute, 1],
      [researchRoute, 5],
    ] as const) {
      expect(route.match(/class="route-chapter-marker" aria-hidden="true"/g)).toHaveLength(
        count,
      );
      expect(route).toContain('@media (max-width: 36rem)');
      expect(route).not.toMatch(/overflow-x:\s*(?:auto|scroll)/);
      expect(route).not.toMatch(/position:\s*(?:fixed|sticky)/);
      expect(route).not.toMatch(/min-width:\s*(?:[1-9]\d*rem|[1-9]\d*px|100vw)/);
      expect(route).not.toMatch(/<(?:canvas|svg)\b/);
    }
  });

  it('advances the authorized route digests to the defect-corrected bytes', () => {
    for (const testSource of [mobileHeaderTest, visualParityNavigationTest]) {
      for (const contract of routeDigestContracts) {
        expect(testSource).toContain(
          `['${contract.entry}', '${digest(contract.source)}']`,
        );
      }
    }

    expect(digest(normalizeAuthorizedRouteDigests(mobileHeaderTest))).toBe(
      '946cd38930533a6b56639fbcdda69adda0fb09277086eeef4957763525554aa9',
    );
    expect(
      digest(normalizeAuthorizedRouteDigests(visualParityNavigationTest)),
    ).toBe(
      '858773deeafbae53f239d9d0a676a7be7da783399e3cb46a2c7c921ab36d7714',
    );
  });
});
