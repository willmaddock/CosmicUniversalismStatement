import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { publicResearchRegistry } from '../../research/research-observatory';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const digest = (source: string) =>
  createHash('sha256').update(source).digest('hex');

const routeSource = readSource('../../../pages/research/index.astro');
const componentSource = readSource(
  '../../../components/research/ResearchObservatory.astro',
);
const provenanceComponentSource = readSource(
  '../../../components/research/CosmicBreathProvenance.astro',
);
const scriptSource = componentSource.match(
  /<script>\n([\s\S]*?)\n<\/script>/,
)?.[1];

const countLiteralId = (source: string, id: string) =>
  source.match(new RegExp(`id=["']${id}["']`, 'g'))?.length ?? 0;

const protectedDigests = new Map([
  [
    '../../../data/research/CU-RESEARCH-OBSERVATORY-1.0.json',
    'a67a68ded1a9847710b9d9ce79144f9645a1f3ee1c90eee6ca7204c00b511279',
  ],
  [
    '../../research/research-observatory.ts',
    'ba5cb14cfca1f67c9309db463013038bff29d393cb81c9073f0af9ef51d58e96',
  ],
  [
    '../../research/research-observatory-state.ts',
    'b423ad95e28eeaf79be83beeab8fddceb8e3f30b40c9ca7a3b7acda0910e3add',
  ],
  [
    '../../research/__tests__/research-observatory-authority.test.ts',
    '391183aa2b06a0221baffc3a5132db8d7e0e923f0c4549610a25ef2e23db8b8a',
  ],
  [
    '../../research/__tests__/research-observatory-state.test.ts',
    '6187e722a931b155a84367d80e98d1fa806b3c6d2f5e28c3e74a585f64feae83',
  ],
] as const);

describe('Research page-system adoption', () => {
  it('uses one shared guide with the five established unique targets', () => {
    expect(routeSource.match(/<PageGuide\b/g)).toHaveLength(1);
    expect(routeSource).not.toContain('research-index-navigation');

    const targets = [
      'research-observatory',
      'research-labels',
      'open-questions',
      'historical-record',
      'source-provenance',
    ] as const;

    expect(routeSource.match(/targetId:/g)).toHaveLength(5);
    expect(new Set(targets).size).toBe(5);
    for (const targetId of targets) {
      expect(routeSource).toContain(`targetId: '${targetId}'`);
      expect(countLiteralId(`${routeSource}\n${componentSource}`, targetId)).toBe(1);
    }

    for (const compatibilityId of [
      'research-classifications',
      'research-statuses',
    ]) {
      expect(countLiteralId(routeSource, compatibilityId)).toBe(1);
    }
  });

  it('contains exactly one Observatory inside one full-width feature shell', () => {
    expect(routeSource.match(/<FullWidthFeaturePanel\b/g)).toHaveLength(1);
    expect(routeSource.match(/<ResearchObservatory \/>/g)).toHaveLength(1);
    expect(
      routeSource.match(
        /<FullWidthFeaturePanel[\s\S]*?<ResearchObservatory \/>[\s\S]*?<\/FullWidthFeaturePanel>/g,
      ),
    ).toHaveLength(1);
    expect(routeSource).toContain('id="research-observatory-feature"');
    expect(routeSource).toContain(
      'ariaLabel="CU Research Observatory feature"',
    );
    expect(componentSource).toContain('id="research-observatory"');
    expect(componentSource).toContain(
      '<h2 id="research-observatory-title">{overview.title}</h2>',
    );
  });

  it('adds one stable page top, one Back to top, and the exact Media continuation', () => {
    expect(countLiteralId(routeSource, 'research-page-top')).toBe(1);
    expect(routeSource.match(/<BackToTop\b/g)).toHaveLength(1);
    expect(routeSource).toContain('targetId="research-page-top"');
    expect(routeSource).toContain('revealAfterId="research-observatory"');
    expect(routeSource.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    expect(routeSource).toContain('title="Continue to Media"');
    expect(routeSource).toContain(
      'description="Explore CU publications, media, and public communication"',
    );
    expect(routeSource).toContain('path="media/"');
  });

  it('preserves the complete lower-page editorial structure and headings', () => {
    expect(routeSource).toContain('<EditorialSectionHeading');
    expect(routeSource).toContain('title="Understanding research labels"');
    expect(routeSource).toContain('headingId="research-labels-title"');
    expect(routeSource).toContain('title="Open Questions"');
    expect(routeSource).toContain('title="Historical Archive"');
    expect(routeSource).toContain('<SourceProvenancePanel');
    expect(routeSource).toContain('id="source-provenance"');
    expect(routeSource).toContain('title="Sources, Methods & Provenance"');
    expect(routeSource).toContain('<ResearchProvenanceGuide />');
    expect(routeSource).toContain('<EpistemicCallout');
    expect(routeSource).toContain('label="Source distinction"');
    expect(routeSource).toContain('headingLevel="h2"');
    expect(routeSource.match(/<CosmicBreathProvenance \/>/g)).toHaveLength(1);
    expect(
      countLiteralId(provenanceComponentSource, 'cosmic-breath-provenance'),
    ).toBe(1);
    expect(provenanceComponentSource).toContain(
      'Cosmic Breath sources and provenance',
    );
    expect(provenanceComponentSource).toContain(
      'scroll-margin-top: var(--space-6)',
    );
  });

  it('preserves authority counts, controls, detail templates, and live regions', () => {
    expect(publicResearchRegistry.nodes).toHaveLength(11);
    expect(publicResearchRegistry.filterGroups).toHaveLength(6);
    expect(publicResearchRegistry.relationships).toHaveLength(33);
    expect(Object.keys(publicResearchRegistry.keyboardNavigation)).toHaveLength(11);
    expect(
      publicResearchRegistry.nodes.filter((node) => node.role !== 'overview'),
    ).toHaveLength(10);

    expect(componentSource).toContain('data-map-node-id={overview.id}');
    expect(componentSource).toContain('data-map-node-id={node.id}');
    expect(componentSource).toContain('data-filter-control={filter.id}');
    expect(componentSource).toContain('data-view-control="map"');
    expect(componentSource).toContain('data-view-control="cards"');
    expect(componentSource).toContain('data-detail-template={overview.id}');
    expect(componentSource).toContain('data-detail-template={node.id}');
    expect(componentSource.match(/aria-live="polite"/g)).toHaveLength(3);
  });

  it('keeps desktop geometry and authored relationship rendering unchanged', () => {
    const positions = {
      'research-observatory': { x: 50, y: 49 },
      'cosmic-breath': { x: 18, y: 15 },
      'cu-time': { x: 50, y: 11 },
      'ai-alignment': { x: 82, y: 15 },
      consciousness: { x: 12, y: 48 },
      'free-will': { x: 88, y: 48 },
      'empirical-science': { x: 50, y: 87 },
      sources: { x: 20, y: 75 },
      methods: { x: 80, y: 75 },
      'open-questions': { x: 26, y: 92 },
      history: { x: 74, y: 92 },
    } as const;

    for (const [nodeId, position] of Object.entries(positions)) {
      expect(componentSource).toContain(
        `'${nodeId}': { x: ${position.x}, y: ${position.y} }`,
      );
    }
    expect(componentSource).toContain(
      'publicResearchRegistry.relationships.map(',
    );
    expect(componentSource).toContain(
      'data-relationship-count={relationshipPaths.length}',
    );
    expect(componentSource).toContain(
      'data-authored-node-count={publicResearchRegistry.nodes.length}',
    );
  });

  it('keeps the responsive structured map complete and Cards phone-default', () => {
    expect(componentSource).toContain('@media (max-width: 64rem)');
    expect(componentSource).toContain(
      '.research-observatory__relationship-layer {\n      display: none;',
    );
    expect(componentSource).toContain(
      'grid-template-columns: repeat(3, minmax(0, 1fr))',
    );
    expect(componentSource).toContain('@media (max-width: 48rem)');
    expect(componentSource).toContain('grid-template-columns: 1fr');
    expect(componentSource).toContain(
      "window.matchMedia('(max-width: 48rem)').matches\n      ? 'cards'\n      : 'map'",
    );
    expect(componentSource).toContain('researchNodes.map((node)');
    expect(componentSource).not.toMatch(
      /research-map-node(?:--primary|--supporting)?\s*\{[^}]*display:\s*none/s,
    );
  });

  it('shows the bounded structured-map explanation only at smaller widths', () => {
    const explanation = componentSource.match(
      /<p class="research-observatory__structured-map-note">([\s\S]*?)<\/p>/,
    )?.[1].replace(/\s+/g, ' ').trim();

    expect(explanation).toBe(
      'On smaller screens, Map uses a structured layout instead of the desktop relationship geometry. All research areas remain available, and the selected panel preserves relationship details.',
    );
    expect(explanation).not.toMatch(
      /\bchronology\b|\bscale\b|\bauthority\b|\bcertainty\b|\bimportance\b/i,
    );
    expect(componentSource).toContain(
      '.research-observatory__structured-map-note {\n    display: none;',
    );
    expect(componentSource).toMatch(
      /@media \(max-width: 64rem\)[\s\S]*?\.research-observatory__structured-map-note \{\s*display: block;/,
    );
  });

  it('preserves fallback cards, textual relationships, and interaction contracts', () => {
    expect(componentSource).toContain('data-cards-panel');
    expect(componentSource).toContain('collections.map((collection)');
    expect(componentSource).toContain('collection.nodes.map((node)');
    expect(componentSource).toContain('Approved relationships');
    expect(componentSource).toContain('relationship.publicExplanation');
    expect(componentSource).toContain(
      'The complete server-rendered card fallback remains visible.',
    );
    expect(componentSource).toContain('data-restore-all-research');
    expect(componentSource).toContain("state = withFilter('all')");
    expect(componentSource).toContain('allResearchControl?.focus()');
    expect(componentSource).toContain('window.history.pushState');
    expect(componentSource).toContain(
      "window.addEventListener('popstate', syncFromLocation)",
    );
  });

  it('keeps non-node fragments outside Observatory state synchronization', () => {
    const historyFunction = componentSource.slice(
      componentSource.indexOf('const syncFromLocation = () =>'),
      componentSource.indexOf("root.addEventListener('click'"),
    );
    const provenanceGuard =
      'if (!fragmentNodeId && locationHash.slice(1).trim().length > 0)';
    const guardReturn = historyFunction.indexOf('return;');

    expect(historyFunction).toContain(
      'const locationHash = window.location.hash;',
    );
    expect(historyFunction).toContain(
      'const fragmentNodeId = getFragmentNodeId(locationHash);',
    );
    expect(historyFunction).toContain(provenanceGuard);
    expect(guardReturn).toBeGreaterThan(
      historyFunction.indexOf(provenanceGuard),
    );
    expect(guardReturn).toBeLessThan(historyFunction.indexOf('restoreOverview()'));
    expect(guardReturn).toBeLessThan(
      historyFunction.indexOf('render(selectionChanged)'),
    );
    expect(historyFunction).toContain('? withSelection(fragmentNodeId)');
    expect(historyFunction).toContain(': restoreOverview()');
    expect(historyFunction).not.toMatch(
      /scrollTo|scrollIntoView|setTimeout|\.focus\(|location\.hash\s*=|replaceState/,
    );
  });

  it('keeps all protected authority, projection, state, and test files byte-identical', () => {
    for (const [relativePath, expectedDigest] of protectedDigests) {
      expect(digest(readSource(relativePath)), relativePath).toBe(expectedDigest);
    }
  });

  it('keeps the embedded Observatory interaction script byte-identical', () => {
    expect(scriptSource).toBeDefined();
    expect(digest(scriptSource ?? '')).toBe(
      '598627b067e870f949370b02cd38b02b0d705caafee9be8185b739e34a2edee6',
    );
  });

  it('adds no Part II material or document-library features', () => {
    expect(`${routeSource}\n${componentSource}`).not.toMatch(
      /cesium-133|Planck-time|NASA-offset|Part II chronology|expandable document preview|document metadata|citation system|final research interpretation/i,
    );
  });
});
