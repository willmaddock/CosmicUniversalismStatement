import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  createPublicResearchRegistry,
  publicResearchRegistry,
  serializePublicResearchRegistry,
} from '../research-observatory';
import {
  applyResearchFilter,
  createNeutralResearchState,
  getResearchKeyboardNeighbor,
  selectResearchNode,
} from '../research-observatory-state';
import {
  allResearchClassifications,
  allResearchStatuses,
  researchClassifications,
  researchStatuses,
} from '../../../data/research-taxonomy';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const routeSource = readSource('../../../pages/research/index.astro');
const componentSource = readSource(
  '../../../components/research/ResearchObservatory.astro',
);
const astroConfigSource = readSource('../../../../astro.config.mjs');

const approvedCardTitles = [
  'Cosmic Breath & TOM Structure',
  'CU-Time & Cosmic Chronology',
  'AI Alignment & Recursive Intelligence',
  'Consciousness & Foundational Existence',
  'Free Will, Ethics & Memory',
  'Empirical Science Lens',
  'Sources & Manuals',
  'Methods & Governance',
  'Open Questions',
  'Historical Archive',
] as const;

describe('Research Observatory route foundation', () => {
  it('uses the exact approved eyebrow and H1', () => {
    expect(routeSource).toContain('eyebrow="COSMIC UNIVERSALISM RESEARCH"');
    expect(routeSource).toContain(
      'title="Research, Philosophy & Open Questions"',
    );
  });

  it('places one shared page guide before the full-width Observatory feature', () => {
    expect(routeSource).toContain(
      "import ResearchObservatory from '../../components/research/ResearchObservatory.astro';",
    );
    expect(routeSource.match(/<PageGuide\b/g)).toHaveLength(1);
    expect(routeSource.match(/<FullWidthFeaturePanel\b/g)).toHaveLength(1);
    expect(routeSource.indexOf('<ResearchObservatory />')).toBeGreaterThan(-1);
    expect(routeSource.indexOf('<PageGuide')).toBeLessThan(
      routeSource.indexOf('<FullWidthFeaturePanel'),
    );
    expect(routeSource.indexOf('<ResearchObservatory />')).toBeGreaterThan(
      routeSource.indexOf('<FullWidthFeaturePanel'),
    );
  });

  it('renders a separate neutral CU Research Observatory overview', () => {
    expect(publicResearchRegistry.overviewNodeId).toBe('research-observatory');
    expect(componentSource).toContain(
      'id="research-observatory"',
    );
    expect(componentSource).toContain('{overview.title}');
    expect(
      publicResearchRegistry.nodes.find(
        (node) => node.id === 'research-observatory',
      )?.title,
    ).toBe('CU Research Observatory');
  });

  it('provides exactly the ten approved research cards', () => {
    const cardNodes = publicResearchRegistry.nodes.filter(
      (node) => node.role !== 'overview',
    );
    expect(cardNodes).toHaveLength(10);
    expect(cardNodes.map((node) => node.title)).toEqual(approvedCardTitles);
    expect(componentSource).toContain(
      "node.role === 'primary'",
    );
    expect(componentSource).toContain(
      "node.role === 'supporting'",
    );
  });

  it('does not render the central overview as an ordinary card', () => {
    const primaryNodes = publicResearchRegistry.nodes.filter(
      (node) => node.role === 'primary',
    );
    const supportingNodes = publicResearchRegistry.nodes.filter(
      (node) => node.role === 'supporting',
    );
    expect([...primaryNodes, ...supportingNodes]).not.toContainEqual(
      expect.objectContaining({ id: 'research-observatory' }),
    );
    expect(componentSource).toContain(
      "const researchNodes = publicResearchRegistry.nodes.filter",
    );
    expect(componentSource).toContain(
      "(node) => node.role !== 'overview'",
    );
  });

  it('renders classifications and lifecycle statuses from the projection', () => {
    expect(componentSource).toContain(
      '<ResearchStatusBadge status={node.status} />',
    );
    expect(componentSource).toContain(
      'node.classifications.map((classification)',
    );
    expect(componentSource).toContain(
      '<ClassificationBadge classification={classification} />',
    );
    expect(
      publicResearchRegistry.nodes
        .filter((node) => node.role !== 'overview')
        .every((node) => node.status.length > 0),
    ).toBe(true);
  });

  it('resolves internal links through the GitHub Pages base path', () => {
    const projection = createPublicResearchRegistry(
      (path) => `/CosmicUniversalismStatement/${path}`,
    );
    const internalDestinations = projection.nodes.flatMap((node) => [
      node.primaryDestination,
      node.governingSourceDestination,
    ]).filter(
      (destination) =>
        destination !== undefined && !destination.external,
    );
    expect(internalDestinations.length).toBeGreaterThan(0);
    expect(
      internalDestinations.every((destination) =>
        destination.href.startsWith('/CosmicUniversalismStatement/')),
    ).toBe(true);
    expect(astroConfigSource).toContain(
      "base: '/CosmicUniversalismStatement'",
    );
  });

  it('delivers the migrated Cosmic Breath action through every rendered path', () => {
    const projection = createPublicResearchRegistry(
      (path) => `/CosmicUniversalismStatement/${path}`,
    );
    const breath = projection.nodes.find((node) => node.id === 'cosmic-breath');
    const action = breath?.governingSourceDestination;

    expect(action).toEqual({
      kind: 'internal',
      href:
        '/CosmicUniversalismStatement/research#cosmic-breath-provenance',
      label: 'Review Cosmic Breath sources and provenance',
      external: false,
    });
    expect(action).not.toHaveProperty('externalLabel');
    expect(action).not.toHaveProperty('opensInNewTab');
    expect(action).not.toHaveProperty('rel');
    expect(componentSource.match(/node\.governingSourceDestination &&/g))
      .toHaveLength(2);
    expect(componentSource).toContain('data-research-public-registry');
    expect(componentSource).toContain(
      "target: destination.external ? '_blank' : undefined",
    );
    expect(componentSource).toContain(
      'rel: destination.external ? destination.rel : undefined',
    );
    expect(componentSource).not.toMatch(
      /analytics|telemetry|tracking(?:Id|Parameter)/i,
    );
    expect(JSON.stringify(projection)).not.toContain(
      'View the Cosmic Breath source',
    );
    expect(JSON.stringify(projection)).not.toContain(
      'Cosmic_Breath_Calculation.md',
    );
  });

  it('renders safe and visibly disclosed external links', () => {
    const externalDestinations = publicResearchRegistry.nodes.flatMap(
      (node) => [
        node.primaryDestination,
        node.governingSourceDestination,
      ],
    ).filter(
      (destination) =>
        destination !== undefined && destination.external,
    );
    expect(externalDestinations.length).toBeGreaterThan(0);
    expect(
      externalDestinations.every(
        (destination) =>
          destination.externalLabel?.includes('(external)')
          && destination.opensInNewTab === true
          && destination.rel === 'noopener noreferrer',
      ),
    ).toBe(true);
    expect(componentSource).toContain(
      "target: destination.external ? '_blank' : undefined",
    );
    expect(componentSource).toContain(
      'rel: destination.external ? destination.rel : undefined',
    );
  });

  it('projects the corrected visitor-facing and governing destinations', () => {
    const nodes = new Map(
      publicResearchRegistry.nodes.map((node) => [node.id, node]),
    );
    expect(nodes.get('cosmic-breath')?.governingSourceDestination)
      .toEqual({
        kind: 'internal',
        href: '/research#cosmic-breath-provenance',
        label: 'Review Cosmic Breath sources and provenance',
        external: false,
      });
    expect(nodes.get('cosmic-breath')?.governingSourceDestination)
      .not.toHaveProperty('externalLabel');
    expect(nodes.get('cosmic-breath')?.governingSourceDestination)
      .not.toHaveProperty('opensInNewTab');
    expect(nodes.get('cosmic-breath')?.governingSourceDestination)
      .not.toHaveProperty('rel');
    expect(nodes.get('methods')?.primaryDestination).toMatchObject({
      kind: 'internal',
      href: '/research#source-provenance',
      external: false,
    });
    expect(nodes.get('methods')?.governingSourceDestination).toMatchObject({
      kind: 'external',
      href: expect.stringMatching(/Website_Master_Implementation_Manual_v1\.1\.pdf$/),
      opensInNewTab: true,
      rel: 'noopener noreferrer',
    });
    expect(nodes.get('history')?.primaryDestination).toMatchObject({
      kind: 'internal',
      href: '/research#historical-record',
      external: false,
    });
    expect(nodes.get('history')?.governingSourceDestination).toMatchObject({
      href: 'https://github.com/willmaddock/CosmicUniversalismStatement/tags',
      externalLabel: expect.stringContaining('repository tag history'),
    });
    expect(nodes.get('empirical-science')).not.toHaveProperty(
      'primaryDestination',
    );
    expect(nodes.get('empirical-science')?.governingSourceDestination)
      .toMatchObject({
        href: 'https://github.com/willmaddock/CosmicUniversalismStatement/blob/website/Docs/CU_Framework.md',
        opensInNewTab: true,
        rel: 'noopener noreferrer',
      });
    expect(nodes.get('consciousness')).not.toHaveProperty(
      'primaryDestination',
    );
    expect(nodes.get('consciousness')?.governingSourceDestination?.href)
      .toMatch(/ResearchFiles\/CU_Consciousness\.md$/);
  });

  it('supports clean one-action and two-action records without filler links', () => {
    const actionCount = (nodeId: string) => {
      const node = publicResearchRegistry.nodes.find(
        (candidate) => candidate.id === nodeId,
      );
      return [
        node?.primaryDestination,
        node?.governingSourceDestination,
      ].filter((destination) => destination !== undefined).length;
    };

    for (const nodeId of [
      'ai-alignment',
      'consciousness',
      'empirical-science',
      'open-questions',
    ]) {
      expect(actionCount(nodeId)).toBe(1);
    }
    for (const nodeId of [
      'cosmic-breath',
      'cu-time',
      'free-will',
      'sources',
      'methods',
      'history',
    ]) {
      expect(actionCount(nodeId)).toBe(2);
    }

    expect(componentSource.match(/node\.primaryDestination &&/g))
      .toHaveLength(2);
    expect(componentSource.match(/node\.governingSourceDestination &&/g))
      .toHaveLength(2);
    expect(componentSource).not.toContain('placeholder action');
    expect(componentSource).not.toContain('disabled action');
  });

  it('renders every approved epistemic boundary', () => {
    const boundedNodes = publicResearchRegistry.nodes.filter(
      (node) => node.epistemicBoundary,
    );
    expect(boundedNodes.length).toBeGreaterThan(0);
    expect(componentSource).toContain('node.epistemicBoundary');
    expect(componentSource).toContain('epistemic boundary');
    expect(componentSource).toContain('<strong>Boundary:</strong>');
  });

  it('states the complete geometry and non-ranking boundary', () => {
    for (const phrase of [
      'chronology',
      'physical',
      'TOM duration',
      'scientific certainty',
      'authority rank',
      'moral value',
      'research importance',
      'completion percentage',
    ]) {
      expect(componentSource).toContain(phrase);
    }
  });

  it('provides exactly the five approved major on-this-page destinations', () => {
    const destinations = [
      ['research-observatory', 'CU Research Observatory'],
      ['research-labels', 'Research labels'],
      ['open-questions', 'Open Questions'],
      ['historical-record', 'Historical Archive'],
      ['source-provenance', 'Sources & Methods'],
    ] as const;

    expect(routeSource.match(/targetId:/g)).toHaveLength(5);
    destinations.forEach(([targetId, label]) => {
      expect(routeSource).toContain(`targetId: '${targetId}'`);
      expect(routeSource).toContain(`label: '${label}'`);
    });
    expect(new Set(destinations.map(([targetId]) => targetId)).size).toBe(5);
    expect(routeSource).not.toContain(
      '<nav class="research-index-navigation"',
    );
    expect(routeSource).toContain(
      'ariaLabel="Research page guide"',
    );
    expect(routeSource).toContain(
      'title="Explore Research"',
    );
    expect(destinations).toEqual([
      ['research-observatory', 'CU Research Observatory'],
      ['research-labels', 'Research labels'],
      ['open-questions', 'Open Questions'],
      ['historical-record', 'Historical Archive'],
      ['source-provenance', 'Sources & Methods'],
    ]);
  });

  it('combines complete classifications and statuses as distinct label concepts', () => {
    expect(routeSource).toContain('id="research-labels"');
    expect(routeSource).toContain('Understanding research labels');
    expect(routeSource).toContain(
      'Classifications describe what kind of material an entry contains.',
    );
    expect(routeSource).toContain(
      'Lifecycle statuses describe an entry’s current research or editorial state.',
    );
    expect(routeSource).toContain('id="research-classifications"');
    expect(routeSource).toContain('id="research-statuses"');
    expect(routeSource).toContain(
      'allResearchClassifications.map((classification)',
    );
    expect(routeSource).toContain('allResearchStatuses.map((status)');
    expect(allResearchClassifications).toHaveLength(7);
    expect(allResearchStatuses).toHaveLength(7);
    expect(allResearchClassifications.map(
      (classification) => researchClassifications[classification].label,
    )).toEqual([
      'Empirical Reference',
      'CU Mathematical Model',
      'CU Theoretical Proposition',
      'Philosophical Interpretation',
      'In-World Narrative',
      'Open Question',
      'Historical / Superseded',
    ]);
    expect(allResearchStatuses.map(
      (status) => researchStatuses[status].label,
    )).toEqual([
      'Foundational',
      'Active Research',
      'Provisional',
      'Under Review',
      'Open Problem',
      'Superseded',
      'Archived',
    ]);
  });

  it('removes the redundant pathways while retaining substantive lower sections', () => {
    expect(routeSource).not.toContain('id="research-pathways"');
    expect(routeSource).not.toContain('title="Research pathways"');
    expect(routeSource).not.toContain('<h3>Scientific context</h3>');
    expect(routeSource).not.toContain('<h3>Framework work</h3>');
    expect(routeSource).not.toContain(
      '<h3>Interpretive and unresolved work</h3>',
    );
    expect(routeSource).toContain('id="open-questions"');
    expect(routeSource).toContain('title="Open Questions"');
    expect(routeSource).toContain('id="historical-record"');
    expect(routeSource).toContain('title="Historical Archive"');
    expect(routeSource).toMatch(
      /archive preserves historical, superseded, and archived material/,
    );
    expect(routeSource).toContain('id="source-provenance"');
    expect(routeSource).toContain('title="Sources, Methods & Provenance"');
    expect(routeSource).toContain('<ResearchProvenanceGuide />');
    for (const concept of [
      'original CU repository source',
      'external empirical references',
      'version',
      'last-reviewed date',
      'revision history',
      'CU claim',
      'empirical context',
      'CU framework',
    ]) {
      expect(routeSource).toContain(concept);
    }
  });

  it('preserves approved Observatory destinations and compatibility fragments', () => {
    const internalDestinations = publicResearchRegistry.nodes.flatMap(
      (node) => [node.primaryDestination, node.governingSourceDestination],
    ).filter(
      (destination) =>
        destination !== undefined && !destination.external,
    );

    expect(internalDestinations.map((destination) => destination.href))
      .toEqual(expect.arrayContaining([
        '/research#open-questions',
        '/research#source-provenance',
      ]));
    for (const id of [
      'research-observatory',
      'research-labels',
      'research-classifications',
      'research-statuses',
      'open-questions',
      'historical-record',
      'source-provenance',
    ]) {
      expect(`${componentSource}\n${routeSource}`).toContain(`id="${id}"`);
    }
  });

  it('keeps complete cards server-rendered while enhancement controls start hidden', () => {
    expect(componentSource).toContain('collections.map((collection)');
    expect(componentSource).toContain('collection.nodes.map((node)');
    expect(componentSource).toContain('data-enhancement-controls');
    expect(componentSource).toMatch(/data-enhancement-controls\s+hidden/);
    expect(componentSource).toMatch(/data-select-node=\{node\.id\}[\s\S]*?hidden/);
    expect(componentSource).toContain(
      'The complete server-rendered card fallback remains visible.',
    );
    expect(componentSource).not.toContain('client:');
    expect(routeSource).not.toContain('<script');
    expect(routeSource).not.toContain('client:');
  });

  it('does not expose private governance fields or values', () => {
    const publicOutput = JSON.stringify(publicResearchRegistry);
    for (const privateValue of [
      '"governance"',
      '"owner"',
      '"decisionIds"',
      '"reviewStatus"',
      '"contentBoundary"',
      'owner-approved-part-i',
      'William Maddock',
    ]) {
      expect(publicOutput).not.toContain(privateValue);
      expect(componentSource).not.toContain(privateValue);
      expect(routeSource).not.toContain(privateValue);
    }
  });

  it('does not reference the concept image or prohibited graph behavior', () => {
    expect(componentSource).not.toMatch(
      /Research_Observatory_Concept|force-directed|draggable|parallax/,
    );
    expect(routeSource).not.toContain('Research_Observatory_Concept');
  });

  it('keeps RM-2 scoped to the Research route and static component', () => {
    expect(routeSource).toContain(
      "import ContentPageLayout from '../../layouts/ContentPageLayout.astro';",
    );
    expect(routeSource).not.toContain('CosmicBreathCycleExplorer');
    expect(routeSource).not.toContain('CUTimeConverter');
    expect(routeSource).not.toContain('prompt-builder');
    expect(
      new URL('../../../pages/research/index.astro', import.meta.url).pathname,
    ).toMatch(/website\/src\/pages\/research\/index\.astro$/);
  });

  it('uses responsive logical grids and explicit reduced-motion protection', () => {
    expect(componentSource).toContain(
      'grid-template-columns: repeat(2, minmax(0, 1fr))',
    );
    expect(componentSource).toContain(
      '@media (max-width: 48rem)',
    );
    expect(componentSource).toContain('grid-template-columns: 1fr');
    expect(componentSource).toContain('overflow-wrap: anywhere');
    expect(componentSource).toContain('@media (prefers-reduced-motion: reduce)');
    expect(componentSource).toContain('animation: none !important');
    expect(componentSource).not.toMatch(/@keyframes|animation-name|animation-duration/);
  });

  it('reveals semantic Map and Cards controls only after enhancement succeeds', () => {
    expect(componentSource).toContain('data-view-control="map"');
    expect(componentSource).toContain('data-view-control="cards"');
    expect(componentSource).toContain('aria-pressed="false"');
    expect(componentSource).toContain('aria-pressed="true"');
    expect(componentSource).toContain("root.dataset.enhanced = 'true'");
    expect(componentSource.indexOf('render();')).toBeLessThan(
      componentSource.indexOf("root.dataset.enhanced = 'true'"),
    );
    expect(componentSource).toContain(
      "window.matchMedia('(max-width: 48rem)').matches",
    );
  });

  it('renders ten map-node controls plus a distinct central overview control', () => {
    expect(publicResearchRegistry.nodes.filter(
      (node) => node.role !== 'overview',
    )).toHaveLength(10);
    expect(componentSource).toContain('researchNodes.map((node)');
    expect(componentSource).toContain('data-map-node-id={node.id}');
    expect(componentSource).toContain('data-map-node-id={overview.id}');
    expect(componentSource).toContain(
      'class="research-map-node research-map-node--overview"',
    );
  });

  it('renders all six authority-derived filter controls with pressed semantics', () => {
    expect(publicResearchRegistry.filterGroups.map((filter) => filter.label))
      .toEqual([
        'All Research',
        'CU Models',
        'Philosophy',
        'Empirical References',
        'Open Questions',
        'Historical',
      ]);
    expect(componentSource).toContain(
      'publicResearchRegistry.filterGroups.map((filter)',
    );
    expect(componentSource).toContain('data-filter-control={filter.id}');
    expect(componentSource).toContain(
      "aria-pressed={filter.id === neutralState.filterId ? 'true' : 'false'}",
    );
  });

  it('uses the approved pure state contract to derive browser-safe behavior', () => {
    for (const functionName of [
      'applyResearchFilter',
      'createNeutralResearchState',
      'createResearchStateFromFragment',
      'getMatchingResearchNodeIds',
      'getResearchKeyboardNeighbor',
      'getResearchRelationships',
      'parseResearchFragment',
      'restoreCentralObservatoryState',
      'selectResearchNode',
    ]) {
      expect(componentSource).toContain(functionName);
    }
    expect(componentSource).toContain(
      'serializePublicResearchRegistry(browserPublicRegistry)',
    );
    expect(componentSource).toContain('data-research-public-registry');
    expect(componentSource).toContain('data-research-state-contract');
    expect(componentSource).toContain(
      'stateContract.selectionStates[state.filterId]?.[nodeId]',
    );
    expect(componentSource).toContain(
      'stateContract.filterStates[state.selectedNodeId]?.[filterId]',
    );
  });

  it('keeps the browser payload restricted to the safe public projection', () => {
    const serialized = serializePublicResearchRegistry();
    expect(componentSource).toContain(
      'set:html={serializedPublicRegistry}',
    );
    for (const privateValue of [
      '"governance"',
      '"owner"',
      '"decisionIds"',
      '"reviewStatus"',
      '"contentBoundary"',
      'owner-approved-part-i',
      'William Maddock',
    ]) {
      expect(serialized).not.toContain(privateValue);
    }
    expect(componentSource).not.toContain('researchObservatoryAuthority');
  });

  it('provides synchronized neutral and selected detail templates', () => {
    expect(componentSource).toContain('data-detail-panel');
    expect(componentSource).toContain('data-detail-template={overview.id}');
    expect(componentSource).toContain('data-detail-template={node.id}');
    expect(componentSource).toContain(
      'detailBody.replaceChildren(template.content.cloneNode(true))',
    );
    expect(componentSource).toContain('node.primaryDestination &&');
    expect(componentSource).toContain('node.primaryDestination.href');
    expect(componentSource).toContain(
      'node.governingSourceDestination &&',
    );
    expect(componentSource).toContain('node.governingSourceDestination.href');
  });

  it('surfaces selectedOutsideFilter and preserves selected details', () => {
    expect(componentSource).toContain('selectedOutsideFilter');
    expect(componentSource).toContain('data-outside-filter-notice');
    expect(componentSource).toContain(
      'This selected area is outside the current filter.',
    );
    expect(componentSource).toContain('data-restore-all-research');
    expect(componentSource).toContain("state = withFilter('all')");
  });

  it('provides polite match, selection, and view announcements', () => {
    expect(componentSource.match(/aria-live="polite"/g)).toHaveLength(3);
    expect(componentSource).toContain('data-match-announcement');
    expect(componentSource).toContain('data-selection-announcement');
    expect(componentSource).toContain('data-view-announcement');
  });

  it('wires valid fragments, neutral fallback, and history restoration', () => {
    expect(componentSource).toContain('parseResearchFragment');
    expect(componentSource).toContain('getFragmentNodeId(window.location.hash)');
    expect(componentSource).toContain('window.history.pushState');
    expect(componentSource).toContain(
      "window.addEventListener('popstate', syncFromLocation)",
    );
    expect(componentSource).toContain(
      "window.addEventListener('hashchange', syncFromLocation)",
    );
    expect(componentSource).not.toContain('window.location.hash =');
    expect(componentSource).not.toContain('replaceState(');
  });

  it('synchronizes only empty and recognized Observatory fragments', () => {
    const historyFunction = componentSource.slice(
      componentSource.indexOf('const syncFromLocation = () =>'),
      componentSource.indexOf("root.addEventListener('click'"),
    );
    const provenanceGuard =
      'if (!fragmentNodeId && locationHash.slice(1).trim().length > 0)';

    expect(historyFunction).toContain(
      'const locationHash = window.location.hash;',
    );
    expect(historyFunction).toContain(
      'const fragmentNodeId = getFragmentNodeId(locationHash);',
    );
    expect(historyFunction).toContain(provenanceGuard);
    expect(historyFunction.indexOf('return;')).toBeGreaterThan(
      historyFunction.indexOf(provenanceGuard),
    );
    expect(historyFunction.indexOf('return;')).toBeLessThan(
      historyFunction.indexOf('restoreOverview()'),
    );
    expect(historyFunction.indexOf('return;')).toBeLessThan(
      historyFunction.indexOf('render(selectionChanged)'),
    );
    expect(historyFunction).toContain('? withSelection(fragmentNodeId)');
    expect(historyFunction).toContain(': restoreOverview()');
    expect(historyFunction).toContain('render(selectionChanged)');
    expect(historyFunction).not.toMatch(
      /scrollTo|scrollIntoView|setTimeout|\.focus\(|location\.hash\s*=|replaceState/,
    );
    expect(componentSource).toContain(
      "window.addEventListener('popstate', syncFromLocation)",
    );
    expect(componentSource).toContain(
      "window.addEventListener('hashchange', syncFromLocation)",
    );
    expect(componentSource).toContain(
      'const initialFragmentNodeId = getFragmentNodeId(window.location.hash);',
    );
  });

  it('keeps selection and filter changes from moving detail focus', () => {
    expect(componentSource).not.toMatch(
      /detailBody\.focus|detailPanel\.focus/,
    );
    expect(componentSource).not.toContain('filterControl.focus');
    expect(componentSource).not.toContain('viewControl.focus');
  });

  it('restores focus to All Research only after its disappearing reset control renders', () => {
    const restoreBranch = componentSource.slice(
      componentSource.indexOf(
        'if (restoreAll && root.contains(restoreAll))',
      ),
      componentSource.indexOf(
        "mapControls.addEventListener('keydown'",
      ),
    );
    const applyIndex = restoreBranch.indexOf("state = withFilter('all')");
    const renderIndex = restoreBranch.indexOf('render(false, true)');
    const lookupIndex = restoreBranch.indexOf(
      "root.querySelector<HTMLButtonElement>(\n          '[data-filter-control=\"all\"]'",
    );
    const focusIndex = restoreBranch.indexOf('allResearchControl?.focus()');

    expect(applyIndex).toBeGreaterThan(-1);
    expect(renderIndex).toBeGreaterThan(applyIndex);
    expect(lookupIndex).toBeGreaterThan(renderIndex);
    expect(focusIndex).toBeGreaterThan(lookupIndex);
    expect(restoreBranch).not.toContain('setTimeout');
  });

  it('limits automatic focus repair to Restore All Research', () => {
    const ordinaryFilterBranch = componentSource.slice(
      componentSource.indexOf(
        'if (filterControl && root.contains(filterControl))',
      ),
      componentSource.indexOf(
        "const selectControl = target.closest<HTMLButtonElement>",
      ),
    );
    const selectionFunction = componentSource.slice(
      componentSource.indexOf('const selectNode = ('),
      componentSource.indexOf('const restoreNeutral = ('),
    );
    const historyFunction = componentSource.slice(
      componentSource.indexOf('const syncFromLocation = () =>'),
      componentSource.indexOf("root.addEventListener('click'"),
    );

    expect(ordinaryFilterBranch).not.toContain('.focus()');
    expect(selectionFunction).not.toContain('.focus()');
    expect(historyFunction).not.toContain('.focus()');
    expect(componentSource).not.toContain('detailPanel.focus');
    expect(componentSource).not.toContain('detailBody.focus');
  });

  it('preserves the selected node when Restore All Research resets the filter', () => {
    const selectedOutsideFilter = applyResearchFilter(
      selectResearchNode(createNeutralResearchState(), 'ai-alignment'),
      'empirical-references',
    );
    const restored = applyResearchFilter(selectedOutsideFilter, 'all');

    expect(selectedOutsideFilter.selectedOutsideFilter).toBe(true);
    expect(restored).toEqual({
      selectedNodeId: 'ai-alignment',
      filterId: 'all',
      selectedOutsideFilter: false,
    });
  });

  it('consumes every scoped Arrow command before deterministic neighbor lookup', () => {
    const keydownStart = componentSource.indexOf(
      "mapControls.addEventListener('keydown'",
    );
    const keydownEnd = componentSource.indexOf(
      "window.addEventListener('popstate'",
      keydownStart,
    );
    const keydownBranch = componentSource.slice(keydownStart, keydownEnd);
    const arrowEnd = keydownBranch.indexOf(
      "if (event.key === 'Enter' || event.key === ' ')",
    );
    const arrowBranch = keydownBranch.slice(0, arrowEnd);

    expect(keydownStart).toBeGreaterThan(-1);
    expect(keydownEnd).toBeGreaterThan(keydownStart);
    for (const key of [
      'ArrowUp',
      'ArrowRight',
      'ArrowDown',
      'ArrowLeft',
    ]) {
      expect(arrowBranch).toContain(key);
    }

    const preventDefaultIndex = arrowBranch.indexOf('event.preventDefault()');
    const neighborLookupIndex = arrowBranch.indexOf(
      'registry.keyboardNavigation[nodeId]?.[event.key]',
    );
    const focusConditionIndex = arrowBranch.indexOf(
      'if (neighbor && neighborId !== nodeId)',
    );
    const focusIndex = arrowBranch.indexOf('neighbor.focus()');

    expect(preventDefaultIndex).toBeGreaterThan(-1);
    expect(neighborLookupIndex).toBeGreaterThan(preventDefaultIndex);
    expect(focusConditionIndex).toBeGreaterThan(neighborLookupIndex);
    expect(focusIndex).toBeGreaterThan(focusConditionIndex);
    expect(arrowBranch.match(/event\.preventDefault\(\)/g)).toHaveLength(1);
  });

  it('retains edge-node focus without restoring native Arrow scrolling', () => {
    expect(getResearchKeyboardNeighbor('sources', 'ArrowUp')).toBe('sources');
    expect(getResearchKeyboardNeighbor('history', 'ArrowDown')).toBe('history');

    const arrowBranch = componentSource.slice(
      componentSource.indexOf("if (\n        event.key === 'ArrowUp'"),
      componentSource.indexOf(
        "if (event.key === 'Enter' || event.key === ' ')",
      ),
    );
    expect(arrowBranch).toContain('event.preventDefault()');
    expect(arrowBranch).toContain(
      'registry.keyboardNavigation[nodeId]?.[event.key] ?? nodeId',
    );
    expect(arrowBranch).toContain(
      'if (neighbor && neighborId !== nodeId)',
    );
    expect(arrowBranch).not.toContain('target.blur()');
    expect(arrowBranch).not.toContain('replaceWith');
    expect(arrowBranch).not.toContain('cloneNode');
  });

  it('keeps map keyboard handling scoped and preserves non-Arrow contracts', () => {
    const keydownBranch = componentSource.slice(
      componentSource.indexOf("mapControls.addEventListener('keydown'"),
      componentSource.indexOf("window.addEventListener('popstate'"),
    );
    expect(keydownBranch).toContain(
      'if (!(target instanceof HTMLButtonElement)) return;',
    );
    expect(keydownBranch).toContain(
      'const nodeId = target.dataset.mapNodeId;',
    );
    expect(keydownBranch).toContain('if (!nodeId) return;');
    for (const key of ['Enter', "' '", 'Home', 'Escape']) {
      expect(keydownBranch).toContain(key);
    }
    expect(keydownBranch).not.toContain("event.key === 'Tab'");
    expect(componentSource).not.toContain(
      "window.addEventListener('keydown'",
    );
    expect(componentSource).not.toContain(
      "document.addEventListener('keydown'",
    );
    expect(keydownBranch).not.toContain('setTimeout');
    expect(keydownBranch).not.toContain('cloneNode');
    expect(keydownBranch).not.toContain('mapPanel.hidden');
    expect(keydownBranch).not.toContain('cardsPanel.hidden');
  });

  it('restores full opacity only while a nonmatching map node is focus-visible', () => {
    const filteredStyle = componentSource.slice(
      componentSource.indexOf(
        ".research-map-node[data-matching='false'] {",
      ),
      componentSource.indexOf(
        ".research-map-node[data-related='true'] {",
      ),
    );
    expect(filteredStyle).toContain(
      ".research-map-node[data-matching='false'] {",
    );
    expect(filteredStyle).toContain('border-style: dashed');
    expect(filteredStyle).toContain('color: var(--color-text-muted)');
    expect(filteredStyle).toContain('opacity: 0.55');
    expect(filteredStyle).toContain(
      ".research-map-node[data-matching='false']:focus-visible {",
    );
    expect(filteredStyle).toContain('opacity: 1');
    expect(componentSource).toContain(
      'root.dataset.selectedOutsideFilter = String(',
    );
    expect(componentSource).toContain(
      'control.dataset.matching = String(isMatching)',
    );
  });

  it('preserves approved relationships as accessible text alongside the SVG', () => {
    expect(componentSource).toContain('getResearchRelationships(');
    expect(componentSource).toContain('Approved relationships');
    expect(componentSource).toContain(
      'relationship.publicExplanation',
    );
    expect(componentSource).toContain('<svg');
    expect(componentSource).not.toContain('createElementNS');
  });

  it('uses the same public research-node records for Map and Cards', () => {
    expect(componentSource).toContain(
      "const researchNodes = publicResearchRegistry.nodes.filter",
    );
    expect(componentSource).toContain('researchNodes.map((node)');
    expect(componentSource).toContain(
      "nodes: researchNodes.filter((node) => node.role === 'primary')",
    );
    expect(componentSource).toContain(
      "nodes: researchNodes.filter((node) => node.role === 'supporting')",
    );
  });

  it('does not add persistence, telemetry, or unsafe HTML construction', () => {
    expect(componentSource).not.toMatch(
      /localStorage|sessionStorage|document\.cookie|analytics|telemetry/,
    );
    expect(componentSource).not.toContain('innerHTML');
    expect(componentSource).not.toContain('insertAdjacentHTML');
  });

  it('renders a decorative noninteractive relationship SVG only inside the hidden enhanced map', () => {
    const mapStart = componentSource.indexOf(
      'class="research-observatory__map"',
    );
    const svgStart = componentSource.indexOf(
      'class="research-observatory__relationship-layer"',
    );
    const mapEnd = componentSource.indexOf('</section>', svgStart);
    expect(mapStart).toBeGreaterThan(-1);
    expect(svgStart).toBeGreaterThan(mapStart);
    expect(mapEnd).toBeGreaterThan(svgStart);
    expect(componentSource.slice(mapStart, svgStart)).toContain('hidden');
    expect(componentSource).toContain('aria-hidden="true"');
    expect(componentSource).toContain('focusable="false"');
    expect(componentSource).toContain('tabindex="-1"');
    expect(componentSource).toContain('pointer-events: none');
  });

  it('derives every visual path from projected relationship records', () => {
    expect(publicResearchRegistry.relationships).toHaveLength(33);
    expect(componentSource).toContain(
      'publicResearchRegistry.relationships.map(',
    );
    expect(componentSource).toContain(
      'data-relationship-count={relationshipPaths.length}',
    );
    expect(componentSource).toContain(
      'data-relationship-source={relationship.sourceId}',
    );
    expect(componentSource).toContain(
      'data-relationship-target={relationship.targetId}',
    );
    expect(componentSource).not.toMatch(
      /const\s+(relationships|relationshipPairs|relationshipRecords)\s*=\s*\[/,
    );
  });

  it('provides strong, related, and support line styles with non-color patterns', () => {
    expect(new Set(
      publicResearchRegistry.relationships.map(
        (relationship) => relationship.kind,
      ),
    )).toEqual(new Set(['strong', 'related', 'support']));
    expect(componentSource).toContain(
      '`research-relationship--${relationship.kind}`',
    );
    expect(componentSource).toContain(
      '.research-relationship--strong',
    );
    expect(componentSource).toContain(
      '.research-relationship--related',
    );
    expect(componentSource).toContain(
      '.research-relationship--support',
    );
    expect(componentSource).toContain('stroke-dasharray: 10 8');
    expect(componentSource).toContain('stroke-dasharray: 2 9');
  });

  it('renders a readable relationship legend and non-ranking explanation', () => {
    for (const label of [
      'Relationship legend',
      'Strong connection',
      'Related connection',
      'Support/context',
      'Line color and pattern identify relationship type only.',
    ]) {
      expect(componentSource).toContain(label);
    }
    for (const boundary of [
      'authority',
      'certainty',
      'value',
      'importance',
      'duration',
      'chronology',
    ]) {
      expect(componentSource).toContain(boundary);
    }
  });

  it('uses one fixed authored position for every semantic map control', () => {
    for (const node of publicResearchRegistry.nodes) {
      expect(componentSource).toContain(`'${node.id}': {`);
    }
    expect(componentSource).toContain(
      'data-authored-node-count={publicResearchRegistry.nodes.length}',
    );
    expect(componentSource).toContain('style={nodePositionStyle(overview.id)}');
    expect(componentSource).toContain('style={nodePositionStyle(node.id)}');
    expect(componentSource).not.toMatch(
      /Math\.random|requestAnimationFrame|forceSimulation|drag\(/,
    );
  });

  it('keeps the central Observatory visually and semantically distinct', () => {
    expect(componentSource).toContain('data-map-role="overview"');
    expect(componentSource).toContain('Navigation center');
    expect(componentSource).toContain(
      '.research-map-node--overview',
    );
    expect(componentSource).toContain('border-style: double');
    expect(componentSource).toContain(
      'do not represent chronology, physical',
    );
  });

  it('updates direct, unrelated, and filter relationship states without removing paths', () => {
    expect(componentSource).toContain('data-direct="false"');
    expect(componentSource).toContain('data-unrelated="false"');
    expect(componentSource).toContain('data-filtered-outside="false"');
    expect(componentSource).toContain(
      'relationship.dataset.direct = String(isDirect)',
    );
    expect(componentSource).toContain(
      'relationship.dataset.unrelated = String(isUnrelated)',
    );
    expect(componentSource).toContain(
      'relationship.dataset.filteredOutside = String(isFilteredOutside)',
    );
    expect(componentSource).not.toContain('relationship.remove()');
  });

  it('marks directly related nodes while preserving selected semantics', () => {
    expect(componentSource).toContain('const relatedNodeIds = new Set<string>()');
    expect(componentSource).toContain(
      'control.dataset.related = String(isRelated)',
    );
    expect(componentSource).toContain(
      "control.setAttribute('aria-pressed', String(isSelected))",
    );
    expect(componentSource).toContain(
      ".research-map-node[data-related='true']",
    );
    expect(componentSource).toContain(
      ".research-map-node[aria-pressed='true']",
    );
  });

  it('uses a wide map/detail composition with a full-width matrix row', () => {
    expect(componentSource).toContain(
      `grid-template-columns:
      minmax(0, 1.42fr)
      minmax(20rem, 1fr)`,
    );
    expect(componentSource).toContain(
      `grid-template-areas:
      'map detail'
      'matrix matrix';`,
    );
    expect(componentSource).toContain(
      '.research-observatory__workspace :global(.research-matrix)',
    );
    expect(componentSource).toContain('grid-area: matrix');
    expect(componentSource).toContain('@media (max-width: 72rem)');
    expect(componentSource).toContain(
      "grid-template-areas:\n        'map'\n        'detail'\n        'matrix'",
    );
  });

  it('degrades authored geometry to a logical grid at narrower widths', () => {
    expect(componentSource).toContain('@media (max-width: 64rem)');
    expect(componentSource).toContain(
      '.research-observatory__relationship-layer {\n      display: none;',
    );
    expect(componentSource).toContain('position: static');
    expect(componentSource).toContain('translate: none');
    expect(componentSource).toContain(
      'grid-template-columns: repeat(2, minmax(0, 1fr))',
    );
    expect(componentSource).toContain('grid-column: span 2');
  });

  it('keeps Cards and no-JavaScript output independent of the SVG', () => {
    expect(componentSource).toContain(
      "mapPanel.hidden = activeView !== 'map'",
    );
    expect(componentSource).toContain(
      "cardsPanel.hidden = activeView !== 'cards'",
    );
    expect(componentSource.indexOf('data-cards-panel')).toBeLessThan(
      componentSource.indexOf('data-detail-template={overview.id}'),
    );
    expect(componentSource).toContain(
      'The complete server-rendered card fallback remains visible.',
    );
  });

  it('uses restrained transitions with complete reduced-motion overrides', () => {
    expect(componentSource).toContain('transition: opacity 140ms ease');
    expect(componentSource).toContain(
      '@media (prefers-reduced-motion: reduce)',
    );
    expect(componentSource).toContain('transition: none !important');
    expect(componentSource).toContain(
      'stroke-dashoffset: 0 !important',
    );
    expect(componentSource).not.toMatch(
      /@keyframes|animation-name|stroke-dashoffset:\s*-[0-9]|orbit|pulse|spin/,
    );
  });

  it('visually distinguishes primary and governing-source actions', () => {
    expect(componentSource).toContain(
      'research-card__action--primary',
    );
    expect(componentSource).toContain(
      'research-card__action--source',
    );
    expect(componentSource).toContain(
      '.research-card__action--primary a',
    );
    expect(componentSource).toContain(
      '.research-card__action--source a',
    );
  });
});
