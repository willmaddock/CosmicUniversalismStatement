import { describe, expect, it } from 'vitest';
import rawAuthority from '../../../data/research/CU-RESEARCH-OBSERVATORY-1.0.json';
import {
  createPublicResearchRegistry,
  parseResearchObservatoryAuthority,
  publicResearchRegistry,
  researchObservatoryAuthority,
  serializePublicResearchRegistry,
} from '../research-observatory';
import {
  researchClassifications,
  researchStatuses,
} from '../../../data/research-taxonomy';

const cloneAuthority = (): Record<string, any> =>
  structuredClone(rawAuthority) as Record<string, any>;

const getAuthorityNode = (nodeId: string) => {
  const node = researchObservatoryAuthority.nodes.find(
    (candidate) => candidate.id === nodeId,
  );
  expect(node).toBeDefined();
  return node!;
};

describe('Research Observatory authority', () => {
  it('loads the exact versioned authority and approved node roles', () => {
    expect(researchObservatoryAuthority.authorityId).toBe(
      'CU-RESEARCH-OBSERVATORY-1.0',
    );
    expect(researchObservatoryAuthority.version).toBe('1.0');
    expect(researchObservatoryAuthority.overviewNodeId).toBe(
      'research-observatory',
    );
    expect(researchObservatoryAuthority.nodes).toHaveLength(11);
    expect(
      researchObservatoryAuthority.nodes.filter(
        (node) => node.role === 'overview',
      ),
    ).toHaveLength(1);
    expect(
      researchObservatoryAuthority.nodes.filter(
        (node) => node.role === 'primary',
      ),
    ).toHaveLength(6);
    expect(
      researchObservatoryAuthority.nodes.filter(
        (node) => node.role === 'supporting',
      ),
    ).toHaveLength(4);
  });

  it('uses only the live research classifications and lifecycle statuses', () => {
    const classificationIds = new Set(Object.keys(researchClassifications));
    const statusIds = new Set(Object.keys(researchStatuses));
    for (const node of researchObservatoryAuthority.nodes) {
      expect(statusIds.has(node.status)).toBe(true);
      expect(
        node.classifications.every((classification) =>
          classificationIds.has(classification)),
      ).toBe(true);
    }
    expect(Object.keys(researchStatuses)).toEqual([
      'foundational',
      'active-research',
      'provisional',
      'under-review',
      'open-problem',
      'superseded',
      'archived',
    ]);
  });

  it('contains unique IDs, safe destinations, and complete keyboard records', () => {
    const nodeIds = researchObservatoryAuthority.nodes.map((node) => node.id);
    expect(new Set(nodeIds).size).toBe(11);
    expect(Object.keys(researchObservatoryAuthority.keyboardNavigation).sort())
      .toEqual([...nodeIds].sort());

    for (const node of researchObservatoryAuthority.nodes) {
      expect(node.summary.trim().length).toBeGreaterThan(0);
      for (const destination of [
        node.primaryDestination,
        node.governingSourceDestination,
      ].filter((candidate) => candidate !== undefined)) {
        if (destination.kind === 'internal') {
          expect(destination.path).not.toMatch(/^\/|^[a-z]+:|\.\./i);
        } else {
          expect(destination.href).toMatch(/^https:\/\//);
          expect(destination.externalLabel).toMatch(/external/i);
        }
      }
    }
  });

  it('preserves the Cosmic Breath source-action slot while migrating it internally', () => {
    const breath = getAuthorityNode('cosmic-breath');
    expect(breath.id).toBe('cosmic-breath');
    expect(breath.governingSourceDestination).toEqual({
      kind: 'internal',
      path: 'research#cosmic-breath-provenance',
      label: 'Review Cosmic Breath sources and provenance',
    });

    const projection = createPublicResearchRegistry(
      (path) => `/CosmicUniversalismStatement/${path}`,
    );
    const projectedBreath = projection.nodes.find(
      (node) => node.id === 'cosmic-breath',
    );
    expect(projectedBreath?.governingSourceDestination).toEqual({
      kind: 'internal',
      href:
        '/CosmicUniversalismStatement/research#cosmic-breath-provenance',
      label: 'Review Cosmic Breath sources and provenance',
      external: false,
    });
    expect(projectedBreath?.governingSourceDestination).not.toHaveProperty(
      'externalLabel',
    );
    expect(projectedBreath?.governingSourceDestination).not.toHaveProperty(
      'opensInNewTab',
    );
    expect(projectedBreath?.governingSourceDestination).not.toHaveProperty(
      'rel',
    );
    expect(JSON.stringify(researchObservatoryAuthority)).not.toContain(
      'View the Cosmic Breath source',
    );
    expect(JSON.stringify(researchObservatoryAuthority)).not.toContain(
      'Cosmic_Breath_Calculation.md',
    );
  });

  it('uses the approved Methods and Historical Archive destination roles', () => {
    const methods = getAuthorityNode('methods');
    expect(methods.primaryDestination).toEqual({
      kind: 'internal',
      path: 'research#source-provenance',
      label: 'Review source and provenance standards',
    });
    expect(methods.governingSourceDestination).toEqual({
      kind: 'external',
      href: 'https://github.com/willmaddock/CosmicUniversalismStatement/blob/website/documentation/master-manual/Cosmic_Universalism_Website_Master_Implementation_Manual_v1.1.pdf',
      label: 'Open Website Master Manual PDF',
      externalLabel:
        'Open Website Master Manual PDF (external) — opens in a new tab',
    });
    const methodsSource = methods.governingSourceDestination;
    expect(methodsSource?.kind).toBe('external');
    if (methodsSource?.kind === 'external') {
      expect(methodsSource.href).not.toMatch(/\.tex$/);
    }

    const history = getAuthorityNode('history');
    expect(history.primaryDestination).toEqual({
      kind: 'internal',
      path: 'research#historical-record',
      label: 'Open Historical Archive',
    });
    expect(history.governingSourceDestination).toEqual({
      kind: 'external',
      href: 'https://github.com/willmaddock/CosmicUniversalismStatement/tags',
      label: 'View repository tag history',
      externalLabel:
        'View repository tag history (external) — opens in a new tab',
    });
  });

  it('uses one relevant source action for Empirical Science and Consciousness', () => {
    const empirical = getAuthorityNode('empirical-science');
    expect(empirical.primaryDestination).toBeUndefined();
    expect(empirical.governingSourceDestination).toEqual({
      kind: 'external',
      href: 'https://github.com/willmaddock/CosmicUniversalismStatement/blob/website/Docs/CU_Framework.md',
      label: 'View the CU Framework source',
      externalLabel:
        'View the CU Framework source on GitHub (external) — opens in a new tab',
    });

    const consciousness = getAuthorityNode('consciousness');
    expect(consciousness.primaryDestination).toBeUndefined();
    expect(consciousness.governingSourceDestination).toEqual({
      kind: 'external',
      href: 'https://github.com/willmaddock/CosmicUniversalismStatement/blob/main/ResearchFiles/CU_Consciousness.md',
      label: 'View the CU Consciousness source',
      externalLabel:
        'View the CU Consciousness source on GitHub (external) — opens in a new tab',
    });
  });

  it('removes generic fill actions while retaining meaningful destinations', () => {
    const aiAlignment = getAuthorityNode('ai-alignment');
    expect(aiAlignment.primaryDestination).toBeUndefined();
    expect(aiAlignment.governingSourceDestination).toMatchObject({
      kind: 'external',
      href: expect.stringMatching(/LLMtrainingInegration\/AivsCU\.md$/),
    });

    const openQuestions = getAuthorityNode('open-questions');
    expect(openQuestions.primaryDestination).toEqual({
      kind: 'internal',
      path: 'research#open-questions',
      label: 'Review open questions',
    });
    expect(openQuestions.governingSourceDestination).toBeUndefined();

    const serialized = JSON.stringify(researchObservatoryAuthority.nodes);
    expect(serialized).not.toContain(
      '/tree/main/LLMtrainingInegration',
    );
    expect(serialized).not.toContain('/tree/main/ResearchFiles');
  });

  it('does not duplicate any node’s primary and governing destination', () => {
    const destinationKey = (
      destination: NonNullable<
        | typeof researchObservatoryAuthority.nodes[number]['primaryDestination']
        | typeof researchObservatoryAuthority.nodes[number]['governingSourceDestination']
      >,
    ) => destination.kind === 'internal'
      ? `internal:${destination.path}`
      : `external:${destination.href}`;

    for (const node of researchObservatoryAuthority.nodes) {
      const actions = [
        node.primaryDestination,
        node.governingSourceDestination,
      ].filter((destination) => destination !== undefined);
      if (node.role !== 'overview') {
        expect(actions.length).toBeGreaterThanOrEqual(1);
        expect(actions.length).toBeLessThanOrEqual(2);
      }
      if (actions.length === 2) {
        expect(
          destinationKey(actions[0]),
          `${node.id} must have distinct public actions`,
        ).not.toBe(destinationKey(actions[1]));
      }
    }
  });

  it('does not add unapproved Part II chronology claims', () => {
    const publicCopy = JSON.stringify(researchObservatoryAuthority.nodes);
    expect(publicCopy).not.toMatch(
      /cesium|planck|nasa|age[- ]offset|base-10 placeholder/i,
    );
  });

  it('contains valid unique undirected relationships with explanations', () => {
    const nodeIds = new Set(
      researchObservatoryAuthority.nodes.map((node) => node.id),
    );
    const pairs = researchObservatoryAuthority.relationships.map(
      (relationship) => {
        expect(nodeIds.has(relationship.sourceId)).toBe(true);
        expect(nodeIds.has(relationship.targetId)).toBe(true);
        expect(relationship.sourceId).not.toBe(relationship.targetId);
        expect(relationship.publicExplanation.trim().length).toBeGreaterThan(0);
        return [relationship.sourceId, relationship.targetId].sort().join('::');
      },
    );
    expect(new Set(pairs).size).toBe(pairs.length);
  });

  it('rejects duplicate IDs and invalid roles, classifications, and statuses', () => {
    const duplicate = cloneAuthority();
    duplicate.nodes[1].id = duplicate.nodes[0].id;
    expect(() => parseResearchObservatoryAuthority(duplicate)).toThrow(
      /unique IDs/,
    );

    const invalidRole = cloneAuthority();
    invalidRole.nodes[1].role = 'featured';
    expect(() => parseResearchObservatoryAuthority(invalidRole)).toThrow(
      /approved node role/,
    );

    const invalidClassification = cloneAuthority();
    invalidClassification.nodes[1].classifications = ['scientific-proof'];
    expect(() =>
      parseResearchObservatoryAuthority(invalidClassification),
    ).toThrow(/existing research classification/);

    const invalidStatus = cloneAuthority();
    invalidStatus.nodes[1].status = 'complete';
    expect(() => parseResearchObservatoryAuthority(invalidStatus)).toThrow(
      /existing research status/,
    );
  });

  it('rejects invalid, dangling, and duplicate relationships', () => {
    const invalidKind = cloneAuthority();
    invalidKind.relationships[0].kind = 'causes';
    expect(() => parseResearchObservatoryAuthority(invalidKind)).toThrow(
      /approved relationship kind/,
    );

    const dangling = cloneAuthority();
    dangling.relationships[0].targetId = 'missing-node';
    expect(() => parseResearchObservatoryAuthority(dangling)).toThrow(
      /reference existing nodes/,
    );

    const duplicate = cloneAuthority();
    duplicate.relationships.push({
      ...duplicate.relationships[0],
      sourceId: duplicate.relationships[0].targetId,
      targetId: duplicate.relationships[0].sourceId,
    });
    expect(() => parseResearchObservatoryAuthority(duplicate)).toThrow(
      /duplicate relationship/,
    );
  });

  it('accepts one or two actions and rejects zero actions for an ordinary node', () => {
    const sourceOnly = cloneAuthority();
    delete sourceOnly.nodes[1].primaryDestination;
    expect(
      parseResearchObservatoryAuthority(sourceOnly).nodes[1].primaryDestination,
    ).toBeUndefined();

    const primaryOnly = cloneAuthority();
    delete primaryOnly.nodes[1].governingSourceDestination;
    expect(
      parseResearchObservatoryAuthority(primaryOnly)
        .nodes[1].governingSourceDestination,
    ).toBeUndefined();

    const noActions = cloneAuthority();
    delete noActions.nodes[1].primaryDestination;
    delete noActions.nodes[1].governingSourceDestination;
    expect(() => parseResearchObservatoryAuthority(noActions)).toThrow(
      /at least one public action/,
    );
  });

  it('rejects missing copy, unsafe present destinations, and unsafe external metadata', () => {
    const missingSummary = cloneAuthority();
    missingSummary.nodes[1].summary = ' ';
    expect(() => parseResearchObservatoryAuthority(missingSummary)).toThrow(
      /non-empty string/,
    );

    const unsafeInternal = cloneAuthority();
    unsafeInternal.nodes[1].primaryDestination.path = '/cosmic-breath';
    expect(() => parseResearchObservatoryAuthority(unsafeInternal)).toThrow(
      /base-path-safe sitePath argument/,
    );

    const unsafeExternal = cloneAuthority();
    unsafeExternal.nodes[2].governingSourceDestination.href =
      'http://example.com/source';
    expect(() => parseResearchObservatoryAuthority(unsafeExternal)).toThrow(
      /must use HTTPS/,
    );

    const undisclosedExternal = cloneAuthority();
    undisclosedExternal.nodes[2].governingSourceDestination.externalLabel =
      'View source';
    expect(() =>
      parseResearchObservatoryAuthority(undisclosedExternal),
    ).toThrow(/disclose external navigation/);

    const nullDestination = cloneAuthority();
    nullDestination.nodes[1].primaryDestination = null;
    expect(() => parseResearchObservatoryAuthority(nullDestination)).toThrow(
      /must be an object/,
    );
  });

  it('rejects invalid keyboard targets and unsupported keyboard keys', () => {
    const dangling = cloneAuthority();
    dangling.keyboardNavigation['research-observatory'].ArrowUp = 'unknown';
    expect(() => parseResearchObservatoryAuthority(dangling)).toThrow(
      /unknown target/,
    );

    const unsupported = cloneAuthority();
    unsupported.keyboardNavigation['research-observatory'].PageDown =
      'free-will';
    expect(() => parseResearchObservatoryAuthority(unsupported)).toThrow(
      /unsupported key/,
    );
  });
});

describe('Research Observatory public projection', () => {
  it('resolves internal paths through the injected sitePath-compatible helper', () => {
    const projection = createPublicResearchRegistry(
      (path) => `/CosmicUniversalismStatement/${path}`,
    );
    expect(projection.targetRoute).toBe(
      '/CosmicUniversalismStatement/research',
    );
    const breath = projection.nodes.find(
      (node) => node.id === 'cosmic-breath',
    );
    expect(breath?.primaryDestination).toEqual({
      kind: 'internal',
      href: '/CosmicUniversalismStatement/cosmic-breath',
      label: 'Explore Cosmic Breath',
      external: false,
    });
  });

  it('adds approved safe metadata to every external destination', () => {
    const destinations = publicResearchRegistry.nodes.flatMap((node) => [
      node.primaryDestination,
      node.governingSourceDestination,
    ]).filter((destination) => destination !== undefined);
    for (const destination of destinations.filter(
      (candidate) => candidate.kind === 'external',
    )) {
      expect(destination).toMatchObject({
        external: true,
        opensInNewTab: true,
        rel: 'noopener noreferrer',
      });
      expect(destination.externalLabel).toMatch(/external/i);
    }
  });

  it('omits missing optional actions without null or placeholder values', () => {
    for (const nodeId of [
      'ai-alignment',
      'consciousness',
      'empirical-science',
    ]) {
      const node = publicResearchRegistry.nodes.find(
        (candidate) => candidate.id === nodeId,
      );
      expect(node?.primaryDestination).toBeUndefined();
      expect(node).not.toHaveProperty('primaryDestination');
      expect(node?.governingSourceDestination).toBeDefined();
    }
    const openQuestions = publicResearchRegistry.nodes.find(
      (node) => node.id === 'open-questions',
    );
    expect(openQuestions?.primaryDestination).toBeDefined();
    expect(openQuestions).not.toHaveProperty('governingSourceDestination');
    expect(JSON.stringify(publicResearchRegistry)).not.toMatch(
      /\"(?:primaryDestination|governingSourceDestination)\":null/,
    );
  });

  it('rejects a public ordinary node with zero actions', () => {
    const invalid = structuredClone(publicResearchRegistry) as Record<
      string,
      any
    >;
    const breath = invalid.nodes.find(
      (node: Record<string, any>) => node.id === 'cosmic-breath',
    );
    delete breath.primaryDestination;
    delete breath.governingSourceDestination;
    expect(() => serializePublicResearchRegistry(invalid)).toThrow(
      /at least one public action/,
    );
  });

  it('excludes all private governance fields', () => {
    const serialized = JSON.stringify(publicResearchRegistry);
    expect(serialized).not.toContain('"governance"');
    expect(serialized).not.toContain('"owner"');
    expect(serialized).not.toContain('"decisionIds"');
    expect(serialized).not.toContain('"reviewStatus"');
    expect(serialized).not.toContain('"contentBoundary"');
  });

  it('rejects contaminated public projections', () => {
    const contaminated = structuredClone(publicResearchRegistry) as Record<
      string,
      any
    >;
    contaminated.governance = { owner: 'private' };
    expect(() => serializePublicResearchRegistry(contaminated)).toThrow(
      /unexpected field "governance"/,
    );

    const contaminatedNode = structuredClone(
      publicResearchRegistry,
    ) as Record<string, any>;
    contaminatedNode.nodes[0].reviewStatus = 'owner-approved-part-i';
    expect(() => serializePublicResearchRegistry(contaminatedNode)).toThrow(
      /unexpected field "reviewStatus"/,
    );

    const contaminatedKeyboard = structuredClone(
      publicResearchRegistry,
    ) as Record<string, any>;
    contaminatedKeyboard.keyboardNavigation['research-observatory']
      .governance = 'private';
    expect(() =>
      serializePublicResearchRegistry(contaminatedKeyboard),
    ).toThrow(/unsupported key "governance"/);
  });

  it('escapes embedded HTML delimiters in serialized public copy', () => {
    const projection = structuredClone(publicResearchRegistry) as Record<
      string,
      any
    >;
    projection.nodes[0].summary =
      '</script><script>alert("not executed")</script>';
    const serialized = serializePublicResearchRegistry(projection);
    expect(serialized).not.toContain('</script>');
    expect(
      (JSON.parse(serialized) as Record<string, any>).nodes[0].summary,
    ).toBe('</script><script>alert("not executed")</script>');
  });
});
