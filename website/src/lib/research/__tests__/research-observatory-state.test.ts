import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { publicResearchRegistry } from '../research-observatory';
import {
  applyResearchFilter,
  countMatchingResearchNodes,
  createNeutralResearchState,
  createResearchStateFromFragment,
  getMatchingResearchNodeIds,
  getRelatedResearchNodeIds,
  getResearchKeyboardNeighbor,
  getResearchRelationships,
  moveResearchKeyboardSelection,
  parseResearchFragment,
  restoreCentralObservatoryState,
  selectResearchNode,
} from '../research-observatory-state';

describe('Research Observatory state', () => {
  it('starts in the neutral central Observatory state', () => {
    expect(createNeutralResearchState()).toEqual({
      selectedNodeId: 'research-observatory',
      filterId: 'all',
      selectedOutsideFilter: false,
    });
  });

  it('selects valid nodes and restores the center for invalid selections', () => {
    const neutral = createNeutralResearchState();
    expect(selectResearchNode(neutral, 'ai-alignment')).toEqual({
      selectedNodeId: 'ai-alignment',
      filterId: 'all',
      selectedOutsideFilter: false,
    });
    expect(selectResearchNode(neutral, 'missing-node')).toEqual(neutral);
  });

  it('parses valid fragments and returns neutral state for absent or invalid fragments', () => {
    expect(parseResearchFragment('#cosmic-breath')).toBe('cosmic-breath');
    expect(
      parseResearchFragment(
        '/CosmicUniversalismStatement/research/#ai-alignment',
      ),
    ).toBe('ai-alignment');
    expect(parseResearchFragment('#research-observatory')).toBeNull();
    expect(parseResearchFragment('#missing')).toBeNull();
    expect(parseResearchFragment('#%E0%A4%A')).toBeNull();

    expect(createResearchStateFromFragment('#cu-time').selectedNodeId).toBe(
      'cu-time',
    );
    expect(createResearchStateFromFragment('#missing')).toEqual(
      createNeutralResearchState(),
    );
  });

  it('filters nodes by the approved classification groups', () => {
    expect(getMatchingResearchNodeIds('all')).toHaveLength(10);
    expect(getMatchingResearchNodeIds('cu-models')).toEqual(
      expect.arrayContaining([
        'cosmic-breath',
        'cu-time',
        'ai-alignment',
        'consciousness',
      ]),
    );
    expect(getMatchingResearchNodeIds('empirical-references')).toEqual([
      'empirical-science',
    ]);
    expect(getMatchingResearchNodeIds('historical')).toEqual(['history']);
    expect(countMatchingResearchNodes('open-questions')).toBe(3);
    expect(getMatchingResearchNodeIds('not-a-filter')).toHaveLength(10);
  });

  it('preserves a selected node outside a new filter with an explicit state flag', () => {
    const selected = selectResearchNode(
      createNeutralResearchState(),
      'ai-alignment',
    );
    const empirical = applyResearchFilter(
      selected,
      'empirical-references',
    );
    expect(empirical).toEqual({
      selectedNodeId: 'ai-alignment',
      filterId: 'empirical-references',
      selectedOutsideFilter: true,
    });

    const philosophy = applyResearchFilter(empirical, 'philosophy');
    expect(philosophy.selectedNodeId).toBe('ai-alignment');
    expect(philosophy.selectedOutsideFilter).toBe(false);
  });

  it('returns relationships and related nodes from either endpoint', () => {
    const relationships = getResearchRelationships('cosmic-breath');
    expect(relationships.length).toBeGreaterThan(0);
    expect(
      relationships.every(
        (relationship) =>
          relationship.sourceId === 'cosmic-breath'
          || relationship.targetId === 'cosmic-breath',
      ),
    ).toBe(true);
    expect(getRelatedResearchNodeIds('cosmic-breath')).toEqual(
      expect.arrayContaining([
        'cu-time',
        'empirical-science',
        'consciousness',
        'sources',
        'methods',
        'open-questions',
        'history',
      ]),
    );
    expect(getResearchRelationships('missing')).toEqual([]);
  });

  it('uses the explicit deterministic keyboard-neighbor table', () => {
    expect(
      getResearchKeyboardNeighbor('research-observatory', 'ArrowUp'),
    ).toBe('cosmic-breath');
    expect(getResearchKeyboardNeighbor('cu-time', 'ArrowRight')).toBe(
      'methods',
    );
    expect(getResearchKeyboardNeighbor('history', 'ArrowRight')).toBe(
      'history',
    );

    const selected = selectResearchNode(
      createNeutralResearchState(),
      'research-observatory',
    );
    expect(
      moveResearchKeyboardSelection(selected, 'ArrowRight').selectedNodeId,
    ).toBe('ai-alignment');
  });

  it('restores the central node while preserving a valid active filter', () => {
    const selected = applyResearchFilter(
      selectResearchNode(createNeutralResearchState(), 'ai-alignment'),
      'empirical-references',
    );
    expect(restoreCentralObservatoryState(selected)).toEqual({
      selectedNodeId: 'research-observatory',
      filterId: 'empirical-references',
      selectedOutsideFilter: false,
    });
    expect(moveResearchKeyboardSelection(selected, 'Home')).toEqual({
      selectedNodeId: 'research-observatory',
      filterId: 'empirical-references',
      selectedOutsideFilter: false,
    });
  });

  it('models fragment selection, filter conflict, and history restoration without DOM state', () => {
    const selectedFromFragment =
      createResearchStateFromFragment('#ai-alignment');
    const outsideFilter = applyResearchFilter(
      selectedFromFragment,
      'empirical-references',
    );
    expect(outsideFilter).toEqual({
      selectedNodeId: 'ai-alignment',
      filterId: 'empirical-references',
      selectedOutsideFilter: true,
    });

    const restoredFromHistory =
      createResearchStateFromFragment('#cosmic-breath');
    expect(restoredFromHistory).toEqual({
      selectedNodeId: 'cosmic-breath',
      filterId: 'all',
      selectedOutsideFilter: false,
    });
    expect(createResearchStateFromFragment('')).toEqual(
      createNeutralResearchState(),
    );
  });

  it('defines deterministic neighbors for every map node and supported arrow key', () => {
    const keys = [
      'ArrowUp',
      'ArrowRight',
      'ArrowDown',
      'ArrowLeft',
    ] as const;
    for (const node of publicResearchRegistry.nodes) {
      for (const key of keys) {
        const neighbor = getResearchKeyboardNeighbor(node.id, key);
        expect(
          publicResearchRegistry.nodes.some(
            (candidate) => candidate.id === neighbor,
          ),
        ).toBe(true);
      }
    }
  });

  it('preserves selection and filter when switching presentation state externally', () => {
    const state = applyResearchFilter(
      selectResearchNode(createNeutralResearchState(), 'history'),
      'cu-models',
    );
    expect(state).toEqual({
      selectedNodeId: 'history',
      filterId: 'cu-models',
      selectedOutsideFilter: true,
    });
    expect(Object.isFrozen(state)).toBe(true);
  });

  it('keeps the pure state module free of DOM access', () => {
    const source = readFileSync(
      new URL('../research-observatory-state.ts', import.meta.url),
      'utf8',
    );
    expect(source).not.toMatch(/\b(document|window|HTMLElement|Element)\b/);
    expect(publicResearchRegistry.nodes).toHaveLength(11);
  });
});
