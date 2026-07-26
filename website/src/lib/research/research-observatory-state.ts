import {
  publicResearchRegistry,
  type PublicResearchRegistry,
  type ResearchKeyboardKey,
  type ResearchRelationship,
} from './research-observatory';

export interface ResearchObservatoryState {
  readonly selectedNodeId: string;
  readonly filterId: string;
  readonly selectedOutsideFilter: boolean;
}

const getNodeIds = (registry: PublicResearchRegistry): ReadonlySet<string> =>
  new Set(registry.nodes.map((node) => node.id));

const normalizeFilterId = (
  filterId: string,
  registry: PublicResearchRegistry,
): string =>
  registry.filterGroups.some((filter) => filter.id === filterId)
    ? filterId
    : 'all';

export const createNeutralResearchState = (
  registry: PublicResearchRegistry = publicResearchRegistry,
): ResearchObservatoryState =>
  Object.freeze({
    selectedNodeId: registry.overviewNodeId,
    filterId: 'all',
    selectedOutsideFilter: false,
  });

export const getMatchingResearchNodeIds = (
  filterId: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): readonly string[] => {
  const normalizedFilterId = normalizeFilterId(filterId, registry);
  const filter = registry.filterGroups.find(
    (candidate) => candidate.id === normalizedFilterId,
  );
  if (!filter) return Object.freeze([]);
  return Object.freeze(
    registry.nodes
      .filter((node) => node.role !== 'overview')
      .filter(
        (node) =>
          normalizedFilterId === 'all'
          || node.classifications.some((classification) =>
            filter.classifications.includes(classification)),
      )
      .map((node) => node.id),
  );
};

export const countMatchingResearchNodes = (
  filterId: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): number => getMatchingResearchNodeIds(filterId, registry).length;

const selectedOutsideFilter = (
  selectedNodeId: string,
  filterId: string,
  registry: PublicResearchRegistry,
): boolean =>
  selectedNodeId !== registry.overviewNodeId
  && !getMatchingResearchNodeIds(filterId, registry).includes(selectedNodeId);

export const restoreCentralObservatoryState = (
  state: ResearchObservatoryState,
  registry: PublicResearchRegistry = publicResearchRegistry,
): ResearchObservatoryState =>
  Object.freeze({
    selectedNodeId: registry.overviewNodeId,
    filterId: normalizeFilterId(state.filterId, registry),
    selectedOutsideFilter: false,
  });

export const selectResearchNode = (
  state: ResearchObservatoryState,
  nodeId: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): ResearchObservatoryState => {
  if (!getNodeIds(registry).has(nodeId)) {
    return restoreCentralObservatoryState(state, registry);
  }
  const filterId = normalizeFilterId(state.filterId, registry);
  return Object.freeze({
    selectedNodeId: nodeId,
    filterId,
    selectedOutsideFilter: selectedOutsideFilter(
      nodeId,
      filterId,
      registry,
    ),
  });
};

export const applyResearchFilter = (
  state: ResearchObservatoryState,
  filterId: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): ResearchObservatoryState => {
  const normalizedFilterId = normalizeFilterId(filterId, registry);
  const selectedNodeId = getNodeIds(registry).has(state.selectedNodeId)
    ? state.selectedNodeId
    : registry.overviewNodeId;
  return Object.freeze({
    selectedNodeId,
    filterId: normalizedFilterId,
    selectedOutsideFilter: selectedOutsideFilter(
      selectedNodeId,
      normalizedFilterId,
      registry,
    ),
  });
};

export const parseResearchFragment = (
  value: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): string | null => {
  const hashIndex = value.lastIndexOf('#');
  const encodedFragment = (
    hashIndex >= 0 ? value.slice(hashIndex + 1) : value
  ).trim();
  if (encodedFragment.length === 0) return null;
  let fragment: string;
  try {
    fragment = decodeURIComponent(encodedFragment);
  } catch {
    return null;
  }
  return registry.nodes.some(
    (node) => node.role !== 'overview' && node.id === fragment,
  )
    ? fragment
    : null;
};

export const createResearchStateFromFragment = (
  value: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): ResearchObservatoryState => {
  const fragment = parseResearchFragment(value, registry);
  const neutral = createNeutralResearchState(registry);
  return fragment === null
    ? neutral
    : selectResearchNode(neutral, fragment, registry);
};

export const getResearchRelationships = (
  nodeId: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): readonly ResearchRelationship[] =>
  Object.freeze(
    registry.relationships.filter(
      (relationship) =>
        relationship.sourceId === nodeId
        || relationship.targetId === nodeId,
    ),
  );

export const getRelatedResearchNodeIds = (
  nodeId: string,
  registry: PublicResearchRegistry = publicResearchRegistry,
): readonly string[] =>
  Object.freeze(
    getResearchRelationships(nodeId, registry).map((relationship) =>
      relationship.sourceId === nodeId
        ? relationship.targetId
        : relationship.sourceId),
  );

export const getResearchKeyboardNeighbor = (
  nodeId: string,
  key: ResearchKeyboardKey,
  registry: PublicResearchRegistry = publicResearchRegistry,
): string =>
  registry.keyboardNavigation[nodeId]?.[key] ?? nodeId;

export const moveResearchKeyboardSelection = (
  state: ResearchObservatoryState,
  key: ResearchKeyboardKey | 'Home',
  registry: PublicResearchRegistry = publicResearchRegistry,
): ResearchObservatoryState => {
  if (key === 'Home') {
    return restoreCentralObservatoryState(state, registry);
  }
  if (!getNodeIds(registry).has(state.selectedNodeId)) {
    return restoreCentralObservatoryState(state, registry);
  }
  return selectResearchNode(
    state,
    getResearchKeyboardNeighbor(state.selectedNodeId, key, registry),
    registry,
  );
};
