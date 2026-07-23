export type TomPhase = 'expansion' | 'compression';
export type TomBoundaryRole =
  | 'new-cosmic-seed'
  | 'expansion-pause'
  | 'reset-pause'
  | null;
export type GuardedTransitionType = 'phase-continuation' | 'next-breath-reset';

export interface RuntimeTomState {
  readonly id: string;
  readonly label: string;
  readonly phase: TomPhase;
  readonly cycleIndex: number;
  readonly phaseIndex: number;
  readonly previousId: string | null;
  readonly nextId: string | null;
  readonly boundaryRole: TomBoundaryRole;
}

export interface RuntimeGuardedTransition {
  readonly id: string;
  readonly fromId: string;
  readonly toId: string;
  readonly toCycleOffset: 0 | 1;
  readonly type: GuardedTransitionType;
  readonly label: string;
  readonly requiresExplicitAction: true;
  readonly selectableTomState: false;
}

export interface StructuralRuntimeLedger {
  readonly states: readonly RuntimeTomState[];
  readonly transitions: readonly RuntimeGuardedTransition[];
}

export const STRUCTURAL_RUNTIME_STATE_KEYS = Object.freeze([
  'id',
  'label',
  'phase',
  'cycleIndex',
  'phaseIndex',
  'previousId',
  'nextId',
  'boundaryRole',
] as const satisfies readonly (keyof RuntimeTomState)[]);

export const STRUCTURAL_RUNTIME_TRANSITION_KEYS = Object.freeze([
  'id',
  'fromId',
  'toId',
  'toCycleOffset',
  'type',
  'label',
  'requiresExplicitAction',
  'selectableTomState',
] as const satisfies readonly (keyof RuntimeGuardedTransition)[]);

const STATE_KEY_SET = new Set<string>(STRUCTURAL_RUNTIME_STATE_KEYS);
const TRANSITION_KEY_SET = new Set<string>(STRUCTURAL_RUNTIME_TRANSITION_KEYS);

const fail = (message: string): never => {
  throw new Error('Invalid Cosmic Breath structural runtime payload: ' + message);
};
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const requireRecord = (value: unknown, name: string): Record<string, unknown> =>
  isRecord(value) ? value : fail(name + ' must be an object');
const requireString = (value: unknown, name: string): string =>
  typeof value === 'string' && value.length > 0
    ? value
    : fail(name + ' must be a non-empty string');
const requireInteger = (value: unknown, name: string): number =>
  Number.isInteger(value) ? value as number : fail(name + ' must be an integer');
const requireNullableString = (value: unknown, name: string): string | null =>
  value === null ? null : requireString(value, name);
const rejectUnexpectedKeys = (
  record: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  name: string,
): void => {
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) fail(name + ' contains non-runtime key "' + key + '"');
  }
};

const parseRuntimeState = (value: unknown, index: number): RuntimeTomState => {
  const name = 'states[' + index + ']';
  const record = requireRecord(value, name);
  rejectUnexpectedKeys(record, STATE_KEY_SET, name);
  const phase = requireString(record.phase, name + '.phase');
  if (phase !== 'expansion' && phase !== 'compression') {
    fail(name + '.phase must be expansion or compression');
  }
  const boundaryRole = record.boundaryRole;
  if (
    boundaryRole !== null
    && boundaryRole !== 'new-cosmic-seed'
    && boundaryRole !== 'expansion-pause'
    && boundaryRole !== 'reset-pause'
  ) {
    fail(name + '.boundaryRole is not approved');
  }
  return Object.freeze({
    id: requireString(record.id, name + '.id'),
    label: requireString(record.label, name + '.label'),
    phase,
    cycleIndex: requireInteger(record.cycleIndex, name + '.cycleIndex'),
    phaseIndex: requireInteger(record.phaseIndex, name + '.phaseIndex'),
    previousId: requireNullableString(record.previousId, name + '.previousId'),
    nextId: requireNullableString(record.nextId, name + '.nextId'),
    boundaryRole,
  });
};

const parseRuntimeTransition = (
  value: unknown,
  index: number,
): RuntimeGuardedTransition => {
  const name = 'transitions[' + index + ']';
  const record = requireRecord(value, name);
  rejectUnexpectedKeys(record, TRANSITION_KEY_SET, name);
  const type = requireString(record.type, name + '.type');
  if (type !== 'phase-continuation' && type !== 'next-breath-reset') {
    fail(name + '.type is not approved');
  }
  if (record.toCycleOffset !== 0 && record.toCycleOffset !== 1) {
    fail(name + '.toCycleOffset must be 0 or 1');
  }
  if (record.requiresExplicitAction !== true) {
    fail(name + '.requiresExplicitAction must be true');
  }
  if (record.selectableTomState !== false) {
    fail(name + '.selectableTomState must be false');
  }
  return Object.freeze({
    id: requireString(record.id, name + '.id'),
    fromId: requireString(record.fromId, name + '.fromId'),
    toId: requireString(record.toId, name + '.toId'),
    toCycleOffset: record.toCycleOffset,
    type,
    label: requireString(record.label, name + '.label'),
    requiresExplicitAction: true,
    selectableTomState: false,
  });
};

const validateRuntimeLedger = (ledger: StructuralRuntimeLedger): void => {
  const { states, transitions } = ledger;
  if (states.length !== 51) fail('states must contain exactly 51 records');
  if (new Set(states.map((state) => state.id)).size !== 51) {
    fail('state IDs must be unique');
  }
  if (!states.every((state, index) => state.cycleIndex === index + 1)) {
    fail('cycleIndex values must be consecutive from 1 through 51');
  }

  const expansion = states.filter((state) => state.phase === 'expansion');
  const compression = states.filter((state) => state.phase === 'compression');
  if (expansion.length !== 26 || compression.length !== 25) {
    fail('states must contain exactly 26 expansion and 25 compression records');
  }
  if (!expansion.every((state, index) => state.phaseIndex === index + 1)) {
    fail('expansion phaseIndex values must be consecutive from 1 through 26');
  }
  if (!compression.every((state, index) => state.phaseIndex === index + 1)) {
    fail('compression phaseIndex values must be consecutive from 1 through 25');
  }

  const byId = new Map(states.map((state) => [state.id, state]));
  for (const state of states) {
    if (state.previousId !== null) {
      const previous = byId.get(state.previousId);
      if (!previous || previous.nextId !== state.id || previous.phase !== state.phase) {
        fail('invalid previousId reference for ' + state.id);
      }
    }
    if (state.nextId !== null) {
      const next = byId.get(state.nextId);
      if (!next || next.previousId !== state.id || next.phase !== state.phase) {
        fail('invalid nextId reference for ' + state.id);
      }
    }
  }

  const subZtom = byId.get('expansion-sub-ztom');
  const atom = byId.get('expansion-atom');
  const btom = byId.get('compression-btom');
  const ztom = byId.get('compression-ztom');
  if (
    subZtom?.boundaryRole !== 'new-cosmic-seed'
    || subZtom.previousId !== null
    || subZtom.cycleIndex !== 1
  ) fail('expansion-sub-ztom boundary is invalid');
  if (
    atom?.boundaryRole !== 'expansion-pause'
    || atom.nextId !== null
    || atom.cycleIndex !== 26
  ) fail('expansion-atom boundary is invalid');
  if (btom?.previousId !== null || btom.cycleIndex !== 27) {
    fail('compression-btom boundary is invalid');
  }
  if (
    ztom?.boundaryRole !== 'reset-pause'
    || ztom.nextId !== null
    || ztom.cycleIndex !== 51
  ) fail('compression-ztom boundary is invalid');

  if (transitions.length !== 2) fail('exactly two guarded transitions are required');
  if (new Set(transitions.map((transition) => transition.id)).size !== 2) {
    fail('transition IDs must be unique');
  }
  for (const transition of transitions) {
    if (!byId.has(transition.fromId) || !byId.has(transition.toId)) {
      fail('unresolved transition reference for ' + transition.id);
    }
  }
  const atomTransition = transitions.find(
    (transition) => transition.id === 'transition-atom-to-btom',
  );
  if (
    atomTransition?.label !== 'Begin Compression'
    || atomTransition.fromId !== 'expansion-atom'
    || atomTransition.toId !== 'compression-btom'
    || atomTransition.toCycleOffset !== 0
    || atomTransition.type !== 'phase-continuation'
  ) fail('atom-to-btom transition is invalid');
  const resetTransition = transitions.find(
    (transition) => transition.id === 'transition-ztom-to-next-sub-ztom',
  );
  if (
    resetTransition?.label !== 'Begin the Next Cosmic Breath'
    || resetTransition.fromId !== 'compression-ztom'
    || resetTransition.toId !== 'expansion-sub-ztom'
    || resetTransition.toCycleOffset !== 1
    || resetTransition.type !== 'next-breath-reset'
  ) fail('ztom-to-next-sub-ztom transition is invalid');
};

export const parseStructuralRuntimeLedger = (value: unknown): StructuralRuntimeLedger => {
  const record = requireRecord(value, 'payload');
  rejectUnexpectedKeys(record, new Set(['states', 'transitions']), 'payload');
  if (!Array.isArray(record.states)) fail('states must be an array');
  if (!Array.isArray(record.transitions)) fail('transitions must be an array');
  const ledger = Object.freeze({
    states: Object.freeze(record.states.map(parseRuntimeState)),
    transitions: Object.freeze(record.transitions.map(parseRuntimeTransition)),
  });
  validateRuntimeLedger(ledger);
  return ledger;
};

export const parseStructuralRuntimePayload = (serialized: string): StructuralRuntimeLedger => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return fail('payload is not valid JSON: ' + message);
  }
  return parseStructuralRuntimeLedger(parsed);
};

export const serializeStructuralRuntimePayload = (value: unknown): string => {
  const ledger = parseStructuralRuntimeLedger(value);
  return JSON.stringify(ledger).replace(
    /[<>&\u2028\u2029]/g,
    (character) => ({
      '<': '\\u003c',
      '>': '\\u003e',
      '&': '\\u0026',
      '\u2028': '\\u2028',
      '\u2029': '\\u2029',
    })[character] ?? character,
  );
};

const createRuntimeLookups = (ledger: StructuralRuntimeLedger) => {
  const byId = new Map(ledger.states.map((state) => [state.id, state]));
  const indexById = new Map(ledger.states.map((state, index) => [state.id, index]));
  const getStateById = (id: string): RuntimeTomState => {
    const state = byId.get(id);
    if (!state) throw new RangeError('Unknown structural TOM state ID: ' + id);
    return state;
  };
  const getStateByIndex = (index: number): RuntimeTomState => {
    if (!Number.isInteger(index) || index < 0 || index >= ledger.states.length) {
      throw new RangeError('Structural TOM index must be an integer from 0 to 50: ' + index);
    }
    return ledger.states[index];
  };
  const getStateIndex = (id: string): number => {
    const index = indexById.get(id);
    if (index === undefined) throw new RangeError('Unknown structural TOM state ID: ' + id);
    return index;
  };
  return { getStateById, getStateByIndex, getStateIndex };
};

export type CycleMode =
  | 'overview'
  | 'exploring'
  | 'dragging'
  | 'settling'
  | 'details'
  | 'boundary-pause';

export interface CosmicBreathCycleState {
  readonly mode: CycleMode;
  readonly selectedIndex: number | null;
  readonly previewIndex: number | null;
  readonly pendingSettleToken: number | null;
  readonly settledSelectionToken: number | null;
  readonly detailsIndex: number | null;
  readonly boundaryRole: TomBoundaryRole;
  readonly cycleCount: number;
  readonly interactionToken: number;
}

const BASE_INITIAL_CYCLE_STATE: CosmicBreathCycleState = Object.freeze({
  mode: 'overview',
  selectedIndex: null,
  previewIndex: null,
  pendingSettleToken: null,
  settledSelectionToken: null,
  detailsIndex: null,
  boundaryRole: null,
  cycleCount: 0,
  interactionToken: 0,
});

export const createCycleStateEngine = (ledger: StructuralRuntimeLedger) => {
  const { getStateByIndex, getStateIndex } = createRuntimeLookups(ledger);
  const atomTransition = ledger.transitions.find(
    (transition) => transition.id === 'transition-atom-to-btom',
  )!;
  const resetTransition = ledger.transitions.find(
    (transition) => transition.id === 'transition-ztom-to-next-sub-ztom',
  )!;
  const freezeState = (state: CosmicBreathCycleState): CosmicBreathCycleState =>
    Object.freeze(state);
  const selectedIndexOrThrow = (state: CosmicBreathCycleState): number => {
    if (state.selectedIndex === null) {
      throw new Error('Cosmic Breath exploration has no selected simulated state');
    }
    return state.selectedIndex;
  };
  const modeForIndex = (index: number): CycleMode => {
    const role = getStateByIndex(index).boundaryRole;
    return role === 'expansion-pause' || role === 'reset-pause'
      ? 'boundary-pause'
      : 'exploring';
  };
  const selectIndex = (
    state: CosmicBreathCycleState,
    index: number,
    cycleCount = state.cycleCount,
  ): CosmicBreathCycleState => {
    const selected = getStateByIndex(index);
    return freezeState({
      mode: modeForIndex(index),
      selectedIndex: index,
      previewIndex: null,
      pendingSettleToken: null,
      settledSelectionToken: null,
      detailsIndex: null,
      boundaryRole: selected.boundaryRole,
      cycleCount,
      interactionToken: state.interactionToken + 1,
    });
  };
  const enterExploration = (
    state: CosmicBreathCycleState,
    initialStateId = 'expansion-sub-ztom',
  ): CosmicBreathCycleState => selectIndex(state, getStateIndex(initialStateId));
  const selectStateById = (
    state: CosmicBreathCycleState,
    id: string,
  ): CosmicBreathCycleState => selectIndex(state, getStateIndex(id));
  const selectStateByIndex = (
    state: CosmicBreathCycleState,
    index: number,
  ): CosmicBreathCycleState => {
    getStateByIndex(index);
    return selectIndex(state, index);
  };
  const moveOrdinarily = (
    state: CosmicBreathCycleState,
    direction: 'previous' | 'next',
  ): CosmicBreathCycleState => {
    const currentIndex = selectedIndexOrThrow(state);
    const current = getStateByIndex(currentIndex);
    const adjacentId = direction === 'previous' ? current.previousId : current.nextId;
    return adjacentId === null
      ? selectIndex(state, currentIndex)
      : selectIndex(state, getStateIndex(adjacentId));
  };
  const moveToPreviousState = (state: CosmicBreathCycleState): CosmicBreathCycleState =>
    moveOrdinarily(state, 'previous');
  const moveToNextState = (state: CosmicBreathCycleState): CosmicBreathCycleState =>
    moveOrdinarily(state, 'next');
  const continueFromAtom = (state: CosmicBreathCycleState): CosmicBreathCycleState => {
    const current = getStateByIndex(selectedIndexOrThrow(state));
    if (current.id !== atomTransition.fromId) {
      throw new Error('Begin Compression is available only while atom is selected');
    }
    return selectIndex(state, getStateIndex(atomTransition.toId));
  };
  const beginNextCosmicBreath = (state: CosmicBreathCycleState): CosmicBreathCycleState => {
    const current = getStateByIndex(selectedIndexOrThrow(state));
    if (current.id !== resetTransition.fromId) {
      throw new Error('Begin the Next Cosmic Breath is available only while ztom is selected');
    }
    return selectIndex(
      state,
      getStateIndex(resetTransition.toId),
      state.cycleCount + resetTransition.toCycleOffset,
    );
  };
  const beginDragging = (state: CosmicBreathCycleState): CosmicBreathCycleState => {
    const selectedIndex = selectedIndexOrThrow(state);
    return freezeState({
      ...state,
      mode: 'dragging',
      previewIndex: selectedIndex,
      pendingSettleToken: null,
      settledSelectionToken: null,
      detailsIndex: null,
      boundaryRole: null,
      interactionToken: state.interactionToken + 1,
    });
  };
  const updateDragPreview = (
    state: CosmicBreathCycleState,
    requestedIndex: number,
  ): CosmicBreathCycleState => {
    if (state.mode !== 'dragging') throw new Error('Drag preview requires dragging mode');
    getStateByIndex(requestedIndex);
    return freezeState({ ...state, previewIndex: requestedIndex });
  };
  const releaseDragIntoSettling = (
    state: CosmicBreathCycleState,
    requestedIndex = state.previewIndex,
  ): CosmicBreathCycleState => {
    if (state.mode !== 'dragging') throw new Error('Drag release requires dragging mode');
    if (requestedIndex === null) {
      throw new Error('Drag release requires a structural preview index');
    }
    getStateByIndex(requestedIndex);
    const token = state.interactionToken + 1;
    return freezeState({
      ...state,
      mode: 'settling',
      selectedIndex: requestedIndex,
      previewIndex: null,
      pendingSettleToken: token,
      settledSelectionToken: null,
      detailsIndex: null,
      boundaryRole: getStateByIndex(requestedIndex).boundaryRole,
      interactionToken: token,
    });
  };
  const completeSettling = (
    state: CosmicBreathCycleState,
    token: number,
  ): CosmicBreathCycleState => {
    if (state.mode !== 'settling' || state.pendingSettleToken !== token) return state;
    const selectedIndex = selectedIndexOrThrow(state);
    return freezeState({
      ...state,
      mode: modeForIndex(selectedIndex),
      pendingSettleToken: null,
      settledSelectionToken: token,
      boundaryRole: getStateByIndex(selectedIndex).boundaryRole,
    });
  };
  const revealSettledDetails = (
    state: CosmicBreathCycleState,
    token: number,
  ): CosmicBreathCycleState => {
    if (state.settledSelectionToken !== token || state.selectedIndex === null) return state;
    return freezeState({ ...state, mode: 'details', detailsIndex: state.selectedIndex });
  };
  const returnToOverview = (_state: CosmicBreathCycleState): CosmicBreathCycleState =>
    BASE_INITIAL_CYCLE_STATE;

  return Object.freeze({
    INITIAL_CYCLE_STATE: BASE_INITIAL_CYCLE_STATE,
    enterExploration,
    selectStateById,
    selectStateByIndex,
    moveToPreviousState,
    moveToNextState,
    continueFromAtom,
    beginNextCosmicBreath,
    beginDragging,
    updateDragPreview,
    releaseDragIntoSettling,
    completeSettling,
    revealSettledDetails,
    returnToOverview,
    resetCycleExplorer: returnToOverview,
  });
};

export interface NormalizedPoint {
  readonly x: number;
  readonly y: number;
}

export interface StructuralDragAnchor {
  readonly phase: TomPhase;
  readonly phaseIndex: number;
  readonly canonicalIndex: number;
  readonly stateId: string;
  readonly label: string;
  readonly point: NormalizedPoint;
}

const FIELD_CENTER = Object.freeze({ x: 0.5, y: 0.5 });
const TIE_EPSILON = 1e-12;
const requirePhase = (phase: TomPhase): TomPhase => {
  if (phase !== 'expansion' && phase !== 'compression') {
    throw new RangeError('Drag phase must be expansion or compression: ' + phase);
  }
  return phase;
};
const requireFinitePoint = (point: NormalizedPoint): NormalizedPoint => {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
    throw new RangeError('Normalized drag coordinates must be finite numbers');
  }
  return point;
};
const requireNormalizedPoint = (point: NormalizedPoint): NormalizedPoint => {
  requireFinitePoint(point);
  if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) {
    throw new RangeError('Normalized drag coordinates must be between 0 and 1');
  }
  return point;
};

export const clampNormalizedPoint = (point: NormalizedPoint): NormalizedPoint => {
  requireFinitePoint(point);
  return Object.freeze({
    x: Math.min(1, Math.max(0, point.x)),
    y: Math.min(1, Math.max(0, point.y)),
  });
};

const structuralPoint = (
  phase: TomPhase,
  phaseOffset: number,
  count: number,
): NormalizedPoint => {
  const progress = count === 1 ? 0 : phaseOffset / (count - 1);
  const isExpansion = phase === 'expansion';
  const radius = isExpansion ? 0.055 + (0.385 * progress) : 0.44 - (0.36 * progress);
  const angle = isExpansion
    ? (-Math.PI / 2) + (progress * Math.PI * 1.75)
    : (Math.PI * 0.12) + (progress * Math.PI * 1.65);
  return Object.freeze({
    x: FIELD_CENTER.x + (Math.cos(angle) * radius),
    y: FIELD_CENTER.y + (Math.sin(angle) * radius),
  });
};

export const findNearestAnchor = (
  point: NormalizedPoint,
  anchors: readonly StructuralDragAnchor[],
): StructuralDragAnchor => {
  requireNormalizedPoint(point);
  if (anchors.length === 0) throw new RangeError('Nearest-anchor selection requires anchors');
  let nearest = anchors[0];
  requireNormalizedPoint(nearest.point);
  let nearestDistance = ((point.x - nearest.point.x) ** 2) + ((point.y - nearest.point.y) ** 2);
  for (const anchor of anchors.slice(1)) {
    requireNormalizedPoint(anchor.point);
    const distance = ((point.x - anchor.point.x) ** 2) + ((point.y - anchor.point.y) ** 2);
    if (
      distance < nearestDistance - TIE_EPSILON
      || (
        Math.abs(distance - nearestDistance) <= TIE_EPSILON
        && anchor.canonicalIndex < nearest.canonicalIndex
      )
    ) {
      nearest = anchor;
      nearestDistance = distance;
    }
  }
  return nearest;
};

export const createCycleDragEngine = (ledger: StructuralRuntimeLedger) => {
  const { getStateByIndex } = createRuntimeLookups(ledger);
  const expansionStates = ledger.states.filter((state) => state.phase === 'expansion');
  const compressionStates = ledger.states.filter((state) => state.phase === 'compression');
  const generateStructuralAnchors = (phase: TomPhase): readonly StructuralDragAnchor[] => {
    requirePhase(phase);
    const states = phase === 'expansion' ? expansionStates : compressionStates;
    const canonicalOffset = phase === 'expansion' ? 0 : expansionStates.length;
    return Object.freeze(states.map((state, phaseOffset) => Object.freeze({
      phase,
      phaseIndex: state.phaseIndex,
      canonicalIndex: canonicalOffset + phaseOffset,
      stateId: state.id,
      label: state.label,
      point: structuralPoint(phase, phaseOffset, states.length),
    })));
  };
  const expansionDragAnchors = generateStructuralAnchors('expansion');
  const compressionDragAnchors = generateStructuralAnchors('compression');
  const selectableDragAnchors = Object.freeze([
    ...expansionDragAnchors,
    ...compressionDragAnchors,
  ]);
  const findNearestAnchorInPhase = (
    point: NormalizedPoint,
    phase: TomPhase,
  ): StructuralDragAnchor => findNearestAnchor(
    point,
    requirePhase(phase) === 'expansion' ? expansionDragAnchors : compressionDragAnchors,
  );
  const findNearestSelectableAnchor = (point: NormalizedPoint): StructuralDragAnchor =>
    findNearestAnchor(point, selectableDragAnchors);
  const canonicalIndexForAnchor = (anchor: StructuralDragAnchor): number => {
    const state = getStateByIndex(anchor.canonicalIndex);
    if (state.id !== anchor.stateId || state.phase !== anchor.phase) {
      throw new RangeError('Drag anchor does not match its canonical structural state');
    }
    return anchor.canonicalIndex;
  };
  const phaseBoundedPreviewIndex = (
    requestedCanonicalIndex: number,
    phase: TomPhase,
  ): number => {
    getStateByIndex(requestedCanonicalIndex);
    return requirePhase(phase) === 'expansion'
      ? Math.min(requestedCanonicalIndex, expansionDragAnchors.length - 1)
      : Math.max(requestedCanonicalIndex, expansionDragAnchors.length);
  };
  const getDragAnchorByCanonicalIndex = (canonicalIndex: number): StructuralDragAnchor => {
    const state = getStateByIndex(canonicalIndex);
    const anchors = state.phase === 'expansion'
      ? expansionDragAnchors
      : compressionDragAnchors;
    const phaseOffset = state.phase === 'expansion'
      ? canonicalIndex
      : canonicalIndex - expansionDragAnchors.length;
    const anchor = anchors[phaseOffset];
    if (!anchor || anchor.stateId !== state.id) {
      throw new RangeError('No drag anchor for canonical structural index: ' + canonicalIndex);
    }
    return anchor;
  };
  const snappedMarkerCoordinates = (canonicalIndex: number): NormalizedPoint =>
    getDragAnchorByCanonicalIndex(canonicalIndex).point;
  const isTerminalBoundaryAnchor = (anchor: StructuralDragAnchor): boolean =>
    anchor.stateId === 'expansion-atom' || anchor.stateId === 'compression-ztom';
  const compactStructuralPreviewText = (anchor: StructuralDragAnchor): string => {
    canonicalIndexForAnchor(anchor);
    const phase = anchor.phase === 'expansion' ? 'Expansion' : 'Compression';
    return `${anchor.label} · ${phase} · cycle position ${anchor.canonicalIndex + 1} of ${ledger.states.length}`;
  };
  return Object.freeze({
    generateStructuralAnchors,
    expansionDragAnchors,
    compressionDragAnchors,
    selectableDragAnchors,
    findNearestAnchorInPhase,
    findNearestSelectableAnchor,
    canonicalIndexForAnchor,
    phaseBoundedPreviewIndex,
    getDragAnchorByCanonicalIndex,
    snappedMarkerCoordinates,
    isTerminalBoundaryAnchor,
    compactStructuralPreviewText,
  });
};

export type ExplorerKeyboardCommand =
  | 'previous'
  | 'next'
  | 'first'
  | 'last'
  | 'phase-start'
  | 'phase-end'
  | 'return-overview';
export type SelectionStatusKind = 'sequential' | 'direct' | 'settling' | 'settled';
export interface EditableTargetLike {
  readonly tagName?: string;
  readonly isContentEditable?: boolean;
}
export interface ExplorerControlState {
  readonly previousDisabled: boolean;
  readonly nextDisabled: boolean;
  readonly showBeginCompression: boolean;
  readonly showBeginNextBreath: boolean;
}
export interface StructuralStateSummary {
  readonly label: string;
  readonly phase: 'Expansion' | 'Compression';
  readonly cyclePosition: string;
  readonly phasePosition: string;
  readonly previousState: string;
  readonly nextState: string;
  readonly boundaryRole: string;
}

const keyboardCommands: Readonly<Record<string, ExplorerKeyboardCommand>> = Object.freeze({
  ArrowLeft: 'previous',
  ArrowRight: 'next',
  Home: 'first',
  End: 'last',
  PageUp: 'phase-start',
  PageDown: 'phase-end',
  Escape: 'return-overview',
});

export const getExplorerKeyboardCommand = (key: string): ExplorerKeyboardCommand | null =>
  keyboardCommands[key] ?? null;
export const isEditableControl = (target: EditableTargetLike | null): boolean => {
  if (!target) return false;
  const tagName = target.tagName?.toUpperCase();
  return target.isContentEditable === true
    || tagName === 'INPUT'
    || tagName === 'TEXTAREA'
    || tagName === 'SELECT';
};
export const getExplorerControlState = (
  state: RuntimeTomState,
): ExplorerControlState => Object.freeze({
  previousDisabled: state.previousId === null,
  nextDisabled: state.nextId === null,
  showBeginCompression: state.boundaryRole === 'expansion-pause',
  showBeginNextBreath: state.boundaryRole === 'reset-pause',
});
export const getSelectionStatusText = (
  kind: SelectionStatusKind,
  state: RuntimeTomState,
): string => {
  const position = `${state.label}, ${state.phase} state ${state.phaseIndex}`;
  if (kind === 'direct') return `Direct educational jump to ${position}. Settling.`;
  if (kind === 'settling') return `${position}. Settling.`;
  if (kind === 'settled') return `${position}. Structural details are ready.`;
  return `Moved to ${position}. Settling.`;
};

const boundaryRoleLabels = Object.freeze({
  'new-cosmic-seed': 'New cosmic seed',
  'expansion-pause': 'Expansion pause',
  'reset-pause': 'Reset pause',
});

export const createCycleControlEngine = (ledger: StructuralRuntimeLedger) => {
  const { getStateById } = createRuntimeLookups(ledger);
  const parseStructuralIndex = (value: string | number): number => {
    if (typeof value === 'string' && value.trim() === '') {
      throw new RangeError('Structural TOM index must be an integer from 0 to 50');
    }
    const index = typeof value === 'number' ? value : Number(value);
    if (!Number.isInteger(index) || index < 0 || index >= ledger.states.length) {
      throw new RangeError('Structural TOM index must be an integer from 0 to 50: ' + value);
    }
    return index;
  };
  const clampStructuralIndex = (value: number): number => {
    if (!Number.isFinite(value)) {
      throw new RangeError('Structural TOM index must be a finite number');
    }
    return Math.min(ledger.states.length - 1, Math.max(0, Math.round(value)));
  };
  const getPhaseBoundaryIndices = (
    phase: TomPhase,
  ): Readonly<{ start: number; end: number }> => {
    const expansionCount = ledger.states.filter((state) => state.phase === 'expansion').length;
    return phase === 'expansion'
      ? Object.freeze({ start: 0, end: expansionCount - 1 })
      : Object.freeze({ start: expansionCount, end: ledger.states.length - 1 });
  };
  const labelForId = (id: string | null, emptyLabel: string): string =>
    id === null ? emptyLabel : getStateById(id).label;
  const getStructuralStateSummary = (state: RuntimeTomState): StructuralStateSummary =>
    Object.freeze({
      label: state.label,
      phase: state.phase === 'expansion' ? 'Expansion' : 'Compression',
      cyclePosition: `${state.cycleIndex} of ${ledger.states.length}`,
      phasePosition: `${state.phaseIndex} of ${state.phase === 'expansion' ? 26 : 25}`,
      previousState: labelForId(state.previousId, 'No ordinary previous state'),
      nextState: labelForId(state.nextId, 'No ordinary next state'),
      boundaryRole: state.boundaryRole === null
        ? 'None'
        : boundaryRoleLabels[state.boundaryRole],
    });
  return Object.freeze({
    parseStructuralIndex,
    clampStructuralIndex,
    getPhaseBoundaryIndices,
    getStructuralStateSummary,
  });
};
