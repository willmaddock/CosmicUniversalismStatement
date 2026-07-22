import {
  atomToBtomTransition,
  getTomStateByIndex,
  getTomStateIndex,
  ztomToNextSubZtomTransition,
  type TomBoundaryRole,
} from './cycle-ledger';

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

export const INITIAL_CYCLE_STATE: CosmicBreathCycleState = Object.freeze({
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

const freezeState = (state: CosmicBreathCycleState): CosmicBreathCycleState =>
  Object.freeze(state);

const selectedIndexOrThrow = (state: CosmicBreathCycleState): number => {
  if (state.selectedIndex === null) {
    throw new Error('Cosmic Breath exploration has no selected simulated state');
  }
  return state.selectedIndex;
};

const modeForIndex = (index: number): CycleMode => {
  const role = getTomStateByIndex(index).boundaryRole;
  return role === 'expansion-pause' || role === 'reset-pause'
    ? 'boundary-pause'
    : 'exploring';
};

const selectIndex = (
  state: CosmicBreathCycleState,
  index: number,
  cycleCount = state.cycleCount,
): CosmicBreathCycleState => {
  const selected = getTomStateByIndex(index);
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

const clampPreviewToSelectedPhase = (
  state: CosmicBreathCycleState,
  requestedIndex: number,
): number => {
  getTomStateByIndex(requestedIndex);
  const selected = getTomStateByIndex(selectedIndexOrThrow(state));
  return selected.phase === 'expansion'
    ? Math.min(requestedIndex, 25)
    : Math.max(requestedIndex, 26);
};

export const enterExploration = (
  state: CosmicBreathCycleState,
  initialStateId = 'expansion-sub-ztom',
): CosmicBreathCycleState => selectIndex(state, getTomStateIndex(initialStateId));

export const selectStateById = (
  state: CosmicBreathCycleState,
  id: string,
): CosmicBreathCycleState => selectIndex(state, getTomStateIndex(id));

export const selectStateByIndex = (
  state: CosmicBreathCycleState,
  index: number,
): CosmicBreathCycleState => {
  getTomStateByIndex(index);
  return selectIndex(state, index);
};

const moveOrdinarily = (
  state: CosmicBreathCycleState,
  direction: 'previous' | 'next',
): CosmicBreathCycleState => {
  const currentIndex = selectedIndexOrThrow(state);
  const current = getTomStateByIndex(currentIndex);
  const adjacentId = direction === 'previous' ? current.previousId : current.nextId;
  return adjacentId === null
    ? selectIndex(state, currentIndex)
    : selectIndex(state, getTomStateIndex(adjacentId));
};

export const moveToPreviousState = (
  state: CosmicBreathCycleState,
): CosmicBreathCycleState => moveOrdinarily(state, 'previous');

export const moveToNextState = (
  state: CosmicBreathCycleState,
): CosmicBreathCycleState => moveOrdinarily(state, 'next');

export const continueFromAtom = (
  state: CosmicBreathCycleState,
): CosmicBreathCycleState => {
  const current = getTomStateByIndex(selectedIndexOrThrow(state));
  if (current.id !== atomToBtomTransition.fromId) {
    throw new Error('Begin Compression is available only while atom is selected');
  }
  return selectIndex(state, getTomStateIndex(atomToBtomTransition.toId));
};

export const beginNextCosmicBreath = (
  state: CosmicBreathCycleState,
): CosmicBreathCycleState => {
  const current = getTomStateByIndex(selectedIndexOrThrow(state));
  if (current.id !== ztomToNextSubZtomTransition.fromId) {
    throw new Error('Begin the Next Cosmic Breath is available only while ztom is selected');
  }
  return selectIndex(
    state,
    getTomStateIndex(ztomToNextSubZtomTransition.toId),
    state.cycleCount + ztomToNextSubZtomTransition.toCycleOffset,
  );
};

export const beginDragging = (
  state: CosmicBreathCycleState,
): CosmicBreathCycleState => {
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

export const updateDragPreview = (
  state: CosmicBreathCycleState,
  requestedIndex: number,
): CosmicBreathCycleState => {
  if (state.mode !== 'dragging') throw new Error('Drag preview requires dragging mode');
  return freezeState({
    ...state,
    previewIndex: clampPreviewToSelectedPhase(state, requestedIndex),
  });
};

export const releaseDragIntoSettling = (
  state: CosmicBreathCycleState,
  requestedIndex = state.previewIndex,
): CosmicBreathCycleState => {
  if (state.mode !== 'dragging') throw new Error('Drag release requires dragging mode');
  if (requestedIndex === null) throw new Error('Drag release requires a structural preview index');
  const selectedIndex = clampPreviewToSelectedPhase(state, requestedIndex);
  const token = state.interactionToken + 1;
  return freezeState({
    ...state,
    mode: 'settling',
    selectedIndex,
    previewIndex: null,
    pendingSettleToken: token,
    settledSelectionToken: null,
    detailsIndex: null,
    boundaryRole: getTomStateByIndex(selectedIndex).boundaryRole,
    interactionToken: token,
  });
};

export const completeSettling = (
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
    boundaryRole: getTomStateByIndex(selectedIndex).boundaryRole,
  });
};

export const revealSettledDetails = (
  state: CosmicBreathCycleState,
  token: number,
): CosmicBreathCycleState => {
  if (state.settledSelectionToken !== token || state.selectedIndex === null) return state;
  return freezeState({
    ...state,
    mode: 'details',
    detailsIndex: state.selectedIndex,
  });
};

export const returnToOverview = (
  _state: CosmicBreathCycleState,
): CosmicBreathCycleState => INITIAL_CYCLE_STATE;

export const resetCycleExplorer = returnToOverview;
