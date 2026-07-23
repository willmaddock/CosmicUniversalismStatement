import { structuralRuntimeLedger } from './cycle-ledger';
import { createCycleStateEngine } from './cycle-runtime';

export type {
  CosmicBreathCycleState,
  CycleMode,
} from './cycle-runtime';

const engine = createCycleStateEngine(structuralRuntimeLedger);

export const {
  INITIAL_CYCLE_STATE,
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
  resetCycleExplorer,
} = engine;
