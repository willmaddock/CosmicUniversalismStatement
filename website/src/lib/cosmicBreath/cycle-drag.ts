import { structuralRuntimeLedger } from './cycle-ledger';
import {
  clampNormalizedPoint,
  createCycleDragEngine,
  findNearestAnchor,
} from './cycle-runtime';

export type {
  NormalizedPoint,
  StructuralDragAnchor,
} from './cycle-runtime';
export {
  clampNormalizedPoint,
  findNearestAnchor,
};

const engine = createCycleDragEngine(structuralRuntimeLedger);

export const {
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
} = engine;
