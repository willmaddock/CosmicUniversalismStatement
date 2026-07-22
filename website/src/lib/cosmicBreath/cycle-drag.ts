import {
  compressionTomStates,
  expansionTomStates,
  getTomStateByIndex,
  orderedTomStates,
  type StructuralTomState,
  type TomPhase,
} from './cycle-ledger';

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

const structuralPoint = (phase: TomPhase, phaseOffset: number, count: number): NormalizedPoint => {
  const progress = count === 1 ? 0 : phaseOffset / (count - 1);
  const isExpansion = phase === 'expansion';
  const radius = isExpansion
    ? 0.055 + (0.385 * progress)
    : 0.44 - (0.36 * progress);
  const angle = isExpansion
    ? (-Math.PI / 2) + (progress * Math.PI * 1.75)
    : (Math.PI * 0.12) + (progress * Math.PI * 1.65);
  return Object.freeze({
    x: FIELD_CENTER.x + (Math.cos(angle) * radius),
    y: FIELD_CENTER.y + (Math.sin(angle) * radius),
  });
};

const statesForPhase = (phase: TomPhase): readonly StructuralTomState[] =>
  phase === 'expansion' ? expansionTomStates : compressionTomStates;

export const generateStructuralAnchors = (
  phase: TomPhase,
): readonly StructuralDragAnchor[] => {
  requirePhase(phase);
  const states = statesForPhase(phase);
  const canonicalOffset = phase === 'expansion' ? 0 : expansionTomStates.length;
  return Object.freeze(states.map((state, phaseOffset) => Object.freeze({
    phase,
    phaseIndex: state.phaseIndex,
    canonicalIndex: canonicalOffset + phaseOffset,
    stateId: state.id,
    label: state.label,
    point: structuralPoint(phase, phaseOffset, states.length),
  })));
};

export const expansionDragAnchors = generateStructuralAnchors('expansion');
export const compressionDragAnchors = generateStructuralAnchors('compression');

const distanceSquared = (left: NormalizedPoint, right: NormalizedPoint): number => {
  const x = left.x - right.x;
  const y = left.y - right.y;
  return (x * x) + (y * y);
};

export const findNearestAnchor = (
  point: NormalizedPoint,
  anchors: readonly StructuralDragAnchor[],
): StructuralDragAnchor => {
  requireNormalizedPoint(point);
  if (anchors.length === 0) throw new RangeError('Nearest-anchor selection requires anchors');

  let nearest = anchors[0];
  requireNormalizedPoint(nearest.point);
  let nearestDistance = distanceSquared(point, nearest.point);
  for (const anchor of anchors.slice(1)) {
    requireNormalizedPoint(anchor.point);
    const distance = distanceSquared(point, anchor.point);
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

export const findNearestAnchorInPhase = (
  point: NormalizedPoint,
  phase: TomPhase,
): StructuralDragAnchor => findNearestAnchor(
  point,
  requirePhase(phase) === 'expansion' ? expansionDragAnchors : compressionDragAnchors,
);

export const canonicalIndexForAnchor = (anchor: StructuralDragAnchor): number => {
  const state = getTomStateByIndex(anchor.canonicalIndex);
  if (state.id !== anchor.stateId || state.phase !== anchor.phase) {
    throw new RangeError('Drag anchor does not match its canonical structural state');
  }
  return anchor.canonicalIndex;
};

export const phaseBoundedPreviewIndex = (
  requestedCanonicalIndex: number,
  phase: TomPhase,
): number => {
  getTomStateByIndex(requestedCanonicalIndex);
  return requirePhase(phase) === 'expansion'
    ? Math.min(requestedCanonicalIndex, expansionDragAnchors.length - 1)
    : Math.max(requestedCanonicalIndex, expansionDragAnchors.length);
};

export const getDragAnchorByCanonicalIndex = (
  canonicalIndex: number,
): StructuralDragAnchor => {
  const state = getTomStateByIndex(canonicalIndex);
  const anchors = state.phase === 'expansion' ? expansionDragAnchors : compressionDragAnchors;
  const phaseOffset = state.phase === 'expansion'
    ? canonicalIndex
    : canonicalIndex - expansionDragAnchors.length;
  const anchor = anchors[phaseOffset];
  if (!anchor || anchor.stateId !== state.id) {
    throw new RangeError('No drag anchor for canonical structural index: ' + canonicalIndex);
  }
  return anchor;
};

export const snappedMarkerCoordinates = (canonicalIndex: number): NormalizedPoint =>
  getDragAnchorByCanonicalIndex(canonicalIndex).point;

export const isTerminalBoundaryAnchor = (anchor: StructuralDragAnchor): boolean =>
  anchor.stateId === 'expansion-atom' || anchor.stateId === 'compression-ztom';

export const compactStructuralPreviewText = (anchor: StructuralDragAnchor): string => {
  canonicalIndexForAnchor(anchor);
  const phase = anchor.phase === 'expansion' ? 'Expansion' : 'Compression';
  return `${anchor.label} · ${phase} · cycle position ${anchor.canonicalIndex + 1} of ${orderedTomStates.length}`;
};
