import { describe, expect, it } from 'vitest';
import {
  canonicalIndexForAnchor,
  clampNormalizedPoint,
  compactStructuralPreviewText,
  compressionDragAnchors,
  expansionDragAnchors,
  findNearestAnchor,
  findNearestAnchorInPhase,
  generateStructuralAnchors,
  isTerminalBoundaryAnchor,
  phaseBoundedPreviewIndex,
  snappedMarkerCoordinates,
  type StructuralDragAnchor,
} from '../cycle-drag';

const radiusFromCenter = (point: { readonly x: number; readonly y: number }): number =>
  Math.hypot(point.x - 0.5, point.y - 0.5);

describe('Cosmic Breath structural drag geometry', () => {
  it('generates exactly 26 expansion and 25 compression anchors', () => {
    expect(expansionDragAnchors).toHaveLength(26);
    expect(compressionDragAnchors).toHaveLength(25);
  });

  it('generates deterministic readonly geometry', () => {
    expect(generateStructuralAnchors('expansion')).toEqual(expansionDragAnchors);
    expect(generateStructuralAnchors('compression')).toEqual(compressionDragAnchors);
    expect(Object.isFrozen(expansionDragAnchors)).toBe(true);
    expect(Object.isFrozen(expansionDragAnchors[0])).toBe(true);
    expect(Object.isFrozen(expansionDragAnchors[0].point)).toBe(true);
  });

  it('associates every anchor with one unique canonical state', () => {
    const anchors = [...expansionDragAnchors, ...compressionDragAnchors];
    expect(new Set(anchors.map((anchor) => anchor.stateId)).size).toBe(51);
    expect(new Set(anchors.map((anchor) => anchor.canonicalIndex)).size).toBe(51);
    expect(anchors.map(canonicalIndexForAnchor)).toEqual(
      Array.from({ length: 51 }, (_, index) => index),
    );
  });

  it('maps expansion to cycle positions 1–26 and compression to 27–51', () => {
    expect(expansionDragAnchors.map((anchor) => anchor.canonicalIndex + 1)).toEqual(
      Array.from({ length: 26 }, (_, index) => index + 1),
    );
    expect(compressionDragAnchors.map((anchor) => anchor.canonicalIndex + 1)).toEqual(
      Array.from({ length: 25 }, (_, index) => index + 27),
    );
  });

  it('progresses expansion outward and compression inward', () => {
    const expansionRadii = expansionDragAnchors.map((anchor) => radiusFromCenter(anchor.point));
    const compressionRadii = compressionDragAnchors.map((anchor) => radiusFromCenter(anchor.point));
    expect(expansionRadii.every((radius, index) => index === 0 || radius > expansionRadii[index - 1])).toBe(true);
    expect(compressionRadii.every((radius, index) => index === 0 || radius < compressionRadii[index - 1])).toBe(true);
  });

  it('clamps finite normalized coordinates at every field edge', () => {
    expect(clampNormalizedPoint({ x: -2, y: 3 })).toEqual({ x: 0, y: 1 });
    expect(clampNormalizedPoint({ x: 0.25, y: 0.75 })).toEqual({ x: 0.25, y: 0.75 });
    expect(() => clampNormalizedPoint({ x: Number.NaN, y: 0.5 })).toThrow(RangeError);
  });

  it('selects the nearest anchor inside the active phase', () => {
    for (const anchor of [expansionDragAnchors[8], compressionDragAnchors[12]]) {
      expect(findNearestAnchorInPhase(anchor.point, anchor.phase)).toBe(anchor);
    }
    expect(findNearestAnchorInPhase(compressionDragAnchors[4].point, 'expansion').phase)
      .toBe('expansion');
  });

  it('breaks exact distance ties toward the lower canonical index', () => {
    const anchors: readonly StructuralDragAnchor[] = [
      {
        phase: 'expansion', phaseIndex: 2, canonicalIndex: 1,
        stateId: 'expansion-sub-ytom', label: 'sub-ytom', point: { x: 0.75, y: 0.5 },
      },
      {
        phase: 'expansion', phaseIndex: 1, canonicalIndex: 0,
        stateId: 'expansion-sub-ztom', label: 'sub-ztom', point: { x: 0.25, y: 0.5 },
      },
    ];
    expect(findNearestAnchor({ x: 0.5, y: 0.5 }, anchors).canonicalIndex).toBe(0);
  });

  it('bounds preview indices to the selected phase', () => {
    expect(phaseBoundedPreviewIndex(50, 'expansion')).toBe(25);
    expect(phaseBoundedPreviewIndex(0, 'compression')).toBe(26);
    expect(phaseBoundedPreviewIndex(9, 'expansion')).toBe(9);
    expect(phaseBoundedPreviewIndex(40, 'compression')).toBe(40);
  });

  it('returns stable snapped marker coordinates', () => {
    expect(snappedMarkerCoordinates(0)).toBe(expansionDragAnchors[0].point);
    expect(snappedMarkerCoordinates(25)).toBe(expansionDragAnchors[25].point);
    expect(snappedMarkerCoordinates(26)).toBe(compressionDragAnchors[0].point);
    expect(snappedMarkerCoordinates(50)).toBe(compressionDragAnchors[24].point);
    expect(snappedMarkerCoordinates(14)).toEqual(snappedMarkerCoordinates(14));
  });

  it('identifies atom and ztom as the terminal boundary anchors', () => {
    expect(isTerminalBoundaryAnchor(expansionDragAnchors[25])).toBe(true);
    expect(isTerminalBoundaryAnchor(compressionDragAnchors[24])).toBe(true);
    expect(isTerminalBoundaryAnchor(expansionDragAnchors[0])).toBe(false);
    expect(isTerminalBoundaryAnchor(compressionDragAnchors[0])).toBe(false);
  });

  it('produces compact previews containing only approved structural fields', () => {
    const preview = compactStructuralPreviewText(expansionDragAnchors[23]);
    expect(preview).toBe('sub-ctom · Expansion · cycle position 24 of 51');
    expect(preview).not.toMatch(/duration|formula|quantum|notation|magnitude|chronology/i);
  });

  it('fails predictably for invalid points, phases, indices, and anchors', () => {
    expect(() => findNearestAnchor({ x: -0.1, y: 0.5 }, expansionDragAnchors)).toThrow(RangeError);
    expect(() => findNearestAnchor({ x: 0.5, y: 0.5 }, [])).toThrow(RangeError);
    expect(() => generateStructuralAnchors('invalid' as 'expansion')).toThrow(RangeError);
    expect(() => phaseBoundedPreviewIndex(51, 'expansion')).toThrow(RangeError);
    expect(() => snappedMarkerCoordinates(-1)).toThrow(RangeError);
    expect(() => canonicalIndexForAnchor({
      ...expansionDragAnchors[0],
      stateId: 'compression-ztom',
    })).toThrow(RangeError);
  });
});
