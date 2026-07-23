import { describe, expect, it } from 'vitest';
import {
  INITIAL_CYCLE_STATE,
  beginDragging,
  beginNextCosmicBreath,
  completeSettling,
  continueFromAtom,
  enterExploration,
  moveToNextState,
  moveToPreviousState,
  releaseDragIntoSettling,
  resetCycleExplorer,
  returnToOverview,
  revealSettledDetails,
  selectStateById,
  selectStateByIndex,
  updateDragPreview,
} from '../cycle-state';

describe('Cosmic Breath pure cycle state engine', () => {
  it('starts in the exact overview state with no simulated selection', () => {
    expect(INITIAL_CYCLE_STATE).toEqual({
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
    expect(Object.isFrozen(INITIAL_CYCLE_STATE)).toBe(true);
  });

  it('enters exploration and selects by ID or structural index', () => {
    const entered = enterExploration(INITIAL_CYCLE_STATE);
    expect(entered).toMatchObject({
      mode: 'exploring',
      selectedIndex: 0,
      boundaryRole: 'new-cosmic-seed',
      cycleCount: 0,
    });

    const byId = selectStateById(entered, 'expansion-sub-ctom');
    expect(byId).toMatchObject({ selectedIndex: 23, mode: 'exploring' });
    const byIndex = selectStateByIndex(byId, 27);
    expect(byIndex).toMatchObject({ selectedIndex: 27, mode: 'exploring' });
  });

  it('moves ordinarily inside a phase but never crosses atom or btom', () => {
    const subBtom = selectStateById(INITIAL_CYCLE_STATE, 'expansion-sub-btom');
    const atom = moveToNextState(subBtom);
    expect(atom).toMatchObject({
      selectedIndex: 25,
      mode: 'boundary-pause',
      boundaryRole: 'expansion-pause',
    });
    expect(moveToNextState(atom)).toMatchObject({
      selectedIndex: 25,
      mode: 'boundary-pause',
    });

    const btom = continueFromAtom(atom);
    expect(btom).toMatchObject({
      selectedIndex: 26,
      mode: 'exploring',
      cycleCount: 0,
    });
    expect(moveToPreviousState(btom)).toMatchObject({
      selectedIndex: 26,
      mode: 'exploring',
    });
    expect(moveToNextState(btom).selectedIndex).toBe(27);
  });

  it('pauses at ztom and increments the cycle only on deliberate reset', () => {
    const ytom = selectStateById(INITIAL_CYCLE_STATE, 'compression-ytom');
    const ztom = moveToNextState(ytom);
    expect(ztom).toMatchObject({
      selectedIndex: 50,
      mode: 'boundary-pause',
      boundaryRole: 'reset-pause',
      cycleCount: 0,
    });
    expect(moveToNextState(ztom)).toMatchObject({
      selectedIndex: 50,
      cycleCount: 0,
    });

    const nextBreath = beginNextCosmicBreath(ztom);
    expect(nextBreath).toMatchObject({
      selectedIndex: 0,
      mode: 'exploring',
      boundaryRole: 'new-cosmic-seed',
      cycleCount: 1,
    });
    expect(moveToNextState(nextBreath).cycleCount).toBe(1);
  });

  it('tracks unrestricted direct drag previews across phase boundaries', () => {
    const expansionDrag = beginDragging(
      selectStateById(INITIAL_CYCLE_STATE, 'expansion-sub-ctom'),
    );
    expect(expansionDrag).toMatchObject({
      mode: 'dragging',
      selectedIndex: 23,
      previewIndex: 23,
      detailsIndex: null,
    });
    expect(updateDragPreview(expansionDrag, 50).previewIndex).toBe(50);
    expect(updateDragPreview(expansionDrag, 4).previewIndex).toBe(4);

    const compressionDrag = beginDragging(
      selectStateById(INITIAL_CYCLE_STATE, 'compression-ctom'),
    );
    expect(updateDragPreview(compressionDrag, 0).previewIndex).toBe(0);
    expect(updateDragPreview(compressionDrag, 49).previewIndex).toBe(49);
  });

  it('directly inspects either phase without executing guarded transitions', () => {
    const atom = selectStateById(INITIAL_CYCLE_STATE, 'expansion-atom');
    const inspectedBtom = selectStateById(atom, 'compression-btom');
    expect(inspectedBtom).toMatchObject({ selectedIndex: 26, cycleCount: 0 });
    const draggedBtom = releaseDragIntoSettling(updateDragPreview(beginDragging(atom), 26));
    expect(draggedBtom).toMatchObject({ selectedIndex: 26, cycleCount: 0 });
    expect(moveToNextState(atom).selectedIndex).toBe(25);

    const ztom = selectStateById(inspectedBtom, 'compression-ztom');
    const inspectedSubZtom = selectStateById(ztom, 'expansion-sub-ztom');
    expect(inspectedSubZtom).toMatchObject({ selectedIndex: 0, cycleCount: 0 });
    const draggedSubZtom = releaseDragIntoSettling(updateDragPreview(beginDragging(ztom), 0));
    expect(draggedSubZtom).toMatchObject({ selectedIndex: 0, cycleCount: 0 });
    expect(moveToNextState(ztom).selectedIndex).toBe(50);
    expect(beginNextCosmicBreath(ztom).cycleCount).toBe(1);
  });

  it('settles rapid alternating-phase drag selections on the newest target', () => {
    const firstDrag = updateDragPreview(
      beginDragging(selectStateById(INITIAL_CYCLE_STATE, 'expansion-sub-stom')),
      40,
    );
    const firstSettling = releaseDragIntoSettling(firstDrag);
    const staleToken = firstSettling.pendingSettleToken!;
    const newestDrag = updateDragPreview(beginDragging(firstSettling), 5);
    const newestSettling = releaseDragIntoSettling(newestDrag);
    expect(completeSettling(newestSettling, staleToken)).toBe(newestSettling);
    expect(completeSettling(newestSettling, newestSettling.pendingSettleToken!)).toMatchObject({
      selectedIndex: 5,
      settledSelectionToken: newestSettling.pendingSettleToken,
    });
  });

  it('allows direct inspection of every selectable structural state', () => {
    for (let index = 0; index < 51; index += 1) {
      expect(selectStateByIndex(INITIAL_CYCLE_STATE, index).selectedIndex).toBe(index);
    }
  });

  it('settles at the newest preview and reveals details only for its valid token', () => {
    const dragging = updateDragPreview(
      beginDragging(enterExploration(INITIAL_CYCLE_STATE)),
      10,
    );
    const settling = releaseDragIntoSettling(dragging);
    const token = settling.pendingSettleToken;
    expect(token).not.toBeNull();
    expect(settling).toMatchObject({
      mode: 'settling',
      selectedIndex: 10,
      previewIndex: null,
      detailsIndex: null,
    });

    expect(completeSettling(settling, token! - 1)).toBe(settling);
    const settled = completeSettling(settling, token!);
    expect(settled).toMatchObject({
      mode: 'exploring',
      pendingSettleToken: null,
      settledSelectionToken: token,
    });
    expect(revealSettledDetails(settled, token! - 1)).toBe(settled);
    expect(revealSettledDetails(settled, token!)).toMatchObject({
      mode: 'details',
      detailsIndex: 10,
    });
  });

  it('invalidates stale settle and detail completions when a newer interaction begins', () => {
    const settling = releaseDragIntoSettling(
      updateDragPreview(beginDragging(enterExploration(INITIAL_CYCLE_STATE)), 8),
    );
    const staleToken = settling.pendingSettleToken!;
    const newerDrag = beginDragging(settling);
    expect(newerDrag).toMatchObject({
      mode: 'dragging',
      pendingSettleToken: null,
      settledSelectionToken: null,
      detailsIndex: null,
    });
    expect(completeSettling(newerDrag, staleToken)).toBe(newerDrag);

    const newestSettling = releaseDragIntoSettling(updateDragPreview(newerDrag, 9));
    const newestToken = newestSettling.pendingSettleToken!;
    const settled = completeSettling(newestSettling, newestToken);
    const moved = moveToNextState(settled);
    expect(moved.settledSelectionToken).toBeNull();
    expect(revealSettledDetails(moved, newestToken)).toBe(moved);
  });

  it('returns and resets to the exact initial overview state', () => {
    const active = beginNextCosmicBreath(
      selectStateById(INITIAL_CYCLE_STATE, 'compression-ztom'),
    );
    expect(active.cycleCount).toBe(1);
    expect(returnToOverview(active)).toBe(INITIAL_CYCLE_STATE);
    expect(resetCycleExplorer(active)).toBe(INITIAL_CYCLE_STATE);
  });

  it('fails predictably for invalid selections and invalid operations', () => {
    expect(() => enterExploration(INITIAL_CYCLE_STATE, 'missing')).toThrow(RangeError);
    expect(() => selectStateById(INITIAL_CYCLE_STATE, 'missing')).toThrow(RangeError);
    expect(() => selectStateByIndex(INITIAL_CYCLE_STATE, -1)).toThrow(RangeError);
    expect(() => selectStateByIndex(INITIAL_CYCLE_STATE, 51)).toThrow(RangeError);
    expect(() => moveToNextState(INITIAL_CYCLE_STATE)).toThrow(/no selected/i);
    expect(() => beginDragging(INITIAL_CYCLE_STATE)).toThrow(/no selected/i);
    expect(() => updateDragPreview(enterExploration(INITIAL_CYCLE_STATE), 2)).toThrow(
      /dragging mode/i,
    );
    expect(() => updateDragPreview(beginDragging(enterExploration(INITIAL_CYCLE_STATE)), 51))
      .toThrow(RangeError);
    expect(() => continueFromAtom(enterExploration(INITIAL_CYCLE_STATE))).toThrow(
      /only while atom/i,
    );
    expect(() => beginNextCosmicBreath(enterExploration(INITIAL_CYCLE_STATE))).toThrow(
      /only while ztom/i,
    );
  });
});
