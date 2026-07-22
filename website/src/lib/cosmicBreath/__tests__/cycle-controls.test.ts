import { describe, expect, it } from 'vitest';
import { getTomStateById } from '../cycle-ledger';
import {
  clampStructuralIndex,
  getExplorerControlState,
  getExplorerKeyboardCommand,
  getPhaseBoundaryIndices,
  getSelectionStatusText,
  getStructuralStateSummary,
  isEditableControl,
  parseStructuralIndex,
} from '../cycle-controls';

describe('Cosmic Breath semantic explorer controls', () => {
  it('maps every approved explorer shortcut and ignores other keys', () => {
    expect(getExplorerKeyboardCommand('ArrowLeft')).toBe('previous');
    expect(getExplorerKeyboardCommand('ArrowRight')).toBe('next');
    expect(getExplorerKeyboardCommand('Home')).toBe('first');
    expect(getExplorerKeyboardCommand('End')).toBe('last');
    expect(getExplorerKeyboardCommand('PageUp')).toBe('phase-start');
    expect(getExplorerKeyboardCommand('PageDown')).toBe('phase-end');
    expect(getExplorerKeyboardCommand('Escape')).toBe('return-overview');
    expect(getExplorerKeyboardCommand('Enter')).toBeNull();
    expect(getExplorerKeyboardCommand('a')).toBeNull();
  });

  it('excludes form fields and editable content from explorer shortcuts', () => {
    expect(isEditableControl({ tagName: 'input' })).toBe(true);
    expect(isEditableControl({ tagName: 'TEXTAREA' })).toBe(true);
    expect(isEditableControl({ tagName: 'select' })).toBe(true);
    expect(isEditableControl({ tagName: 'DIV', isContentEditable: true })).toBe(true);
    expect(isEditableControl({ tagName: 'BUTTON' })).toBe(false);
    expect(isEditableControl(null)).toBe(false);
  });

  it('rejects invalid range indices and clamps finite preview values predictably', () => {
    expect(parseStructuralIndex('0')).toBe(0);
    expect(parseStructuralIndex(50)).toBe(50);
    expect(() => parseStructuralIndex('')).toThrow(RangeError);
    expect(() => parseStructuralIndex(-1)).toThrow(RangeError);
    expect(() => parseStructuralIndex(51)).toThrow(RangeError);
    expect(() => parseStructuralIndex(1.5)).toThrow(RangeError);
    expect(() => parseStructuralIndex('not-a-number')).toThrow(RangeError);
    expect(clampStructuralIndex(-8)).toBe(0);
    expect(clampStructuralIndex(12.6)).toBe(13);
    expect(clampStructuralIndex(80)).toBe(50);
    expect(() => clampStructuralIndex(Number.NaN)).toThrow(RangeError);
  });

  it('returns phase bounds for Home-like phase navigation', () => {
    expect(getPhaseBoundaryIndices('expansion')).toEqual({ start: 0, end: 25 });
    expect(getPhaseBoundaryIndices('compression')).toEqual({ start: 26, end: 50 });
  });

  it('derives ordinary and guarded boundary controls from the canonical states', () => {
    expect(getExplorerControlState(getTomStateById('expansion-sub-ztom'))).toEqual({
      previousDisabled: true,
      nextDisabled: false,
      showBeginCompression: false,
      showBeginNextBreath: false,
    });
    expect(getExplorerControlState(getTomStateById('expansion-atom'))).toEqual({
      previousDisabled: false,
      nextDisabled: true,
      showBeginCompression: true,
      showBeginNextBreath: false,
    });
    expect(getExplorerControlState(getTomStateById('compression-btom'))).toEqual({
      previousDisabled: true,
      nextDisabled: false,
      showBeginCompression: false,
      showBeginNextBreath: false,
    });
    expect(getExplorerControlState(getTomStateById('compression-ztom'))).toEqual({
      previousDisabled: false,
      nextDisabled: true,
      showBeginCompression: false,
      showBeginNextBreath: true,
    });
  });

  it('distinguishes direct educational jumps from sequential movement', () => {
    const state = getTomStateById('expansion-sub-ctom');
    expect(getSelectionStatusText('direct', state)).toMatch(/^Direct educational jump/);
    expect(getSelectionStatusText('sequential', state)).toMatch(/^Moved to/);
    expect(getSelectionStatusText('settled', state)).toMatch(/details are ready/);
  });

  it('exposes only approved structural fields in summaries', () => {
    const summary = getStructuralStateSummary(getTomStateById('expansion-atom'));
    expect(summary).toEqual({
      label: 'atom',
      phase: 'Expansion',
      cyclePosition: '26 of 51',
      phasePosition: '26 of 26',
      previousState: 'sub-btom',
      nextState: 'No ordinary next state',
      boundaryRole: 'Expansion pause',
      classification: 'CU theoretical proposition',
      approvalStatus: 'Owner-approved structural decision',
    });
    expect(Object.keys(summary)).not.toEqual(expect.arrayContaining([
      'duration',
      'quantumNotation',
      'machineMagnitude',
      'formula',
      'chronologySensitiveDescription',
    ]));
  });
});
