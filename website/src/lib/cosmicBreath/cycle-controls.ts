import {
  getTomStateById,
  orderedTomStates,
  type StructuralTomState,
  type TomPhase,
} from './cycle-ledger';

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
  readonly classification: 'CU theoretical proposition';
  readonly approvalStatus: 'Owner-approved structural decision';
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

export const getExplorerKeyboardCommand = (
  key: string,
): ExplorerKeyboardCommand | null => keyboardCommands[key] ?? null;

export const isEditableControl = (target: EditableTargetLike | null): boolean => {
  if (!target) return false;
  const tagName = target.tagName?.toUpperCase();
  return target.isContentEditable === true
    || tagName === 'INPUT'
    || tagName === 'TEXTAREA'
    || tagName === 'SELECT';
};

export const parseStructuralIndex = (value: string | number): number => {
  if (typeof value === 'string' && value.trim() === '') {
    throw new RangeError('Structural TOM index must be an integer from 0 to 50');
  }
  const index = typeof value === 'number' ? value : Number(value);
  if (!Number.isInteger(index) || index < 0 || index >= orderedTomStates.length) {
    throw new RangeError('Structural TOM index must be an integer from 0 to 50: ' + value);
  }
  return index;
};

export const clampStructuralIndex = (value: number): number => {
  if (!Number.isFinite(value)) {
    throw new RangeError('Structural TOM index must be a finite number');
  }
  return Math.min(orderedTomStates.length - 1, Math.max(0, Math.round(value)));
};

export const getPhaseBoundaryIndices = (
  phase: TomPhase,
): Readonly<{ start: number; end: number }> => phase === 'expansion'
  ? Object.freeze({ start: 0, end: 25 })
  : Object.freeze({ start: 26, end: 50 });

export const getExplorerControlState = (
  state: StructuralTomState,
): ExplorerControlState => Object.freeze({
  previousDisabled: state.previousId === null,
  nextDisabled: state.nextId === null,
  showBeginCompression: state.boundaryRole === 'expansion-pause',
  showBeginNextBreath: state.boundaryRole === 'reset-pause',
});

const labelForId = (id: string | null, emptyLabel: string): string =>
  id === null ? emptyLabel : getTomStateById(id).label;

const boundaryRoleLabels = Object.freeze({
  'new-cosmic-seed': 'New cosmic seed',
  'expansion-pause': 'Expansion pause',
  'reset-pause': 'Reset pause',
});

export const getStructuralStateSummary = (
  state: StructuralTomState,
): StructuralStateSummary => Object.freeze({
  label: state.label,
  phase: state.phase === 'expansion' ? 'Expansion' : 'Compression',
  cyclePosition: `${state.cycleIndex} of ${orderedTomStates.length}`,
  phasePosition: `${state.phaseIndex} of ${state.phase === 'expansion' ? 26 : 25}`,
  previousState: labelForId(state.previousId, 'No ordinary previous state'),
  nextState: labelForId(state.nextId, 'No ordinary next state'),
  boundaryRole: state.boundaryRole === null
    ? 'None'
    : boundaryRoleLabels[state.boundaryRole],
  classification: 'CU theoretical proposition',
  approvalStatus: 'Owner-approved structural decision',
});

export const getSelectionStatusText = (
  kind: SelectionStatusKind,
  state: StructuralTomState,
): string => {
  const position = `${state.label}, ${state.phase} state ${state.phaseIndex}`;
  if (kind === 'direct') return `Direct educational jump to ${position}. Settling.`;
  if (kind === 'settling') return `${position}. Settling.`;
  if (kind === 'settled') return `${position}. Structural details are ready.`;
  return `Moved to ${position}. Settling.`;
};
