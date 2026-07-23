import { structuralRuntimeLedger } from './cycle-ledger';
import {
  createCycleControlEngine,
  getExplorerControlState,
  getExplorerKeyboardCommand,
  getSelectionStatusText,
  isEditableControl,
  type RuntimeTomState,
  type StructuralStateSummary as RuntimeStructuralStateSummary,
} from './cycle-runtime';

export type {
  EditableTargetLike,
  ExplorerControlState,
  ExplorerKeyboardCommand,
  SelectionStatusKind,
} from './cycle-runtime';
export {
  getExplorerControlState,
  getExplorerKeyboardCommand,
  getSelectionStatusText,
  isEditableControl,
};

const engine = createCycleControlEngine(structuralRuntimeLedger);

export const {
  parseStructuralIndex,
  clampStructuralIndex,
  getPhaseBoundaryIndices,
} = engine;

export interface StructuralStateSummary extends RuntimeStructuralStateSummary {
  readonly classification: 'CU theoretical proposition';
  readonly approvalStatus: 'Owner-approved structural decision';
}

export const getStructuralStateSummary = (
  state: RuntimeTomState,
): StructuralStateSummary => Object.freeze({
  ...engine.getStructuralStateSummary(state),
  classification: 'CU theoretical proposition',
  approvalStatus: 'Owner-approved structural decision',
});
