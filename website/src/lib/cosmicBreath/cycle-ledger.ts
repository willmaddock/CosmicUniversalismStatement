import canonicalLedger from '../../data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json';

export type TomPhase = 'expansion' | 'compression';
export type TomBoundaryRole =
  | 'new-cosmic-seed'
  | 'expansion-pause'
  | 'reset-pause'
  | null;
export type TomClassification = 'cu-theoretical-proposition';
export type StructuralApprovalStatus = 'owner-approved-structural';
export type GuardedTransitionType = 'phase-continuation' | 'next-breath-reset';

export interface StructuralTomState {
  readonly id: string;
  readonly label: string;
  readonly phase: TomPhase;
  readonly cycleIndex: number;
  readonly phaseIndex: number;
  readonly previousId: string | null;
  readonly nextId: string | null;
  readonly boundaryRole: TomBoundaryRole;
  readonly classification: TomClassification;
  readonly approvalStatus: StructuralApprovalStatus;
  readonly provenance: readonly string[];
}

export interface GuardedTomTransition {
  readonly id: string;
  readonly fromId: string;
  readonly toId: string;
  readonly toCycleOffset: 0 | 1;
  readonly type: GuardedTransitionType;
  readonly label: string;
  readonly requiresExplicitAction: true;
  readonly selectableTomState: false;
  readonly approvalStatus: StructuralApprovalStatus;
  readonly provenance: readonly string[];
}

export interface DeclaredPhaseAnchors {
  readonly expansion: string;
  readonly compression: string;
  readonly completeCosmicBreath: string;
  readonly approvalStatus: 'owner-approved-model-level-anchors';
  readonly derivation: string;
}

export interface CompleteStructuralLedger {
  readonly ledgerId: 'CB-TOM-STRUCTURAL-1.0';
  readonly owner: 'William Maddock';
  readonly ownerDecisionDate: '2026-07-22';
  readonly status: 'owner-approved-structural-decisions-numerical-fields-withheld';
  readonly declaredPhaseAnchors: DeclaredPhaseAnchors;
  readonly withheldFields: readonly string[];
  readonly states: readonly StructuralTomState[];
  readonly transitions: readonly GuardedTomTransition[];
}

const EXPECTED_WITHHELD_FIELDS = Object.freeze([
  'quantumNotation',
  'duration',
  'machineMagnitude',
  'chronologySensitiveDescription',
] as const);
const FORBIDDEN_STATE_FIELDS = Object.freeze([
  ...EXPECTED_WITHHELD_FIELDS,
  'formula',
  'formulaIndex',
  'quantumState',
]);

const fail = (message: string): never => {
  throw new Error('Invalid canonical Cosmic Breath ledger: ' + message);
};
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const requireRecord = (value: unknown, name: string): Record<string, unknown> =>
  isRecord(value) ? value : fail(name + ' must be an object');
const requireString = (value: unknown, name: string): string =>
  typeof value === 'string' ? value : fail(name + ' must be a string');
const requireInteger = (value: unknown, name: string): number =>
  Number.isInteger(value) ? value as number : fail(name + ' must be an integer');
const requireNullableString = (value: unknown, name: string): string | null =>
  value === null ? null : requireString(value, name);
const requireStringArray = (value: unknown, name: string): readonly string[] => {
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === 'string')) {
    fail(name + ' must be an array of strings');
  }
  return Object.freeze([...value]);
};
const requireExact = <T extends string>(value: unknown, expected: T, name: string): T =>
  value === expected ? expected : fail(name + ' must equal "' + expected + '"');

const parseState = (value: unknown, index: number): StructuralTomState => {
  const name = 'states[' + index + ']';
  const record = requireRecord(value, name);
  for (const field of FORBIDDEN_STATE_FIELDS) {
    if (field in record) fail(name + ' must not contain withheld field "' + field + '"');
  }

  const phase = requireString(record.phase, name + '.phase');
  if (phase !== 'expansion' && phase !== 'compression') {
    fail(name + '.phase must be "expansion" or "compression"');
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
    classification: requireExact(
      record.classification,
      'cu-theoretical-proposition',
      name + '.classification',
    ),
    approvalStatus: requireExact(
      record.approvalStatus,
      'owner-approved-structural',
      name + '.approvalStatus',
    ),
    provenance: requireStringArray(record.provenance, name + '.provenance'),
  });
};

const parseTransition = (value: unknown, index: number): GuardedTomTransition => {
  const name = 'transitions[' + index + ']';
  const record = requireRecord(value, name);
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
    approvalStatus: requireExact(
      record.approvalStatus,
      'owner-approved-structural',
      name + '.approvalStatus',
    ),
    provenance: requireStringArray(record.provenance, name + '.provenance'),
  });
};

const validateStructuralInvariants = (ledger: CompleteStructuralLedger): void => {
  const { states, transitions } = ledger;
  if (states.length !== 51) fail('states must contain exactly 51 entries');
  if (new Set(states.map((state) => state.id)).size !== 51) fail('state IDs must be unique');
  if (!states.every((state, index) => state.cycleIndex === index + 1)) {
    fail('cycleIndex values must be consecutive from 1 through 51');
  }

  const expansion = states.filter((state) => state.phase === 'expansion');
  const compression = states.filter((state) => state.phase === 'compression');
  if (expansion.length !== 26 || compression.length !== 25) {
    fail('states must contain exactly 26 expansion and 25 compression entries');
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
        fail('invalid previous adjacency for ' + state.id);
      }
    }
    if (state.nextId !== null) {
      const next = byId.get(state.nextId);
      if (!next || next.previousId !== state.id || next.phase !== state.phase) {
        fail('invalid next adjacency for ' + state.id);
      }
    }
  }

  const subZtom = byId.get('expansion-sub-ztom');
  const atom = byId.get('expansion-atom');
  const btom = byId.get('compression-btom');
  const ztom = byId.get('compression-ztom');
  if (subZtom?.cycleIndex !== 1 || subZtom.boundaryRole !== 'new-cosmic-seed' || subZtom.previousId !== null) {
    fail('sub-ztom must be the opening new-cosmic-seed state');
  }
  if (atom?.cycleIndex !== 26 || atom.boundaryRole !== 'expansion-pause' || atom.nextId !== null) {
    fail('atom must be the final guarded expansion state');
  }
  if (btom?.cycleIndex !== 27 || btom.previousId !== null) {
    fail('btom must be the opening guarded compression state');
  }
  if (ztom?.cycleIndex !== 51 || ztom.boundaryRole !== 'reset-pause' || ztom.nextId !== null) {
    fail('ztom must be the final guarded compression state');
  }

  if (transitions.length !== 2 || transitions.some((transition) => transition.selectableTomState)) {
    fail('exactly two non-selectable guarded transitions are required');
  }
  const atomTransition = transitions.find(
    (transition) => transition.id === 'transition-atom-to-btom',
  );
  const resetTransition = transitions.find(
    (transition) => transition.id === 'transition-ztom-to-next-sub-ztom',
  );
  if (
    !atomTransition
    || atomTransition.fromId !== 'expansion-atom'
    || atomTransition.toId !== 'compression-btom'
    || atomTransition.toCycleOffset !== 0
  ) {
    fail('atom-to-btom transition is invalid');
  }
  if (
    !resetTransition
    || resetTransition.fromId !== 'compression-ztom'
    || resetTransition.toId !== 'expansion-sub-ztom'
    || resetTransition.toCycleOffset !== 1
  ) {
    fail('ztom-to-next-sub-ztom transition is invalid');
  }
};

const parseLedger = (value: unknown): CompleteStructuralLedger => {
  const record = requireRecord(value, 'ledger');
  if (!Array.isArray(record.states)) fail('states must be an array');
  if (!Array.isArray(record.transitions)) fail('transitions must be an array');
  const anchors = requireRecord(record.declaredPhaseAnchors, 'declaredPhaseAnchors');
  const withheldFields = requireStringArray(record.withheldFields, 'withheldFields');
  if (
    withheldFields.length !== EXPECTED_WITHHELD_FIELDS.length
    || !EXPECTED_WITHHELD_FIELDS.every((field, index) => withheldFields[index] === field)
  ) {
    fail('withheldFields must preserve the approved field list and order');
  }

  const ledger: CompleteStructuralLedger = Object.freeze({
    ledgerId: requireExact(record.ledgerId, 'CB-TOM-STRUCTURAL-1.0', 'ledgerId'),
    owner: requireExact(record.owner, 'William Maddock', 'owner'),
    ownerDecisionDate: requireExact(record.ownerDecisionDate, '2026-07-22', 'ownerDecisionDate'),
    status: requireExact(
      record.status,
      'owner-approved-structural-decisions-numerical-fields-withheld',
      'status',
    ),
    declaredPhaseAnchors: Object.freeze({
      expansion: requireString(anchors.expansion, 'declaredPhaseAnchors.expansion'),
      compression: requireString(anchors.compression, 'declaredPhaseAnchors.compression'),
      completeCosmicBreath: requireString(
        anchors.completeCosmicBreath,
        'declaredPhaseAnchors.completeCosmicBreath',
      ),
      approvalStatus: requireExact(
        anchors.approvalStatus,
        'owner-approved-model-level-anchors',
        'declaredPhaseAnchors.approvalStatus',
      ),
      derivation: requireString(anchors.derivation, 'declaredPhaseAnchors.derivation'),
    }),
    withheldFields,
    states: Object.freeze(record.states.map(parseState)),
    transitions: Object.freeze(record.transitions.map(parseTransition)),
  });

  validateStructuralInvariants(ledger);
  return ledger;
};

export const structuralLedger = parseLedger(canonicalLedger as unknown);
export const orderedTomStates = structuralLedger.states;
export const guardedTransitions = structuralLedger.transitions;
export const expansionTomStates = Object.freeze(
  orderedTomStates.filter((state) => state.phase === 'expansion'),
);
export const compressionTomStates = Object.freeze(
  orderedTomStates.filter((state) => state.phase === 'compression'),
);

const stateById = new Map(orderedTomStates.map((state) => [state.id, state]));
const indexById = new Map(orderedTomStates.map((state, index) => [state.id, index]));

export const atomToBtomTransition = guardedTransitions.find(
  (transition) => transition.id === 'transition-atom-to-btom',
)!;
export const ztomToNextSubZtomTransition = guardedTransitions.find(
  (transition) => transition.id === 'transition-ztom-to-next-sub-ztom',
)!;

export const getTomStateById = (id: string): StructuralTomState => {
  const state = stateById.get(id);
  if (!state) throw new RangeError('Unknown structural TOM state ID: ' + id);
  return state;
};

export const getTomStateByIndex = (index: number): StructuralTomState => {
  if (!Number.isInteger(index) || index < 0 || index >= orderedTomStates.length) {
    throw new RangeError('Structural TOM index must be an integer from 0 to 50: ' + index);
  }
  return orderedTomStates[index];
};

export const getTomStateIndex = (id: string): number => {
  const index = indexById.get(id);
  if (index === undefined) throw new RangeError('Unknown structural TOM state ID: ' + id);
  return index;
};
