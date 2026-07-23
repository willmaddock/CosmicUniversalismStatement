import companionLedger from '../../data/cosmic-breath/CB-TOM-CONTENT-1.0.json';
import {
  getTomStateById,
  orderedTomStates,
  type StructuralTomState,
  type TomPhase,
} from './cycle-ledger';

export type StructuralStateId = StructuralTomState['id'];
export type ContentApprovalStatus = 'draft' | 'owner-reviewed' | 'owner-approved';
export type SourceAlignmentStatus =
  | 'unreviewed'
  | 'requires-source-alignment-review'
  | 'source-aligned';
export type StructuralTheme =
  | 'seed'
  | 'renewal'
  | 'emergence'
  | 'differentiation'
  | 'matter'
  | 'organization'
  | 'integration'
  | 'recursion'
  | 'intelligence'
  | 'memory'
  | 'entropy'
  | 'expansion'
  | 'compression'
  | 'convergence'
  | 'transformation'
  | 'boundary'
  | 'pause'
  | 'reset';

export const PUBLIC_EPISTEMIC_LABEL =
  'CU structural proposition — not an empirical measurement' as const;

export interface SourceProvenance {
  readonly uploadedSourceName: 'Cosmic_Breath_Calculation(6).md';
  readonly repositorySourcePath: 'ResearchFiles/Cosmic_Breath_Calculation.md';
  readonly sourceSection:
    | 'Expansion Phase (sub-ztom ➝ atom)'
    | 'Compression Phase (btom ➝ ztom)';
  readonly sourceTomLabel: string;
  readonly ownerAuthority: 'William Maddock';
}

export interface TomContentRecord {
  readonly structuralStateId: StructuralStateId;
  readonly exactQuantumStateDisplay: string;
  readonly exactDurationEstimateDisplay: string;
  readonly exactSourceCuDescription: string;
  readonly publicCuDescription?: string;
  readonly canonicalCuDescription?: string;
  readonly tomLayerExplanation: string;
  readonly quantumStateHelperShort: string;
  readonly quantumStateHelperLong: string;
  readonly durationHelperShort: string;
  readonly durationHelperLong: string;
  readonly publicEpistemicLabel: typeof PUBLIC_EPISTEMIC_LABEL;
  readonly canonicalTitle?: string;
  readonly structuralTheme?: readonly StructuralTheme[];
  readonly relatedTransitionContentIds?: readonly string[];
  readonly extendedDescription?: string;
  readonly sourceProvenance: SourceProvenance;
  readonly sourceAlignmentStatus: SourceAlignmentStatus;
  readonly contentApprovalStatus: ContentApprovalStatus;
}

export interface PublicTomContent {
  readonly structuralStateId: StructuralStateId;
  readonly sourceTomLabel: string;
  readonly phase: TomPhase;
  readonly cycleIndex: number;
  readonly cyclePosition: string;
  readonly phaseIndex: number;
  readonly exactQuantumStateDisplay: string;
  readonly exactDurationEstimateDisplay: string;
  readonly publicCuDescription: string;
  readonly publicEpistemicLabel: typeof PUBLIC_EPISTEMIC_LABEL;
  readonly canonicalTitle?: string;
  readonly structuralTheme?: readonly StructuralTheme[];
  readonly relatedTransitionContentIds?: readonly string[];
  readonly tomLayerExplanation: string;
  readonly quantumStateHelperShort: string;
  readonly quantumStateHelperLong: string;
  readonly durationHelperShort: string;
  readonly durationHelperLong: string;
}

export const PUBLIC_TOM_CONTENT_KEYS = Object.freeze([
  'structuralStateId',
  'sourceTomLabel',
  'phase',
  'cycleIndex',
  'cyclePosition',
  'phaseIndex',
  'exactQuantumStateDisplay',
  'exactDurationEstimateDisplay',
  'publicCuDescription',
  'publicEpistemicLabel',
  'canonicalTitle',
  'structuralTheme',
  'relatedTransitionContentIds',
  'tomLayerExplanation',
  'quantumStateHelperShort',
  'quantumStateHelperLong',
  'durationHelperShort',
  'durationHelperLong',
] as const satisfies readonly (keyof PublicTomContent)[]);

export interface DeclaredContentPhaseTotals {
  readonly expansion: 'approximately 2.8 trillion years';
  readonly compression: 'approximately 308 billion years';
  readonly completeCosmicBreath: 'approximately 3.108 trillion years';
  readonly derivation: 'Owner-approved declared CU totals; not recomputed from state rows.';
}

export interface ValidatedContentLedger {
  readonly ledgerId: 'CB-TOM-CONTENT-1.0';
  readonly owner: 'William Maddock';
  readonly status: 'companion-content-authority-in-development';
  readonly structuralAuthority: 'CB-TOM-STRUCTURAL-1.0';
  readonly publicEpistemicLabel: typeof PUBLIC_EPISTEMIC_LABEL;
  readonly declaredPhaseTotals: DeclaredContentPhaseTotals;
  readonly stateContent: readonly TomContentRecord[];
}

export interface TransitionContentRecord {
  readonly transitionId:
    | 'transition-atom-to-btom'
    | 'transition-ztom-to-next-sub-ztom';
  readonly canonicalTitle?: string;
  readonly canonicalCuDescription?: string;
  readonly structuralTheme?: readonly StructuralTheme[];
  readonly contentApprovalStatus: ContentApprovalStatus;
}

const CONTENT_APPROVAL_STATUSES = Object.freeze([
  'draft',
  'owner-reviewed',
  'owner-approved',
] as const);
const SOURCE_ALIGNMENT_STATUSES = Object.freeze([
  'unreviewed',
  'requires-source-alignment-review',
  'source-aligned',
] as const);
const STRUCTURAL_THEMES = Object.freeze([
  'seed',
  'renewal',
  'emergence',
  'differentiation',
  'matter',
  'organization',
  'integration',
  'recursion',
  'intelligence',
  'memory',
  'entropy',
  'expansion',
  'compression',
  'convergence',
  'transformation',
  'boundary',
  'pause',
  'reset',
] as const);
const TRANSITION_IDS = Object.freeze([
  'transition-atom-to-btom',
  'transition-ztom-to-next-sub-ztom',
] as const);
const STATE_OPTIONAL_STRING_FIELDS = Object.freeze([
  'publicCuDescription',
  'canonicalCuDescription',
  'canonicalTitle',
  'extendedDescription',
] as const);
const TRANSITION_FORBIDDEN_FIELDS = Object.freeze([
  'structuralStateId',
  'sourceTomLabel',
  'phase',
  'cycleIndex',
  'cyclePosition',
  'phaseIndex',
  'exactQuantumStateDisplay',
  'exactDurationEstimateDisplay',
] as const);

const fail = (message: string): never => {
  throw new Error('Invalid Cosmic Breath content ledger: ' + message);
};
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const requireRecord = (value: unknown, name: string): Record<string, unknown> =>
  isRecord(value) ? value : fail(name + ' must be an object');
const requireString = (value: unknown, name: string): string =>
  typeof value === 'string' && value.length > 0
    ? value
    : fail(name + ' must be a non-empty string');
const requireExact = <T extends string>(value: unknown, expected: T, name: string): T =>
  value === expected ? expected : fail(name + ' must equal "' + expected + '"');
const optionalString = (record: Record<string, unknown>, field: string, name: string): string | undefined =>
  record[field] === undefined ? undefined : requireString(record[field], name + '.' + field);
const requireEnum = <T extends string>(
  value: unknown,
  allowed: readonly T[],
  name: string,
): T => {
  const entry = requireString(value, name);
  return allowed.includes(entry as T) ? entry as T : fail(name + ' is not approved');
};

const parseThemes = (value: unknown, name: string): readonly StructuralTheme[] | undefined => {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length < 1 || value.length > 4) {
    fail(name + ' must contain one to four themes');
  }
  const themes = value.map((entry, index) =>
    requireEnum(entry, STRUCTURAL_THEMES, name + '[' + index + ']'));
  if (new Set(themes).size !== themes.length) fail(name + ' must not contain duplicates');
  const themeIndexes = themes.map((theme) => STRUCTURAL_THEMES.indexOf(theme));
  if (!themeIndexes.every((index, position) => position === 0 || index > themeIndexes[position - 1])) {
    fail(name + ' must use controlled-vocabulary order');
  }
  return Object.freeze(themes);
};

const parseTransitionIds = (value: unknown, name: string): readonly string[] | undefined => {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) fail(name + ' must be an array');
  const ids = value.map((entry, index) => requireEnum(entry, TRANSITION_IDS, name + '[' + index + ']'));
  if (new Set(ids).size !== ids.length) fail(name + ' must not contain duplicates');
  return Object.freeze(ids);
};

const parseProvenance = (
  value: unknown,
  structuralState: StructuralTomState,
  name: string,
): SourceProvenance => {
  const record = requireRecord(value, name);
  const expectedSection = structuralState.phase === 'expansion'
    ? 'Expansion Phase (sub-ztom ➝ atom)'
    : 'Compression Phase (btom ➝ ztom)';
  return Object.freeze({
    uploadedSourceName: requireExact(
      record.uploadedSourceName,
      'Cosmic_Breath_Calculation(6).md',
      name + '.uploadedSourceName',
    ),
    repositorySourcePath: requireExact(
      record.repositorySourcePath,
      'ResearchFiles/Cosmic_Breath_Calculation.md',
      name + '.repositorySourcePath',
    ),
    sourceSection: requireExact(record.sourceSection, expectedSection, name + '.sourceSection'),
    sourceTomLabel: requireExact(
      record.sourceTomLabel,
      structuralState.label,
      name + '.sourceTomLabel',
    ),
    ownerAuthority: requireExact(
      record.ownerAuthority,
      'William Maddock',
      name + '.ownerAuthority',
    ),
  });
};

const parseStateContent = (value: unknown, index: number): TomContentRecord => {
  const name = 'stateContent[' + index + ']';
  const record = requireRecord(value, name);
  const structuralStateId = requireString(record.structuralStateId, name + '.structuralStateId');
  let structuralState: StructuralTomState;
  try {
    structuralState = getTomStateById(structuralStateId);
  } catch {
    return fail(name + ' has unknown structuralStateId "' + structuralStateId + '"');
  }
  const optionalFields = Object.fromEntries(
    STATE_OPTIONAL_STRING_FIELDS
      .map((field) => [field, optionalString(record, field, name)] as const)
      .filter((entry): entry is readonly [string, string] => entry[1] !== undefined),
  );
  const themes = parseThemes(record.structuralTheme, name + '.structuralTheme');
  const transitionIds = parseTransitionIds(
    record.relatedTransitionContentIds,
    name + '.relatedTransitionContentIds',
  );

  return Object.freeze({
    structuralStateId,
    exactQuantumStateDisplay: requireString(
      record.exactQuantumStateDisplay,
      name + '.exactQuantumStateDisplay',
    ),
    exactDurationEstimateDisplay: requireString(
      record.exactDurationEstimateDisplay,
      name + '.exactDurationEstimateDisplay',
    ),
    exactSourceCuDescription: requireString(
      record.exactSourceCuDescription,
      name + '.exactSourceCuDescription',
    ),
    tomLayerExplanation: requireString(
      record.tomLayerExplanation,
      name + '.tomLayerExplanation',
    ),
    quantumStateHelperShort: requireString(
      record.quantumStateHelperShort,
      name + '.quantumStateHelperShort',
    ),
    quantumStateHelperLong: requireString(
      record.quantumStateHelperLong,
      name + '.quantumStateHelperLong',
    ),
    durationHelperShort: requireString(
      record.durationHelperShort,
      name + '.durationHelperShort',
    ),
    durationHelperLong: requireString(
      record.durationHelperLong,
      name + '.durationHelperLong',
    ),
    ...optionalFields,
    publicEpistemicLabel: requireExact(
      record.publicEpistemicLabel,
      PUBLIC_EPISTEMIC_LABEL,
      name + '.publicEpistemicLabel',
    ),
    ...(themes === undefined ? {} : { structuralTheme: themes }),
    ...(transitionIds === undefined ? {} : { relatedTransitionContentIds: transitionIds }),
    sourceProvenance: parseProvenance(
      record.sourceProvenance,
      structuralState,
      name + '.sourceProvenance',
    ),
    sourceAlignmentStatus: requireEnum(
      record.sourceAlignmentStatus,
      SOURCE_ALIGNMENT_STATUSES,
      name + '.sourceAlignmentStatus',
    ),
    contentApprovalStatus: requireEnum(
      record.contentApprovalStatus,
      CONTENT_APPROVAL_STATUSES,
      name + '.contentApprovalStatus',
    ),
  });
};

export const parseContentLedger = (value: unknown): ValidatedContentLedger => {
  const record = requireRecord(value, 'ledger');
  if (!Array.isArray(record.stateContent)) fail('stateContent must be an array');
  if (record.stateContent.length !== 51) fail('stateContent must contain exactly 51 records');
  const parsedRecords = record.stateContent.map(parseStateContent);
  const recordIds = parsedRecords.map((entry) => entry.structuralStateId);
  if (new Set(recordIds).size !== recordIds.length) fail('structuralStateId values must be unique');
  const byId = new Map(parsedRecords.map((entry) => [entry.structuralStateId, entry]));
  const orderedRecords = orderedTomStates.map((state) => {
    const content = byId.get(state.id);
    return content ?? fail('missing content record for structural state "' + state.id + '"');
  });
  const totals = requireRecord(record.declaredPhaseTotals, 'declaredPhaseTotals');

  return Object.freeze({
    ledgerId: requireExact(record.ledgerId, 'CB-TOM-CONTENT-1.0', 'ledgerId'),
    owner: requireExact(record.owner, 'William Maddock', 'owner'),
    status: requireExact(
      record.status,
      'companion-content-authority-in-development',
      'status',
    ),
    structuralAuthority: requireExact(
      record.structuralAuthority,
      'CB-TOM-STRUCTURAL-1.0',
      'structuralAuthority',
    ),
    publicEpistemicLabel: requireExact(
      record.publicEpistemicLabel,
      PUBLIC_EPISTEMIC_LABEL,
      'publicEpistemicLabel',
    ),
    declaredPhaseTotals: Object.freeze({
      expansion: requireExact(
        totals.expansion,
        'approximately 2.8 trillion years',
        'declaredPhaseTotals.expansion',
      ),
      compression: requireExact(
        totals.compression,
        'approximately 308 billion years',
        'declaredPhaseTotals.compression',
      ),
      completeCosmicBreath: requireExact(
        totals.completeCosmicBreath,
        'approximately 3.108 trillion years',
        'declaredPhaseTotals.completeCosmicBreath',
      ),
      derivation: requireExact(
        totals.derivation,
        'Owner-approved declared CU totals; not recomputed from state rows.',
        'declaredPhaseTotals.derivation',
      ),
    }),
    stateContent: Object.freeze(orderedRecords),
  });
};

const publicProjection = (content: TomContentRecord): PublicTomContent => {
  const structural = getTomStateById(content.structuralStateId);
  return Object.freeze({
    structuralStateId: content.structuralStateId,
    sourceTomLabel: content.sourceProvenance.sourceTomLabel,
    phase: structural.phase,
    cycleIndex: structural.cycleIndex,
    cyclePosition: structural.cycleIndex + ' of ' + orderedTomStates.length,
    phaseIndex: structural.phaseIndex,
    exactQuantumStateDisplay: content.exactQuantumStateDisplay,
    exactDurationEstimateDisplay: content.exactDurationEstimateDisplay,
    publicCuDescription: content.publicCuDescription ?? content.exactSourceCuDescription,
    publicEpistemicLabel: content.publicEpistemicLabel,
    ...(content.canonicalTitle === undefined ? {} : { canonicalTitle: content.canonicalTitle }),
    ...(content.structuralTheme === undefined
      ? {}
      : { structuralTheme: Object.freeze([...content.structuralTheme]) }),
    ...(content.relatedTransitionContentIds === undefined
      ? {}
      : { relatedTransitionContentIds: Object.freeze([...content.relatedTransitionContentIds]) }),
    tomLayerExplanation: content.tomLayerExplanation,
    quantumStateHelperShort: content.quantumStateHelperShort,
    quantumStateHelperLong: content.quantumStateHelperLong,
    durationHelperShort: content.durationHelperShort,
    durationHelperLong: content.durationHelperLong,
  });
};

export const parseTransitionContentRecord = (value: unknown): TransitionContentRecord => {
  const record = requireRecord(value, 'transitionContent');
  for (const field of TRANSITION_FORBIDDEN_FIELDS) {
    if (field in record) fail('transitionContent must not contain "' + field + '"');
  }
  const transitionId = requireEnum(record.transitionId, TRANSITION_IDS, 'transitionContent.transitionId');
  const canonicalTitle = optionalString(record, 'canonicalTitle', 'transitionContent');
  const canonicalCuDescription = optionalString(record, 'canonicalCuDescription', 'transitionContent');
  const themes = parseThemes(record.structuralTheme, 'transitionContent.structuralTheme');
  return Object.freeze({
    transitionId,
    ...(canonicalTitle === undefined ? {} : { canonicalTitle }),
    ...(canonicalCuDescription === undefined ? {} : { canonicalCuDescription }),
    ...(themes === undefined ? {} : { structuralTheme: themes }),
    contentApprovalStatus: requireEnum(
      record.contentApprovalStatus,
      CONTENT_APPROVAL_STATUSES,
      'transitionContent.contentApprovalStatus',
    ),
  });
};

export const contentLedger = parseContentLedger(companionLedger as unknown);
export const orderedTomContent = contentLedger.stateContent;
export const orderedPublicTomContent = Object.freeze(orderedTomContent.map(publicProjection));

const PUBLIC_TOM_CONTENT_KEY_SET = new Set<string>(PUBLIC_TOM_CONTENT_KEYS);

export const serializePublicTomContentPayload = (
  records: readonly unknown[] = orderedPublicTomContent,
): string => {
  if (records.length !== orderedTomStates.length) {
    fail('public payload must contain exactly 51 records');
  }

  const structuralIds = new Set<string>();
  records.forEach((value, index) => {
    const record = requireRecord(value, 'publicPayload[' + index + ']');
    for (const key of Object.keys(record)) {
      if (!PUBLIC_TOM_CONTENT_KEY_SET.has(key)) {
        fail('publicPayload[' + index + '] contains non-public key "' + key + '"');
      }
    }
    structuralIds.add(requireString(
      record.structuralStateId,
      'publicPayload[' + index + '].structuralStateId',
    ));
  });

  if (structuralIds.size !== orderedTomStates.length) {
    fail('public payload structuralStateId values must be unique');
  }
  for (const state of orderedTomStates) {
    if (!structuralIds.has(state.id)) {
      fail('public payload is missing structural state "' + state.id + '"');
    }
  }

  return JSON.stringify(records).replace(
    /[<>&\u2028\u2029]/g,
    (character) => ({
      '<': '\\u003c',
      '>': '\\u003e',
      '&': '\\u0026',
      '\u2028': '\\u2028',
      '\u2029': '\\u2029',
    })[character] ?? character,
  );
};

const contentById = new Map(orderedTomContent.map((entry) => [entry.structuralStateId, entry]));
const publicContentById = new Map(
  orderedPublicTomContent.map((entry) => [entry.structuralStateId, entry]),
);

export const getTomContentById = (id: StructuralStateId): TomContentRecord =>
  contentById.get(id) ?? fail('unknown structuralStateId "' + id + '"');

export const getPublicTomContentById = (id: StructuralStateId): PublicTomContent =>
  publicContentById.get(id) ?? fail('unknown structuralStateId "' + id + '"');
