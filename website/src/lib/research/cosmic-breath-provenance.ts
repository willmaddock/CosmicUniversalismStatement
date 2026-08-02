import rawAuthority from '../../data/research/CU-COSMIC-BREATH-PROVENANCE-1.0.json';

// RC-3A-1 defines the governed authority types and schema.
// RC-3A-2 adds build-time validation and an unconnected static public projection.

export const APPROVED_COSMIC_BREATH_SOURCE_ROLES = Object.freeze([
  "Protected structural authority — exact existing scope",
  "Protected companion authority — exact existing scope",
  "Manifest or integrity record",
  "Implementation or validation record",
  "Public explanatory projection",
  "Primary/original provenance record",
  "Supporting source",
  "Contextual source",
  "Original empirical source",
  "Local bibliography or index record",
  "Historical-record candidate — disposition still required",
  "Different-byte variant — separate disposition required",
  "Companion-format record — separate disposition required",
  "Destination record only",
  "No substantive source role",
  "Withheld from public source presentation",
  "Deferred — additional evidence required",
  "Deferred — owner scope decision required"
] as const);

export type ApprovedCosmicBreathSourceRole =
  (typeof APPROVED_COSMIC_BREATH_SOURCE_ROLES)[number];

export const COSMIC_BREATH_PROVENANCE_ENCODING_STATES = Object.freeze([
  "Public source record eligible",
  "Public projection record eligible",
  "Public empirical record eligible",
  "Public bibliography/index record eligible",
  "Public destination record eligible",
  "Private governance or implementation record only",
  "Withheld from public presentation",
  "Deferred — additional evidence required",
  "Not encoded in the Cosmic Breath provenance experience",
  "Existing deployed action — migration decision required"
] as const);

export type CosmicBreathProvenanceEncodingState =
  (typeof COSMIC_BREATH_PROVENANCE_ENCODING_STATES)[number];

export const COSMIC_BREATH_SOURCE_TYPES = Object.freeze([
  'local',
  'external',
  'projection',
  'destination',
] as const);

export type CosmicBreathSourceType =
  (typeof COSMIC_BREATH_SOURCE_TYPES)[number];

export const COSMIC_BREATH_ACCESS_STATES = Object.freeze([
  'reviewed',
  'access-limited',
  'unavailable',
] as const);

export type CosmicBreathAccessState =
  (typeof COSMIC_BREATH_ACCESS_STATES)[number];

export const COSMIC_BREATH_EPISTEMIC_LABELS = Object.freeze([
  'CU',
  'external',
  'projection',
] as const);

export type CosmicBreathEpistemicLabel =
  (typeof COSMIC_BREATH_EPISTEMIC_LABELS)[number];

declare const validatedValue: unique symbol;

export type StableSourceRecordId = string & {
  readonly [validatedValue]: 'stable-source-record-id';
};
export type NonEmptyString = string & {
  readonly [validatedValue]: 'non-empty-string';
};
export type IsoDate = string & {
  readonly [validatedValue]: 'iso-date';
};
export type Sha256 = string & {
  readonly [validatedValue]: 'sha256';
};
export type ImmutableReference = string & {
  readonly [validatedValue]: 'immutable-reference';
};

export interface InternalSourceDestination {
  readonly kind: 'internal';
  readonly path: string;
  readonly label: string;
}

export interface ExternalSourceDestination {
  readonly kind: 'external';
  readonly href: string;
  readonly label: string;
  readonly externalLabel: string;
}

export type ApprovedSourceDestination =
  | InternalSourceDestination
  | ExternalSourceDestination;

export interface PublicCosmicBreathSourceRecord {
  readonly stableSourceRecordId: StableSourceRecordId;
  readonly publicTitle: NonEmptyString;
  readonly shortTitle?: NonEmptyString;
  readonly approvedSourceRole: ApprovedCosmicBreathSourceRole;
  readonly approvedScope: NonEmptyString | readonly NonEmptyString[];
  readonly explicitNonScope: NonEmptyString | readonly NonEmptyString[];
  readonly sourceType: CosmicBreathSourceType;
  readonly filenameOrExternalIdentity?: NonEmptyString;
  readonly verifiedVersion?: NonEmptyString;
  readonly verifiedDate?: IsoDate;
  readonly immutableRefOrStableIdentifier?: ImmutableReference;
  readonly sha256?: Sha256;
  readonly currentnessStatement?: NonEmptyString;
  readonly protectedAuthorityRelationship?: NonEmptyString;
  readonly publicProjectionRelationship?: NonEmptyString;
  readonly originalExternalAuthority?: NonEmptyString;
  readonly limitations: NonEmptyString | readonly NonEmptyString[];
  readonly accessState: CosmicBreathAccessState;
  readonly reviewDate?: IsoDate;
  readonly approvedDestination?: ApprovedSourceDestination;
  readonly historicalOrVariantNotice?: NonEmptyString;
  readonly licensingNotice?: NonEmptyString;
  readonly epistemicLabel: CosmicBreathEpistemicLabel;
}

export const PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES = Object.freeze([
  "stableSourceRecordId",
  "publicTitle",
  "shortTitle",
  "approvedSourceRole",
  "approvedScope",
  "explicitNonScope",
  "sourceType",
  "filenameOrExternalIdentity",
  "verifiedVersion",
  "verifiedDate",
  "immutableRefOrStableIdentifier",
  "sha256",
  "currentnessStatement",
  "protectedAuthorityRelationship",
  "publicProjectionRelationship",
  "originalExternalAuthority",
  "limitations",
  "accessState",
  "reviewDate",
  "approvedDestination",
  "historicalOrVariantNotice",
  "licensingNotice",
  "epistemicLabel"
] as const satisfies readonly (keyof PublicCosmicBreathSourceRecord)[]);

export type PublicCosmicBreathSourceRecordField =
  (typeof PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES)[number];

export const COSMIC_BREATH_SCHEMA_REQUIREMENTS = Object.freeze([
  "Required",
  "Required for public records",
  "Optional",
  "Conditionally required",
  "Optional/conditional",
  "Required when unresolved or historical",
  "Required for CB-001/004-related records",
  "Required for public empirical records",
  "Required when applicable"
] as const);

export type CosmicBreathSchemaRequirement =
  (typeof COSMIC_BREATH_SCHEMA_REQUIREMENTS)[number];

export interface PublicSourceRecordSchemaFieldDefinition {
  readonly type: string;
  readonly requirement: CosmicBreathSchemaRequirement;
  readonly sourceOfTruth: string;
  readonly qualification: string;
  readonly visibility: 'Public';
  readonly validation: string;
  readonly renderingEffect: string;
  readonly testRequirement: string;
}

export const PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA = Object.freeze({
  "stableSourceRecordId": {
    "type": "lowercase kebab-case string",
    "requirement": "Required",
    "sourceOfTruth": "Approved RC-3 public authority",
    "qualification": "Unique and stable",
    "visibility": "Public",
    "validation": "ID pattern and uniqueness",
    "renderingEffect": "Fragment/card identity",
    "testRequirement": "Schema and fragment test"
  },
  "publicTitle": {
    "type": "string",
    "requirement": "Required for public records",
    "sourceOfTruth": "Owner-approved public content",
    "qualification": "Neutral; no claim expansion",
    "visibility": "Public",
    "validation": "Non-empty",
    "renderingEffect": "Visible heading",
    "testRequirement": "Rendering test"
  },
  "shortTitle": {
    "type": "string",
    "requirement": "Optional",
    "sourceOfTruth": "Owner-approved public content",
    "qualification": "Must remain unambiguous",
    "visibility": "Public",
    "validation": "Non-empty if present",
    "renderingEffect": "Compact label",
    "testRequirement": "Rendering test"
  },
  "approvedSourceRole": {
    "type": "authorized-role enum",
    "requirement": "Required",
    "sourceOfTruth": "RC-2 v0.3",
    "qualification": "Exact governed role",
    "visibility": "Public",
    "validation": "Vocabulary allowlist",
    "renderingEffect": "Role label",
    "testRequirement": "Role test"
  },
  "approvedScope": {
    "type": "string or string[]",
    "requirement": "Required",
    "sourceOfTruth": "RC-2 decision/matrix",
    "qualification": "Bounded scope only",
    "visibility": "Public",
    "validation": "Non-empty",
    "renderingEffect": "Scope disclosure",
    "testRequirement": "Scope test"
  },
  "explicitNonScope": {
    "type": "string or string[]",
    "requirement": "Required",
    "sourceOfTruth": "RC-2 decision/matrix",
    "qualification": "Must prevent whole-node inference",
    "visibility": "Public",
    "validation": "Non-empty",
    "renderingEffect": "Non-scope disclosure",
    "testRequirement": "Non-scope test"
  },
  "sourceType": {
    "type": "enum",
    "requirement": "Required",
    "sourceOfTruth": "RC-2 role plus architecture",
    "qualification": "Local/external/projection/destination",
    "visibility": "Public",
    "validation": "Allowed enum",
    "renderingEffect": "Type label",
    "testRequirement": "Schema test"
  },
  "filenameOrExternalIdentity": {
    "type": "string",
    "requirement": "Optional",
    "sourceOfTruth": "Verified repository-relative or external identity",
    "qualification": "No private absolute path",
    "visibility": "Public",
    "validation": "Reject absolute filesystem paths",
    "renderingEffect": "Identity display",
    "testRequirement": "Privacy test"
  },
  "verifiedVersion": {
    "type": "string",
    "requirement": "Optional",
    "sourceOfTruth": "Directly verified record metadata",
    "qualification": "Omit if unresolved",
    "visibility": "Public",
    "validation": "No placeholder",
    "renderingEffect": "Version row",
    "testRequirement": "Omission test"
  },
  "verifiedDate": {
    "type": "ISO date",
    "requirement": "Optional",
    "sourceOfTruth": "Directly verified record metadata",
    "qualification": "Omit if unresolved",
    "visibility": "Public",
    "validation": "ISO validation",
    "renderingEffect": "Date row",
    "testRequirement": "Date test"
  },
  "immutableRefOrStableIdentifier": {
    "type": "string",
    "requirement": "Conditionally required",
    "sourceOfTruth": "Digest/commit/tag/DOI/institutional ID decision",
    "qualification": "Required for public source/empirical records",
    "visibility": "Public",
    "validation": "Pattern by identity kind",
    "renderingEffect": "Stable link/ID",
    "testRequirement": "Identity test"
  },
  "sha256": {
    "type": "64-char lowercase hex",
    "requirement": "Optional/conditional",
    "sourceOfTruth": "Verified digest record",
    "qualification": "Digest verifies bytes, not claims",
    "visibility": "Public",
    "validation": "SHA-256 pattern",
    "renderingEffect": "Integrity row",
    "testRequirement": "Digest test"
  },
  "currentnessStatement": {
    "type": "string",
    "requirement": "Required when unresolved or historical",
    "sourceOfTruth": "RC-2 lifecycle disposition",
    "qualification": "No invented currentness",
    "visibility": "Public",
    "validation": "Required by state",
    "renderingEffect": "Status notice",
    "testRequirement": "Lifecycle test"
  },
  "protectedAuthorityRelationship": {
    "type": "string",
    "requirement": "Required for CB-001/004-related records",
    "sourceOfTruth": "RC-2 authority decisions",
    "qualification": "Preserve separation",
    "visibility": "Public",
    "validation": "Controlled requirement",
    "renderingEffect": "Authority relation",
    "testRequirement": "Separation test"
  },
  "publicProjectionRelationship": {
    "type": "string",
    "requirement": "Optional",
    "sourceOfTruth": "RC-2 projection role",
    "qualification": "Projection is not authority",
    "visibility": "Public",
    "validation": "No governance fields",
    "renderingEffect": "Projection note",
    "testRequirement": "Projection test"
  },
  "originalExternalAuthority": {
    "type": "string",
    "requirement": "Required for public empirical records",
    "sourceOfTruth": "Reviewed original source",
    "qualification": "Does not validate CU",
    "visibility": "Public",
    "validation": "Required by empirical state",
    "renderingEffect": "Authority disclosure",
    "testRequirement": "Empirical test"
  },
  "limitations": {
    "type": "string or string[]",
    "requirement": "Required",
    "sourceOfTruth": "RC-2 qualification/gap",
    "qualification": "No hidden blockers in public copy",
    "visibility": "Public",
    "validation": "Non-empty",
    "renderingEffect": "Limitations list",
    "testRequirement": "Limitation test"
  },
  "accessState": {
    "type": "enum",
    "requirement": "Required",
    "sourceOfTruth": "RC-1 access review",
    "qualification": "Reviewed/access-limited/unavailable",
    "visibility": "Public",
    "validation": "Allowed enum",
    "renderingEffect": "Access badge/notice",
    "testRequirement": "Access-state test"
  },
  "reviewDate": {
    "type": "ISO date",
    "requirement": "Optional",
    "sourceOfTruth": "Verified review record",
    "qualification": "Omit if unavailable",
    "visibility": "Public",
    "validation": "ISO validation",
    "renderingEffect": "Review row",
    "testRequirement": "Omission test"
  },
  "approvedDestination": {
    "type": "internal/external destination object",
    "requirement": "Optional/conditional",
    "sourceOfTruth": "RC-3 destination decision",
    "qualification": "Base-safe or HTTPS/disclosed",
    "visibility": "Public",
    "validation": "Destination validator",
    "renderingEffect": "Action/link",
    "testRequirement": "Link test"
  },
  "historicalOrVariantNotice": {
    "type": "string",
    "requirement": "Required when applicable",
    "sourceOfTruth": "RC-2 lifecycle/variant treatment",
    "qualification": "No inferred succession",
    "visibility": "Public",
    "validation": "State-conditioned",
    "renderingEffect": "Notice",
    "testRequirement": "Variant test"
  },
  "licensingNotice": {
    "type": "string",
    "requirement": "Required when applicable",
    "sourceOfTruth": "Record-level licensing decision",
    "qualification": "No legal conclusion",
    "visibility": "Public",
    "validation": "State-conditioned",
    "renderingEffect": "Notice/download suppression",
    "testRequirement": "License test"
  },
  "epistemicLabel": {
    "type": "controlled string",
    "requirement": "Required",
    "sourceOfTruth": "RC-2 role/classification boundary",
    "qualification": "Distinguish CU/external/projection",
    "visibility": "Public",
    "validation": "Allowlist",
    "renderingEffect": "Epistemic label",
    "testRequirement": "Boundary test"
  }
} as const satisfies Readonly<
  Record<
    PublicCosmicBreathSourceRecordField,
    PublicSourceRecordSchemaFieldDefinition
  >
>);

export interface CosmicBreathProvenanceAuthorityRecord {
  readonly recordId: string;
  readonly stableSourceRecordId: string;
  readonly sourceType: CosmicBreathSourceType;
  readonly sourceIdentity: string;
  readonly independentFinding: string;
  readonly approvedScope: string;
  readonly approvedSourceRole: ApprovedCosmicBreathSourceRole;
  readonly encodingState: CosmicBreathProvenanceEncodingState;
  readonly publicPresentationEligibility: string;
  readonly publicFieldsSource: string;
  readonly qualificationRequirements: string;
  readonly rc2Qualification: string;
  readonly immutableIdentityRequirements: string;
  readonly noticeRequirements: string;
  readonly releaseBlocker: string;
  readonly rc2RemainingGap: string;
  readonly implementationStageEffect: string;
}

export interface CosmicBreathProvenanceAuthority {
  readonly authorityId: 'CU-COSMIC-BREATH-PROVENANCE-1.0';
  readonly version: '1.0';
  readonly selectedArea: 'cosmic-breath';
  readonly stage: 'RC-3A-1';
  readonly publicProjectionStatus: 'not implemented';
  readonly publicCopyStatus: 'not approved';
  readonly classificationTreatment: {
    readonly 'CU Theoretical Proposition': 'carried';
    readonly 'CU Mathematical Model': 'withheld';
    readonly 'Empirical node classification': 'rejected';
  };
  readonly records: readonly CosmicBreathProvenanceAuthorityRecord[];
}

const PRIVATE_AUTHORITY_KEYS = Object.freeze([
  'authorityId',
  'version',
  'selectedArea',
  'stage',
  'publicProjectionStatus',
  'publicCopyStatus',
  'classificationTreatment',
  'records',
] as const);

const PRIVATE_RECORD_KEYS = Object.freeze([
  'recordId',
  'stableSourceRecordId',
  'sourceType',
  'sourceIdentity',
  'independentFinding',
  'approvedScope',
  'approvedSourceRole',
  'encodingState',
  'publicPresentationEligibility',
  'publicFieldsSource',
  'qualificationRequirements',
  'rc2Qualification',
  'immutableIdentityRequirements',
  'noticeRequirements',
  'releaseBlocker',
  'rc2RemainingGap',
  'implementationStageEffect',
] as const satisfies readonly (keyof CosmicBreathProvenanceAuthorityRecord)[]);

export const PUBLIC_COSMIC_BREATH_PROVENANCE_ENCODING_STATES = Object.freeze([
  'Public source record eligible',
  'Public projection record eligible',
  'Public empirical record eligible',
  'Public bibliography/index record eligible',
  'Public destination record eligible',
] as const satisfies readonly CosmicBreathProvenanceEncodingState[]);

const EXPECTED_ENCODING_STATE_TOTALS = Object.freeze({
  'Public source record eligible': 6,
  'Public projection record eligible': 7,
  'Public empirical record eligible': 5,
  'Public bibliography/index record eligible': 1,
  'Public destination record eligible': 1,
  'Private governance or implementation record only': 26,
  'Withheld from public presentation': 13,
  'Deferred — additional evidence required': 18,
  'Not encoded in the Cosmic Breath provenance experience': 2,
  'Existing deployed action — migration decision required': 1,
} satisfies Record<CosmicBreathProvenanceEncodingState, number>);

const PUBLIC_FIELD_NAME_SET = new Set<string>(
  PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES,
);
const SOURCE_ROLE_SET = new Set<string>(APPROVED_COSMIC_BREATH_SOURCE_ROLES);
const ENCODING_STATE_SET = new Set<string>(
  COSMIC_BREATH_PROVENANCE_ENCODING_STATES,
);
const PUBLIC_ENCODING_STATE_SET = new Set<string>(
  PUBLIC_COSMIC_BREATH_PROVENANCE_ENCODING_STATES,
);
const SOURCE_TYPE_SET = new Set<string>(COSMIC_BREATH_SOURCE_TYPES);
const ACCESS_STATE_SET = new Set<string>(COSMIC_BREATH_ACCESS_STATES);
const EPISTEMIC_LABEL_SET = new Set<string>(
  COSMIC_BREATH_EPISTEMIC_LABELS,
);
const PLACEHOLDER_VALUE =
  /^(?:unknown|tbd|pending|unavailable|not provided|unverified|unresolved)$/i;
const ABSOLUTE_PRIVATE_PATH =
  /^(?:\/Users\/|\/home\/|file:\/\/|~\/|[A-Za-z]:[\\/])/i;
const PROHIBITED_PUBLIC_AUTHORITY_CLAIM =
  /\b(?:whole[- ]node authority|complete marker authority|empirical authority for TOM placement|final relationship(?:-source)? support)\b/i;

type UnknownObject = Record<string, unknown>;

interface CanonicalRecordTreatment {
  readonly stableSourceRecordId: string;
  readonly sourceType: CosmicBreathSourceType;
  readonly approvedSourceRole: ApprovedCosmicBreathSourceRole;
  readonly encodingState: CosmicBreathProvenanceEncodingState;
}

const CANONICAL_RECORD_TREATMENTS = new Map<string, CanonicalRecordTreatment>(
  (rawAuthority.records as readonly CosmicBreathProvenanceAuthorityRecord[]).map(
    (record) => [
      record.recordId,
      {
        stableSourceRecordId: record.stableSourceRecordId,
        sourceType: record.sourceType,
        approvedSourceRole: record.approvedSourceRole,
        encodingState: record.encodingState,
      },
    ],
  ),
);

export class CosmicBreathProvenanceValidationError extends Error {
  readonly recordId: string;
  readonly field: string;

  constructor(recordId: string, field: string, reason: string) {
    super(`${recordId}.${field}: ${reason}`);
    this.name = 'CosmicBreathProvenanceValidationError';
    this.recordId = recordId;
    this.field = field;
  }
}

function validationError(
  recordId: string,
  field: string,
  reason: string,
): never {
  throw new CosmicBreathProvenanceValidationError(recordId, field, reason);
}

function isObject(value: unknown): value is UnknownObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function assertObject(
  value: unknown,
  recordId: string,
  field: string,
): asserts value is UnknownObject {
  if (!isObject(value)) validationError(recordId, field, 'must be an object');
}

function assertExactKeys(
  value: UnknownObject,
  allowedKeys: readonly string[],
  recordId: string,
): void {
  const allowed = new Set(allowedKeys);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) validationError(recordId, key, 'unexpected field');
  }
  for (const key of allowedKeys) {
    if (!(key in value)) validationError(recordId, key, 'required field is missing');
  }
}

function assertNonEmptyString(
  value: unknown,
  recordId: string,
  field: string,
): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    validationError(recordId, field, 'must be a non-empty string');
  }
}

function assertNoPlaceholder(
  value: string,
  recordId: string,
  field: string,
): void {
  if (PLACEHOLDER_VALUE.test(value.trim())) {
    validationError(recordId, field, 'placeholder metadata is prohibited');
  }
}

function assertNoAbsolutePrivatePath(
  value: string,
  recordId: string,
  field: string,
): void {
  if (ABSOLUTE_PRIVATE_PATH.test(value.trim())) {
    validationError(recordId, field, 'absolute private paths are prohibited');
  }
}

function assertStringOrStringArray(
  value: unknown,
  recordId: string,
  field: string,
): asserts value is string | readonly string[] {
  if (typeof value === 'string') {
    assertNonEmptyString(value, recordId, field);
    assertNoPlaceholder(value, recordId, field);
    return;
  }
  if (!Array.isArray(value) || value.length === 0) {
    validationError(recordId, field, 'must be a non-empty string or string array');
  }
  for (const item of value) {
    assertNonEmptyString(item, recordId, field);
    assertNoPlaceholder(item, recordId, field);
  }
}

function assertOptionalString(
  value: unknown,
  recordId: string,
  field: string,
): asserts value is string | undefined {
  if (value === undefined) return;
  assertNonEmptyString(value, recordId, field);
  assertNoPlaceholder(value, recordId, field);
}

function assertIsoDate(value: unknown, recordId: string, field: string): void {
  assertNonEmptyString(value, recordId, field);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(`${value}T00:00:00Z`))) {
    validationError(recordId, field, 'must be a valid ISO date');
  }
}

function assertApprovedDestination(
  value: unknown,
  recordId: string,
): asserts value is ApprovedSourceDestination {
  assertObject(value, recordId, 'approvedDestination');
  assertNonEmptyString(value.kind, recordId, 'approvedDestination.kind');
  if (value.kind === 'internal') {
    assertExactKeys(value, ['kind', 'path', 'label'], recordId);
    assertNonEmptyString(value.path, recordId, 'approvedDestination.path');
    assertNonEmptyString(value.label, recordId, 'approvedDestination.label');
    if (!value.path.startsWith('/')) {
      validationError(recordId, 'approvedDestination.path', 'must be an internal root-relative path');
    }
    return;
  }
  if (value.kind === 'external') {
    assertExactKeys(
      value,
      ['kind', 'href', 'label', 'externalLabel'],
      recordId,
    );
    assertNonEmptyString(value.href, recordId, 'approvedDestination.href');
    assertNonEmptyString(value.label, recordId, 'approvedDestination.label');
    assertNonEmptyString(
      value.externalLabel,
      recordId,
      'approvedDestination.externalLabel',
    );
    if (!value.href.startsWith('https://')) {
      validationError(recordId, 'approvedDestination.href', 'must use HTTPS');
    }
    if (/github\.com\/[^/]+\/[^/]+\/(?:blob|tree)\/(?:main|website)(?:\/|$)/i.test(value.href)) {
      validationError(
        recordId,
        'approvedDestination.href',
        'a mutable branch URL cannot serve as sole authority',
      );
    }
    return;
  }
  validationError(recordId, 'approvedDestination.kind', 'unknown destination kind');
}

function validatePrivateRecord(
  input: unknown,
  index: number,
): CosmicBreathProvenanceAuthorityRecord {
  const fallbackId = `records[${index}]`;
  assertObject(input, fallbackId, 'record');
  const recordId =
    typeof input.recordId === 'string' && input.recordId.length > 0
      ? input.recordId
      : fallbackId;
  assertExactKeys(input, PRIVATE_RECORD_KEYS, recordId);

  for (const field of PRIVATE_RECORD_KEYS) {
    assertNonEmptyString(input[field], recordId, field);
    assertNoPlaceholder(input[field] as string, recordId, field);
  }

  if (!/^(?:CB-\d{3}|CB-E\d{2}|CB-D0[12])$/.test(input.recordId as string)) {
    validationError(recordId, 'recordId', 'unknown stable record ID format');
  }
  if (
    !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(input.stableSourceRecordId as string)
  ) {
    validationError(
      recordId,
      'stableSourceRecordId',
      'must use lowercase kebab-case',
    );
  }
  if (
    input.stableSourceRecordId !==
    (input.recordId as string).toLowerCase()
  ) {
    validationError(
      recordId,
      'stableSourceRecordId',
      'must correspond to recordId',
    );
  }
  if (!SOURCE_ROLE_SET.has(input.approvedSourceRole as string)) {
    validationError(recordId, 'approvedSourceRole', 'unknown source role');
  }
  if (!ENCODING_STATE_SET.has(input.encodingState as string)) {
    validationError(recordId, 'encodingState', 'unknown encoding state');
  }
  if (!SOURCE_TYPE_SET.has(input.sourceType as string)) {
    validationError(recordId, 'sourceType', 'unknown source type');
  }
  assertNoAbsolutePrivatePath(
    input.sourceIdentity as string,
    recordId,
    'sourceIdentity',
  );
  assertNoAbsolutePrivatePath(
    input.publicFieldsSource as string,
    recordId,
    'publicFieldsSource',
  );

  const canonical = CANONICAL_RECORD_TREATMENTS.get(input.recordId as string);
  if (!canonical) validationError(recordId, 'recordId', 'unknown governed record');
  for (const field of [
    'stableSourceRecordId',
    'sourceType',
    'approvedSourceRole',
    'encodingState',
  ] as const) {
    if (input[field] !== canonical[field]) {
      validationError(recordId, field, 'conflicts with governed record treatment');
    }
  }

  return input as unknown as CosmicBreathProvenanceAuthorityRecord;
}

function assertExactEncodingStateTotals(
  records: readonly CosmicBreathProvenanceAuthorityRecord[],
): void {
  const totals = Object.fromEntries(
    COSMIC_BREATH_PROVENANCE_ENCODING_STATES.map((state) => [state, 0]),
  ) as Record<CosmicBreathProvenanceEncodingState, number>;
  for (const record of records) totals[record.encodingState] += 1;
  for (const state of COSMIC_BREATH_PROVENANCE_ENCODING_STATES) {
    if (totals[state] !== EXPECTED_ENCODING_STATE_TOTALS[state]) {
      validationError('authority', 'records', `incorrect total for ${state}`);
    }
  }
}

function assertRecordSpecificTreatments(
  records: readonly CosmicBreathProvenanceAuthorityRecord[],
): void {
  const byId = new Map(records.map((record) => [record.recordId, record]));
  const record = (recordId: string) => {
    const result = byId.get(recordId);
    if (!result) validationError(recordId, 'recordId', 'governed record is missing');
    return result;
  };

  if (record('CB-001').recordId === record('CB-003').recordId) {
    validationError('CB-003', 'recordId', 'must remain separate from CB-001');
  }
  for (const recordId of ['CB-004', 'CB-005', 'CB-058', 'CB-059']) {
    if (PUBLIC_ENCODING_STATE_SET.has(record(recordId).encodingState)) {
      validationError(recordId, 'encodingState', 'private record cannot be publicly eligible');
    }
  }
  if (!/not whole-node|not whole node/i.test(record('CB-001').qualificationRequirements)) {
    validationError('CB-001', 'qualificationRequirements', 'whole-node non-scope is required');
  }
  if (!/lifecycle|formula|missing/i.test(record('CB-006').releaseBlocker)) {
    validationError('CB-006', 'releaseBlocker', 'governed release blockers are required');
  }
  for (const recordId of ['CB-E05', 'CB-E06', 'CB-E09', 'CB-E13', 'CB-E19']) {
    const empirical = record(recordId);
    if (
      empirical.encodingState !== 'Public empirical record eligible' ||
      empirical.approvedSourceRole !== 'Original empirical source' ||
      !/does not validate CU/i.test(empirical.noticeRequirements)
    ) {
      validationError(recordId, 'encodingState', 'bounded empirical treatment changed');
    }
  }
  const deferredEmpiricalCount = records.filter(
    (candidate) =>
      /^CB-E\d{2}$/.test(candidate.recordId) &&
      candidate.encodingState === 'Deferred — additional evidence required',
  ).length;
  if (deferredEmpiricalCount !== 14) {
    validationError('authority', 'records', 'fourteen empirical records must remain deferred');
  }
  if (
    record('CB-024').encodingState !==
      'Public bibliography/index record eligible' ||
    record('CB-024').approvedSourceRole !==
      'Local bibliography or index record'
  ) {
    validationError('CB-024', 'encodingState', 'bibliography/index treatment changed');
  }
  if (
    record('CB-D01').encodingState !==
    'Existing deployed action — migration decision required'
  ) {
    validationError('CB-D01', 'encodingState', 'source action must remain migration-pending');
  }
}

export function parseCosmicBreathProvenanceAuthority(
  input: unknown,
): CosmicBreathProvenanceAuthority {
  assertObject(input, 'authority', 'authority');
  assertExactKeys(input, PRIVATE_AUTHORITY_KEYS, 'authority');
  if (input.authorityId !== 'CU-COSMIC-BREATH-PROVENANCE-1.0') {
    validationError('authority', 'authorityId', 'unknown authority identity');
  }
  if (input.version !== '1.0') {
    validationError('authority', 'version', 'unknown authority version');
  }
  if (input.selectedArea !== 'cosmic-breath') {
    validationError('authority', 'selectedArea', 'unexpected selected area');
  }
  if (input.stage !== 'RC-3A-1') {
    validationError('authority', 'stage', 'unexpected private authority stage');
  }
  if (input.publicProjectionStatus !== 'not implemented') {
    validationError(
      'authority',
      'publicProjectionStatus',
      'private authority status must remain unchanged',
    );
  }
  if (input.publicCopyStatus !== 'not approved') {
    validationError(
      'authority',
      'publicCopyStatus',
      'public copy is not approved',
    );
  }
  assertObject(
    input.classificationTreatment,
    'authority',
    'classificationTreatment',
  );
  assertExactKeys(
    input.classificationTreatment,
    [
      'CU Theoretical Proposition',
      'CU Mathematical Model',
      'Empirical node classification',
    ],
    'classificationTreatment',
  );
  if (
    input.classificationTreatment['CU Theoretical Proposition'] !== 'carried'
  ) {
    validationError(
      'authority',
      'classificationTreatment.CU Theoretical Proposition',
      'governed classification changed',
    );
  }
  if (input.classificationTreatment['CU Mathematical Model'] !== 'withheld') {
    validationError(
      'authority',
      'classificationTreatment.CU Mathematical Model',
      'must remain withheld',
    );
  }
  if (
    input.classificationTreatment['Empirical node classification'] !==
    'rejected'
  ) {
    validationError(
      'authority',
      'classificationTreatment.Empirical node classification',
      'must remain rejected',
    );
  }
  if (!Array.isArray(input.records)) {
    validationError('authority', 'records', 'must be an array');
  }
  if (input.records.length !== 80) {
    validationError('authority', 'records', 'must contain exactly 80 records');
  }

  const records = input.records.map(validatePrivateRecord);
  const ids = new Set<string>();
  for (const record of records) {
    if (ids.has(record.recordId)) {
      validationError(record.recordId, 'recordId', 'duplicate stable record ID');
    }
    ids.add(record.recordId);
  }
  if (records.filter((record) => /^CB-\d{3}$/.test(record.recordId)).length !== 59) {
    validationError('authority', 'records', 'must contain exactly 59 local records');
  }
  if (records.filter((record) => /^CB-E\d{2}$/.test(record.recordId)).length !== 19) {
    validationError('authority', 'records', 'must contain exactly 19 empirical records');
  }
  if (records.filter((record) => /^CB-D0[12]$/.test(record.recordId)).length !== 2) {
    validationError('authority', 'records', 'must contain exactly two destination/action records');
  }
  assertExactEncodingStateTotals(records);
  assertRecordSpecificTreatments(records);
  return input as unknown as CosmicBreathProvenanceAuthority;
}

export function isCosmicBreathPublicEncodingState(
  state: CosmicBreathProvenanceEncodingState,
): boolean {
  return PUBLIC_ENCODING_STATE_SET.has(state);
}

export function validatePublicCosmicBreathSourceRecord(
  input: unknown,
): PublicCosmicBreathSourceRecord {
  assertObject(input, 'publicRecord', 'record');
  const recordId =
    typeof input.stableSourceRecordId === 'string'
      ? input.stableSourceRecordId
      : 'publicRecord';
  for (const key of Object.keys(input)) {
    if (!PUBLIC_FIELD_NAME_SET.has(key)) {
      validationError(recordId, key, 'field is not publicly allowlisted');
    }
  }

  for (const field of [
    'stableSourceRecordId',
    'publicTitle',
    'approvedSourceRole',
    'approvedScope',
    'explicitNonScope',
    'sourceType',
    'limitations',
    'accessState',
    'epistemicLabel',
  ] as const) {
    if (!(field in input)) validationError(recordId, field, 'required field is missing');
  }
  assertNonEmptyString(
    input.stableSourceRecordId,
    recordId,
    'stableSourceRecordId',
  );
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(input.stableSourceRecordId)) {
    validationError(recordId, 'stableSourceRecordId', 'must use lowercase kebab-case');
  }
  assertNonEmptyString(input.publicTitle, recordId, 'publicTitle');
  assertNoPlaceholder(input.publicTitle, recordId, 'publicTitle');
  assertOptionalString(input.shortTitle, recordId, 'shortTitle');
  assertNonEmptyString(input.approvedSourceRole, recordId, 'approvedSourceRole');
  if (!SOURCE_ROLE_SET.has(input.approvedSourceRole)) {
    validationError(recordId, 'approvedSourceRole', 'unknown source role');
  }
  assertStringOrStringArray(input.approvedScope, recordId, 'approvedScope');
  assertStringOrStringArray(input.explicitNonScope, recordId, 'explicitNonScope');
  assertNonEmptyString(input.sourceType, recordId, 'sourceType');
  if (!SOURCE_TYPE_SET.has(input.sourceType)) {
    validationError(recordId, 'sourceType', 'unknown source type');
  }
  assertOptionalString(
    input.filenameOrExternalIdentity,
    recordId,
    'filenameOrExternalIdentity',
  );
  if (input.filenameOrExternalIdentity !== undefined) {
    assertNoAbsolutePrivatePath(
      input.filenameOrExternalIdentity,
      recordId,
      'filenameOrExternalIdentity',
    );
  }
  assertOptionalString(input.verifiedVersion, recordId, 'verifiedVersion');
  if (input.verifiedDate !== undefined) {
    assertIsoDate(input.verifiedDate, recordId, 'verifiedDate');
  }
  assertOptionalString(
    input.immutableRefOrStableIdentifier,
    recordId,
    'immutableRefOrStableIdentifier',
  );
  if (input.sha256 !== undefined) {
    assertNonEmptyString(input.sha256, recordId, 'sha256');
    if (!/^[a-f0-9]{64}$/.test(input.sha256)) {
      validationError(recordId, 'sha256', 'must be 64 lowercase hexadecimal characters');
    }
  }
  assertOptionalString(
    input.currentnessStatement,
    recordId,
    'currentnessStatement',
  );
  assertOptionalString(
    input.protectedAuthorityRelationship,
    recordId,
    'protectedAuthorityRelationship',
  );
  assertOptionalString(
    input.publicProjectionRelationship,
    recordId,
    'publicProjectionRelationship',
  );
  assertOptionalString(
    input.originalExternalAuthority,
    recordId,
    'originalExternalAuthority',
  );
  assertStringOrStringArray(input.limitations, recordId, 'limitations');
  assertNonEmptyString(input.accessState, recordId, 'accessState');
  if (!ACCESS_STATE_SET.has(input.accessState)) {
    validationError(recordId, 'accessState', 'unknown access state');
  }
  if (input.reviewDate !== undefined) {
    assertIsoDate(input.reviewDate, recordId, 'reviewDate');
  }
  if (input.approvedDestination !== undefined) {
    assertApprovedDestination(input.approvedDestination, recordId);
  }
  assertOptionalString(
    input.historicalOrVariantNotice,
    recordId,
    'historicalOrVariantNotice',
  );
  assertOptionalString(input.licensingNotice, recordId, 'licensingNotice');
  assertNonEmptyString(input.epistemicLabel, recordId, 'epistemicLabel');
  if (!EPISTEMIC_LABEL_SET.has(input.epistemicLabel)) {
    validationError(recordId, 'epistemicLabel', 'unknown epistemic label');
  }
  for (const [field, value] of Object.entries(input)) {
    const strings =
      typeof value === 'string'
        ? [value]
        : Array.isArray(value)
          ? value.filter((item): item is string => typeof item === 'string')
          : [];
    for (const text of strings) {
      const isExplicitlyNegated =
        /\b(?:not|no|does not|must not|is not|cannot)\b/i.test(text);
      if (
        PROHIBITED_PUBLIC_AUTHORITY_CLAIM.test(text) &&
        !isExplicitlyNegated
      ) {
        validationError(recordId, field, 'prohibited authority claim');
      }
    }
  }

  if (
    input.stableSourceRecordId === 'cb-001' &&
    input.protectedAuthorityRelationship === undefined
  ) {
    validationError(
      recordId,
      'protectedAuthorityRelationship',
      'CB-001 separation disclosure is required',
    );
  }
  if (input.stableSourceRecordId === 'cb-001') {
    if (
      input.approvedSourceRole !==
      'Protected structural authority — exact existing scope'
    ) {
      validationError(recordId, 'approvedSourceRole', 'CB-001 role escalation');
    }
    const nonScope = Array.isArray(input.explicitNonScope)
      ? input.explicitNonScope.join(' ')
      : input.explicitNonScope;
    if (!/not whole-node|not whole node/i.test(nonScope)) {
      validationError(recordId, 'explicitNonScope', 'CB-001 whole-node non-scope is required');
    }
  }
  if (input.stableSourceRecordId === 'cb-003' && input.approvedSourceRole !== 'Manifest or integrity record') {
    validationError(recordId, 'approvedSourceRole', 'CB-003 must remain a distinct integrity record');
  }
  if (input.stableSourceRecordId === 'cb-004' || input.stableSourceRecordId === 'cb-005') {
    validationError(recordId, 'stableSourceRecordId', 'private record cannot enter the public projection');
  }
  if (input.stableSourceRecordId === 'cb-006') {
    if (input.approvedSourceRole !== 'Primary/original provenance record') {
      validationError(recordId, 'approvedSourceRole', 'CB-006 role escalation');
    }
    if (
      input.currentnessStatement === undefined ||
      !/withheld|blocked/i.test(input.currentnessStatement)
    ) {
      validationError(recordId, 'currentnessStatement', 'CB-006 release-blocked status is required');
    }
    if (input.approvedDestination !== undefined) {
      validationError(recordId, 'approvedDestination', 'CB-006 destination remains unresolved');
    }
  }
  if (input.stableSourceRecordId === 'cb-058' || input.stableSourceRecordId === 'cb-059') {
    validationError(recordId, 'stableSourceRecordId', 'private contextual record cannot enter the public projection');
  }
  if (
    input.approvedSourceRole === 'Original empirical source' &&
    input.originalExternalAuthority === undefined
  ) {
    validationError(
      recordId,
      'originalExternalAuthority',
      'public empirical records require original authority disclosure',
    );
  }
  if (
    input.approvedDestination !== undefined &&
    input.immutableRefOrStableIdentifier === undefined
  ) {
    validationError(
      recordId,
      'immutableRefOrStableIdentifier',
      'a projected destination requires a stable identifier',
    );
  }
  if (
    input.approvedDestination?.kind === 'external' &&
    input.licensingNotice === undefined
  ) {
    validationError(
      recordId,
      'licensingNotice',
      'an external destination requires licensing treatment',
    );
  }
  if (
    (input.approvedSourceRole ===
      'Historical-record candidate — disposition still required' ||
      input.approvedSourceRole ===
        'Different-byte variant — separate disposition required' ||
      input.approvedSourceRole ===
        'Companion-format record — separate disposition required') &&
    input.historicalOrVariantNotice === undefined
  ) {
    validationError(
      recordId,
      'historicalOrVariantNotice',
      'historical or variant treatment requires a notice',
    );
  }

  return input as unknown as PublicCosmicBreathSourceRecord;
}

const LINK_AND_IDENTITY_ONLY_NOTICE =
  'Link and identity only; no quotation, excerpt, reproduction, download, redistribution, or embedding.';
const CB_001_AUTHORITY_RELATIONSHIP =
  'CB-001 remains distinct from CB-003 and does not govern CB-004.';

function epistemicLabelFor(
  record: CosmicBreathProvenanceAuthorityRecord,
): CosmicBreathEpistemicLabel {
  if (record.sourceType === 'external') return 'external';
  if (record.sourceType === 'projection' || record.sourceType === 'destination') {
    return 'projection';
  }
  return 'CU';
}

function projectPublicRecord(
  record: CosmicBreathProvenanceAuthorityRecord,
): PublicCosmicBreathSourceRecord {
  const projected: {
    stableSourceRecordId: string;
    publicTitle: string;
    approvedSourceRole: ApprovedCosmicBreathSourceRole;
    approvedScope: string;
    explicitNonScope: string;
    sourceType: CosmicBreathSourceType;
    filenameOrExternalIdentity: string;
    immutableRefOrStableIdentifier: string;
    currentnessStatement: string;
    protectedAuthorityRelationship?: string;
    publicProjectionRelationship?: string;
    originalExternalAuthority?: string;
    limitations: readonly string[];
    accessState: CosmicBreathAccessState;
    licensingNotice: string;
    epistemicLabel: CosmicBreathEpistemicLabel;
  } = {
    stableSourceRecordId: record.stableSourceRecordId,
    publicTitle: record.sourceIdentity,
    approvedSourceRole: record.approvedSourceRole,
    approvedScope: record.approvedScope,
    explicitNonScope: record.qualificationRequirements,
    sourceType: record.sourceType,
    filenameOrExternalIdentity: record.sourceIdentity,
    immutableRefOrStableIdentifier: record.stableSourceRecordId,
    currentnessStatement: record.publicPresentationEligibility,
    limitations: [
      record.qualificationRequirements,
      record.noticeRequirements,
    ],
    accessState: 'reviewed',
    licensingNotice: LINK_AND_IDENTITY_ONLY_NOTICE,
    epistemicLabel: epistemicLabelFor(record),
  };

  if (record.recordId === 'CB-001') {
    projected.protectedAuthorityRelationship =
      CB_001_AUTHORITY_RELATIONSHIP;
  }
  if (record.sourceType === 'projection') {
    projected.publicProjectionRelationship =
      record.qualificationRequirements;
  }
  if (record.approvedSourceRole === 'Original empirical source') {
    projected.originalExternalAuthority = record.sourceIdentity;
  }

  return validatePublicCosmicBreathSourceRecord(projected);
}

export function buildCosmicBreathPublicSourceRecordProjection(
  input: unknown,
): readonly PublicCosmicBreathSourceRecord[] {
  const authority = parseCosmicBreathProvenanceAuthority(input);
  const projectedRecords: PublicCosmicBreathSourceRecord[] = [];
  for (const record of authority.records) {
    if (!isCosmicBreathPublicEncodingState(record.encodingState)) continue;
    projectedRecords.push(Object.freeze(projectPublicRecord(record)));
  }
  return Object.freeze(projectedRecords);
}

export const COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION =
  buildCosmicBreathPublicSourceRecordProjection(rawAuthority);
