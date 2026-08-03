import { describe, expect, it } from 'vitest';
import rawAuthority from '../../../data/research/CU-COSMIC-BREATH-PROVENANCE-1.0.json';
import {
  APPROVED_COSMIC_BREATH_SOURCE_ROLES,
  COSMIC_BREATH_ACCESS_STATES,
  COSMIC_BREATH_EPISTEMIC_LABELS,
  COSMIC_BREATH_PROVENANCE_ENCODING_STATES,
  COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION,
  COSMIC_BREATH_SOURCE_TYPES,
  PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES,
  PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA,
  buildCosmicBreathPublicSourceRecordProjection,
  parseCosmicBreathProvenanceAuthority,
  validatePublicCosmicBreathSourceRecord,
  type CosmicBreathProvenanceAuthority,
  type PublicCosmicBreathSourceRecord,
} from '../cosmic-breath-provenance';

const authority = rawAuthority as CosmicBreathProvenanceAuthority;
const getRecord = (recordId: string) => {
  const record = authority.records.find((candidate) => candidate.recordId === recordId);
  expect(record, `missing governed record ${recordId}`).toBeDefined();
  return record!;
};

const publicEncodingStates = new Set([
  'Public source record eligible',
  'Public projection record eligible',
  'Public empirical record eligible',
  'Public bibliography/index record eligible',
  'Public destination record eligible',
]);

describe('Cosmic Breath provenance authority — RC-3A-1', () => {
  it('defines the exact conditional 23-field public source-record schema', () => {
    expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES).toHaveLength(23);
    expect(Object.keys(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA)).toEqual(
      PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES,
    );
    expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES).toEqual([
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
]);
    expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA.stableSourceRecordId).toMatchObject({
      type: 'lowercase kebab-case string',
      requirement: 'Required',
      validation: 'ID pattern and uniqueness',
    });
    expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA.approvedSourceRole.type).toBe(
      'authorized-role enum',
    );
    expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA.accessState.type).toBe('enum');
    expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_SCHEMA.epistemicLabel.type).toBe(
      'controlled string',
    );
  });

  it('contains all 80 governed records once with the exact record-family counts', () => {
    expect(authority.authorityId).toBe('CU-COSMIC-BREATH-PROVENANCE-1.0');
    expect(authority.records).toHaveLength(80);
    const ids = authority.records.map((record) => record.recordId);
    expect(new Set(ids).size).toBe(80);
    expect(ids.filter((id) => /^CB-[0-9]{3}$/.test(id))).toHaveLength(59);
    expect(ids.filter((id) => /^CB-E[0-9]{2}$/.test(id))).toHaveLength(19);
    expect(ids.filter((id) => /^CB-D0[12]$/.test(id))).toHaveLength(2);
    for (const record of authority.records) {
      expect(record.stableSourceRecordId).toBe(record.recordId.toLowerCase());
      expect(record.stableSourceRecordId).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    }
  });

  it('preserves the exact closed role and encoding-state vocabularies', () => {
    const roleSet = new Set<string>(APPROVED_COSMIC_BREATH_SOURCE_ROLES);
    const stateSet = new Set<string>(COSMIC_BREATH_PROVENANCE_ENCODING_STATES);
    const sourceTypeSet = new Set<string>(COSMIC_BREATH_SOURCE_TYPES);
    for (const record of authority.records) {
      expect(roleSet.has(record.approvedSourceRole)).toBe(true);
      expect(stateSet.has(record.encodingState)).toBe(true);
      expect(sourceTypeSet.has(record.sourceType)).toBe(true);
    }
    expect(COSMIC_BREATH_PROVENANCE_ENCODING_STATES).toEqual([
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
]);
  });

  it('preserves the exact governed encoding-state totals', () => {
    const counts = Object.fromEntries(
      COSMIC_BREATH_PROVENANCE_ENCODING_STATES.map((state) => [state, 0]),
    ) as Record<string, number>;
    for (const record of authority.records) counts[record.encodingState] += 1;
    expect(counts).toEqual({
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
    });
  });

  it('preserves structural, companion, provenance, and contextual restrictions', () => {
    expect(getRecord('CB-001')).toMatchObject({
      approvedSourceRole: 'Protected structural authority — exact existing scope',
      encodingState: 'Public source record eligible',
    });
    expect(getRecord('CB-001').qualificationRequirements).toMatch(
      /not whole-node|not whole node/i,
    );
    expect(getRecord('CB-003').approvedSourceRole).toBe('Manifest or integrity record');
    expect(getRecord('CB-003').recordId).not.toBe(getRecord('CB-001').recordId);
    for (const id of ['CB-004', 'CB-005']) {
      expect(getRecord(id).encodingState).toBe('Withheld from public presentation');
    }
    expect(getRecord('CB-006')).toMatchObject({
      approvedSourceRole: 'Primary/original provenance record',
      encodingState: 'Public source record eligible',
    });
    expect(getRecord('CB-006').releaseBlocker).toMatch(/lifecycle|formula|missing/i);
    for (const id of ['CB-058', 'CB-059']) {
      expect(getRecord(id)).toMatchObject({
        approvedSourceRole: 'Contextual source',
        encodingState: 'Withheld from public presentation',
      });
      expect(getRecord(id).publicFieldsSource).toMatch(/no browser projection/i);
    }
  });

  it('preserves bounded empirical eligibility, deferrals, and bibliography status', () => {
    const reviewed = ['CB-E05', 'CB-E06', 'CB-E09', 'CB-E13', 'CB-E19'];
    for (const id of reviewed) {
      expect(getRecord(id)).toMatchObject({
        approvedSourceRole: 'Original empirical source',
        encodingState: 'Public empirical record eligible',
        sourceType: 'external',
      });
      expect(getRecord(id).noticeRequirements).toMatch(/does not validate CU/i);
    }
    const empirical = authority.records.filter((record) => /^CB-E[0-9]{2}$/.test(record.recordId));
    expect(
      empirical.filter(
        (record) => record.encodingState === 'Deferred — additional evidence required',
      ),
    ).toHaveLength(14);
    expect(getRecord('CB-024')).toMatchObject({
      approvedSourceRole: 'Local bibliography or index record',
      encodingState: 'Public bibliography/index record eligible',
    });
  });

  it('keeps the deployed source action migration-pending and classifications bounded', () => {
    expect(getRecord('CB-D01')).toMatchObject({
      approvedSourceRole: 'Destination record only',
      encodingState: 'Existing deployed action — migration decision required',
    });
    expect(authority.classificationTreatment).toEqual({
      'CU Theoretical Proposition': 'carried',
      'CU Mathematical Model': 'withheld',
      'Empirical node classification': 'rejected',
    });
  });

  it('omits unverified public metadata and private absolute paths', () => {
    const forbiddenUnverifiedKeys = [
      'publicTitle',
      'shortTitle',
      'verifiedVersion',
      'verifiedDate',
      'immutableRefOrStableIdentifier',
      'sha256',
      'reviewDate',
      'approvedDestination',
      'historicalOrVariantNotice',
      'licensingNotice',
    ];
    for (const record of authority.records) {
      for (const key of forbiddenUnverifiedKeys) {
        expect(record).not.toHaveProperty(key);
      }
      expect(record.sourceIdentity).not.toMatch(/^(?:\/Users\/|file:|~\/)/i);
      if (publicEncodingStates.has(record.encodingState)) {
        expect(record.publicFieldsSource).not.toMatch(/\/Users\/|file:\/\//i);
      }
    }
  });

  it('does not encode marker authority, final relationships, or a browser projection', () => {
    expect(authority.publicProjectionStatus).toBe('not implemented');
    expect(authority.publicCopyStatus).toBe('not approved');
    expect(rawAuthority).not.toHaveProperty('publicRecords');
    expect(rawAuthority).not.toHaveProperty('relationships');
    expect(rawAuthority).not.toHaveProperty('markerAuthority');
  });
});

const cloneAuthority = () =>
  JSON.parse(JSON.stringify(rawAuthority)) as Record<string, any>;
const clonePublicRecord = (recordId = 'cb-001') =>
  JSON.parse(
    JSON.stringify(
      COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION.find(
        (record) => record.stableSourceRecordId === recordId,
      ),
    ),
  ) as Record<string, any>;

describe('Cosmic Breath provenance authority — RC-3A-2 validation', () => {
  it('strictly parses all 80 private records without silently omitting one', () => {
    const parsed = parseCosmicBreathProvenanceAuthority(rawAuthority);
    expect(parsed.records).toHaveLength(80);
    expect(parsed.records.map((record) => record.recordId)).toEqual(
      authority.records.map((record) => record.recordId),
    );

    const invalid = cloneAuthority();
    invalid.records[79].privateTestDetail = 'must be rejected';
    expect(() => parseCosmicBreathProvenanceAuthority(invalid)).toThrow(
      /CB-D02\.privateTestDetail: unexpected field/,
    );
  });

  it('rejects duplicate stable record IDs with record-and-field diagnostics', () => {
    const invalid = cloneAuthority();
    invalid.records[1] = JSON.parse(JSON.stringify(invalid.records[0]));
    expect(() => parseCosmicBreathProvenanceAuthority(invalid)).toThrow(
      /CB-001\.recordId: duplicate stable record ID/,
    );
  });

  it.each([
    ['approvedSourceRole', 'Invented authority role', /CB-001\.approvedSourceRole: unknown source role/],
    ['encodingState', 'Public maybe eligible', /CB-001\.encodingState: unknown encoding state/],
    ['sourceType', 'filesystem', /CB-001\.sourceType: unknown source type/],
  ])('rejects unknown private %s vocabulary values', (field, value, expected) => {
    const invalid = cloneAuthority();
    invalid.records[0][field] = value;
    expect(() => parseCosmicBreathProvenanceAuthority(invalid)).toThrow(expected);
  });

  it('rejects unknown access-state and epistemic-label values', () => {
    const invalidAccess = clonePublicRecord();
    invalidAccess.accessState = 'partly reviewed';
    expect(() => validatePublicCosmicBreathSourceRecord(invalidAccess)).toThrow(
      /cb-001\.accessState: unknown access state/,
    );

    const invalidLabel = clonePublicRecord();
    invalidLabel.epistemicLabel = 'empirical proof';
    expect(() => validatePublicCosmicBreathSourceRecord(invalidLabel)).toThrow(
      /cb-001\.epistemicLabel: unknown epistemic label/,
    );
    expect(COSMIC_BREATH_ACCESS_STATES).toEqual([
      'reviewed',
      'access-limited',
      'unavailable',
    ]);
    expect(COSMIC_BREATH_EPISTEMIC_LABELS).toEqual([
      'CU',
      'external',
      'projection',
    ]);
  });

  it('rejects incorrect private field types and populated prohibited fields', () => {
    const wrongType = cloneAuthority();
    wrongType.records[0].approvedScope = ['not', 'a', 'private', 'string'];
    expect(() => parseCosmicBreathProvenanceAuthority(wrongType)).toThrow(
      /CB-001\.approvedScope: must be a non-empty string/,
    );

    const prohibited = cloneAuthority();
    prohibited.records[0].publicTitle = 'Unapproved public copy';
    expect(() => parseCosmicBreathProvenanceAuthority(prohibited)).toThrow(
      /CB-001\.publicTitle: unexpected field/,
    );
  });

  it('enforces required scope, non-scope, and conditionally required fields', () => {
    const missingScope = clonePublicRecord();
    delete missingScope.approvedScope;
    expect(() => validatePublicCosmicBreathSourceRecord(missingScope)).toThrow(
      /cb-001\.approvedScope: required field is missing/,
    );

    const missingNonScope = clonePublicRecord();
    delete missingNonScope.explicitNonScope;
    expect(() => validatePublicCosmicBreathSourceRecord(missingNonScope)).toThrow(
      /cb-001\.explicitNonScope: required field is missing/,
    );

    const missingRelationship = clonePublicRecord();
    delete missingRelationship.protectedAuthorityRelationship;
    expect(() =>
      validatePublicCosmicBreathSourceRecord(missingRelationship),
    ).toThrow(/cb-001\.protectedAuthorityRelationship/);

    const empirical = clonePublicRecord('cb-e05');
    delete empirical.originalExternalAuthority;
    expect(() => validatePublicCosmicBreathSourceRecord(empirical)).toThrow(
      /cb-e05\.originalExternalAuthority/,
    );
  });

  it('rejects placeholders, unverified metadata, absolute paths, and arbitrary fields', () => {
    const placeholder = clonePublicRecord();
    placeholder.verifiedVersion = 'TBD';
    expect(() => validatePublicCosmicBreathSourceRecord(placeholder)).toThrow(
      /cb-001\.verifiedVersion: placeholder metadata is prohibited/,
    );

    const privatePath = clonePublicRecord();
    privatePath.filenameOrExternalIdentity = '/Users/owner/private/source.pdf';
    expect(() => validatePublicCosmicBreathSourceRecord(privatePath)).toThrow(
      /cb-001\.filenameOrExternalIdentity: absolute private paths are prohibited/,
    );

    const arbitrary = clonePublicRecord();
    arbitrary.privateDeliberation = 'not public';
    expect(() => validatePublicCosmicBreathSourceRecord(arbitrary)).toThrow(
      /cb-001\.privateDeliberation: field is not publicly allowlisted/,
    );
  });

  it('rejects role and encoding-state escalation against every governed record treatment', () => {
    const roleEscalation = cloneAuthority();
    roleEscalation.records[1].approvedSourceRole =
      'Protected structural authority — exact existing scope';
    expect(() => parseCosmicBreathProvenanceAuthority(roleEscalation)).toThrow(
      /CB-002\.approvedSourceRole: conflicts with governed record treatment/,
    );

    const stateEscalation = cloneAuthority();
    stateEscalation.records.find((record: any) => record.recordId === 'CB-004').encodingState =
      'Public source record eligible';
    expect(() => parseCosmicBreathProvenanceAuthority(stateEscalation)).toThrow(
      /CB-004\.encodingState: conflicts with governed record treatment/,
    );
  });

  it('rejects prohibited authority claims and unsafe or unresolved destinations', () => {
    const authorityClaim = clonePublicRecord();
    authorityClaim.approvedScope = 'Complete marker authority';
    expect(() => validatePublicCosmicBreathSourceRecord(authorityClaim)).toThrow(
      /cb-001\.approvedScope: prohibited authority claim/,
    );

    const insecureDestination = clonePublicRecord();
    insecureDestination.approvedDestination = {
      kind: 'external',
      href: 'http://example.com/source',
      label: 'Source',
      externalLabel: 'External',
    };
    expect(() =>
      validatePublicCosmicBreathSourceRecord(insecureDestination),
    ).toThrow(/cb-001\.approvedDestination\.href: must use HTTPS/);

    const mutableBranchDestination = clonePublicRecord();
    mutableBranchDestination.approvedDestination = {
      kind: 'external',
      href: 'https://github.com/example/project/blob/main/source.md',
      label: 'Source',
      externalLabel: 'External',
    };
    expect(() =>
      validatePublicCosmicBreathSourceRecord(mutableBranchDestination),
    ).toThrow(/mutable branch URL cannot serve as sole authority/);
  });

  it('reports only the responsible record and field for validation failures', () => {
    const invalid = cloneAuthority();
    invalid.records.find((record: any) => record.recordId === 'CB-E19').sourceType =
      'unknown';
    try {
      parseCosmicBreathProvenanceAuthority(invalid);
      expect.fail('validation should have failed');
    } catch (error) {
      const message = String(error);
      expect(message).toMatch(/CB-E19\.sourceType/);
      expect(message).not.toContain('releaseBlocker');
      expect(message).not.toContain('owner');
      expect(message).not.toContain('workflow');
    }
  });
});

describe('Cosmic Breath provenance authority — RC-3A-2 projection and privacy', () => {
  const projection = COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION;
  const projectedIds = projection.map((record) => record.stableSourceRecordId);
  const privateFields = [
    'recordId',
    'sourceIdentity',
    'independentFinding',
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
  ];

  it('evaluates eligibility record by record and projects exactly 20 governed records', () => {
    const expectedIds = authority.records
      .filter((record) => publicEncodingStates.has(record.encodingState))
      .map((record) => record.stableSourceRecordId);
    expect(projection).toHaveLength(20);
    expect(projectedIds).toEqual(expectedIds);
    expect(new Set(projectedIds).size).toBe(projection.length);
    for (const record of authority.records) {
      expect(projectedIds.includes(record.stableSourceRecordId)).toBe(
        publicEncodingStates.has(record.encodingState),
      );
    }
  });

  it('uses only the explicit 23-category allowlist and omits unavailable optional metadata', () => {
    const projectedFieldNames = new Set(
      projection.flatMap((record) => Object.keys(record)),
    );
    expect(projectedFieldNames.size).toBe(16);
    for (const field of projectedFieldNames) {
      expect(PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES).toContain(field);
    }
    for (const record of projection) {
      expect(record).not.toHaveProperty('shortTitle');
      expect(record).not.toHaveProperty('verifiedVersion');
      expect(record).not.toHaveProperty('verifiedDate');
      expect(record).not.toHaveProperty('sha256');
      expect(record).not.toHaveProperty('reviewDate');
      expect(record).not.toHaveProperty('approvedDestination');
      expect(record).not.toHaveProperty('historicalOrVariantNotice');
    }
  });

  it('excludes every private authority field and arbitrary carry-forward field', () => {
    for (const record of projection) {
      for (const field of privateFields) expect(record).not.toHaveProperty(field);
      expect(Object.keys(record).every((key) =>
        PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES.includes(
          key as (typeof PUBLIC_COSMIC_BREATH_SOURCE_RECORD_FIELD_NAMES)[number],
        ),
      )).toBe(true);
    }
  });

  it('excludes absolute private paths, governing artifact bodies, and private blocker text', () => {
    const serialized = JSON.stringify(projection);
    expect(serialized).not.toMatch(/\/Users\/|file:\/\/|~\//i);
    expect(serialized).not.toMatch(
      /Part_II_(?:RM0|RC0|RC1|RC2|RC3)|P2-RC-3A|rollback instruction|owner deliberation/i,
    );
    for (const record of authority.records) {
      if (record.releaseBlocker.length > 10) {
        expect(serialized).not.toContain(record.releaseBlocker);
      }
      expect(serialized).not.toContain(record.implementationStageEffect);
    }
  });

  it('keeps CB-001 and CB-003 separate with bounded structural and integrity roles', () => {
    const structural = projection.find((record) => record.stableSourceRecordId === 'cb-001')!;
    const integrity = projection.find((record) => record.stableSourceRecordId === 'cb-003')!;
    expect(structural).not.toBe(integrity);
    expect(structural.approvedSourceRole).toBe(
      'Protected structural authority — exact existing scope',
    );
    expect(structural.approvedScope).toBe('Exact approved structural fields only');
    expect(structural.explicitNonScope).toMatch(/not whole-node/i);
    expect(structural.protectedAuthorityRelationship).toMatch(
      /distinct from CB-003.*does not govern CB-004/i,
    );
    expect(integrity.approvedSourceRole).toBe('Manifest or integrity record');
  });

  it('excludes CB-004, CB-005, CB-058, and CB-059', () => {
    expect(projectedIds).not.toContain('cb-004');
    expect(projectedIds).not.toContain('cb-005');
    expect(projectedIds).not.toContain('cb-058');
    expect(projectedIds).not.toContain('cb-059');
  });

  it('keeps CB-006 qualified, non-sole, destination-free, and release-blocked', () => {
    const record = projection.find((candidate) => candidate.stableSourceRecordId === 'cb-006')!;
    expect(record.approvedSourceRole).toBe('Primary/original provenance record');
    expect(record.explicitNonScope).toMatch(/non-sole authority/i);
    expect(record.currentnessStatement).toMatch(/withheld/i);
    expect(record).not.toHaveProperty('approvedDestination');
    expect(record).not.toHaveProperty('sha256');
    expect(record).not.toHaveProperty('verifiedVersion');
  });

  it('preserves bounded empirical treatment and excludes fourteen deferred records', () => {
    const reviewed = ['cb-e05', 'cb-e06', 'cb-e09', 'cb-e13', 'cb-e19'];
    for (const recordId of reviewed) {
      const record = projection.find(
        (candidate) => candidate.stableSourceRecordId === recordId,
      )!;
      expect(record.approvedSourceRole).toBe('Original empirical source');
      expect(record.sourceType).toBe('external');
      expect(record.epistemicLabel).toBe('external');
      expect(record.originalExternalAuthority).toBe(
        record.filenameOrExternalIdentity,
      );
      expect(
        Array.isArray(record.limitations)
          ? record.limitations.join(' ')
          : record.limitations,
      ).toMatch(/does not validate CU/i);
      expect(record.licensingNotice).toMatch(/link and identity only/i);
    }
    const deferredIds = authority.records
      .filter(
        (record) =>
          /^CB-E\d{2}$/.test(record.recordId) &&
          record.encodingState === 'Deferred — additional evidence required',
      )
      .map((record) => record.stableSourceRecordId);
    expect(deferredIds).toHaveLength(14);
    for (const recordId of deferredIds) expect(projectedIds).not.toContain(recordId);
  });

  it('retains CB-024 as bibliography/index only and excludes the migration-pending action', () => {
    expect(
      projection.find((record) => record.stableSourceRecordId === 'cb-024'),
    ).toMatchObject({
      approvedSourceRole: 'Local bibliography or index record',
      sourceType: 'local',
    });
    expect(projectedIds).not.toContain('cb-d01');
    expect(projectedIds).toContain('cb-d02');
  });

  it('does not produce marker authority, final relationship support, or classification escalation', () => {
    const serialized = JSON.stringify(projection);
    expect(serialized).not.toMatch(/complete marker authority/i);
    expect(serialized).not.toMatch(/empirical authority for TOM placement/i);
    expect(serialized).not.toMatch(/final relationship(?:-source)? support/i);
    expect(serialized).not.toContain('CU Mathematical Model');
    expect(serialized).not.toContain('Empirical node classification');
    expect(authority.classificationTreatment).toEqual({
      'CU Theoretical Proposition': 'carried',
      'CU Mathematical Model': 'withheld',
      'Empirical node classification': 'rejected',
    });
  });

  it('constructs a stable, frozen, deterministic projection in authority order', () => {
    const second = buildCosmicBreathPublicSourceRecordProjection(rawAuthority);
    expect(second).toEqual(projection);
    expect(second).not.toBe(projection);
    expect(Object.isFrozen(projection)).toBe(true);
    expect(projection.every(Object.isFrozen)).toBe(true);
    expect(projectedIds).toEqual(
      authority.records
        .filter((record) => publicEncodingStates.has(record.encodingState))
        .map((record) => record.stableSourceRecordId),
    );
  });

  it('validates every projected record against the public schema boundary', () => {
    for (const record of projection) {
      expect(validatePublicCosmicBreathSourceRecord(record)).toBe(record);
    }
    expect(
      projection.every(
        (record): record is PublicCosmicBreathSourceRecord => record !== undefined,
      ),
    ).toBe(true);
  });
});
