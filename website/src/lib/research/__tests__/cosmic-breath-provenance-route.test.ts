import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION } from '../cosmic-breath-provenance';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const digest = (source: string) =>
  createHash('sha256').update(source).digest('hex');

const routeSource = readSource('../../../pages/research/index.astro');
const componentSource = readSource(
  '../../../components/research/CosmicBreathProvenance.astro',
);
const observatoryAuthoritySource = readSource(
  '../../../data/research/CU-RESEARCH-OBSERVATORY-1.0.json',
);

const protectedRc3Files = new Map([
  [
    '../../../data/research/CU-COSMIC-BREATH-PROVENANCE-1.0.json',
    'a5795aed63b59a3adace79218603efd7946d8db55acc2b8728ac147eddedc88b',
  ],
  [
    '../cosmic-breath-provenance.ts',
    'a6d6d38f2c10cbb22525682dab78da53ee01f9b84b0564b4ab3bb3a3d28cdef1',
  ],
  [
    './cosmic-breath-provenance-authority.test.ts',
    'f2cfc875ff6ff9924f44006af198745de2e6515a38beafb653b755bcabea9c0a',
  ],
] as const);

describe('Cosmic Breath provenance destination — RC-3A-3', () => {
  it('mounts one additive subsection inside the existing source-provenance chapter', () => {
    expect(routeSource).toContain(
      "import CosmicBreathProvenance from '../../components/research/CosmicBreathProvenance.astro';",
    );
    expect(routeSource.match(/<CosmicBreathProvenance \/>/g)).toHaveLength(1);

    const panelStart = routeSource.indexOf('<SourceProvenancePanel');
    const mount = routeSource.indexOf('<CosmicBreathProvenance />');
    const panelEnd = routeSource.indexOf('</SourceProvenancePanel>');
    expect(panelStart).toBeGreaterThan(-1);
    expect(mount).toBeGreaterThan(panelStart);
    expect(panelEnd).toBeGreaterThan(mount);
  });

  it('uses the exact approved fragment, heading, and heading hierarchy', () => {
    expect(componentSource.match(/id="cosmic-breath-provenance"/g))
      .toHaveLength(1);
    expect(componentSource).toContain(
      'aria-labelledby="cosmic-breath-provenance-title"',
    );
    expect(componentSource).toContain(
      '<h3 id="cosmic-breath-provenance-title">',
    );
    expect(componentSource.match(
      /Cosmic Breath sources and provenance/g,
    )).toHaveLength(1);
  });

  it('consumes only the governed public projection', () => {
    expect(componentSource).toContain(
      "import { COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION } from '../../lib/research/cosmic-breath-provenance';",
    );
    expect(componentSource).not.toMatch(
      /CU-COSMIC-BREATH-PROVENANCE|rawAuthority|parseCosmicBreathProvenanceAuthority|buildCosmicBreathPublicSourceRecordProjection/,
    );
  });

  it('renders the 20 deterministic projected records once through one map', () => {
    expect(COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION).toHaveLength(20);
    expect(new Set(
      COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION.map(
        (record) => record.stableSourceRecordId,
      ),
    ).size).toBe(20);
    expect(componentSource.match(
      /COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION\.map\(\(record\)/g,
    )).toHaveLength(1);
    expect(componentSource.match(
      /data-source-record-id=\{record\.stableSourceRecordId\}/g,
    )).toHaveLength(1);
    expect(componentSource).toContain(
      '<h4 id={recordHeadingId}>{record.publicTitle}</h4>',
    );
  });

  it('uses semantic server-rendered section, list, article, and definition-list markup', () => {
    for (const markup of [
      '<section',
      '<ol class="cosmic-breath-provenance__records" role="list">',
      '<article aria-labelledby={recordHeadingId}>',
      '<dl>',
      '<dt>',
      '<dd>',
    ]) {
      expect(componentSource).toContain(markup);
    }
  });

  it('wraps complete long metadata values without truncation or script measurement', () => {
    expect(componentSource).toContain('overflow-wrap: anywhere');
    expect(componentSource).toContain('min-width: 0');
    expect(componentSource).toContain(
      'grid-template-columns: minmax(8rem, 12rem) minmax(0, 1fr)',
    );
    expect(componentSource).toMatch(
      /@media \(max-width: 48rem\)[\s\S]*?grid-template-columns: minmax\(0, 1fr\)/,
    );
    expect(componentSource).not.toMatch(
      /text-overflow|line-clamp|white-space:\s*nowrap|\.slice\(|\.substring\(|ResizeObserver/,
    );
  });

  it('uses the repository-native direct-fragment offset without custom scrolling', () => {
    expect(componentSource).toMatch(
      /\.cosmic-breath-provenance\s*\{[^}]*scroll-margin-top: var\(--space-6\)/,
    );
    expect(componentSource).not.toMatch(
      /scrollIntoView|scrollTo\(|location\.hash|\.focus\(/,
    );
    expect(componentSource.match(/id="cosmic-breath-provenance"/g))
      .toHaveLength(1);
  });

  it('uses only the exact approved labels for fields present in the projection', () => {
    for (const label of [
      'Record ID',
      'Source role',
      'Approved scope',
      'Not established by this source',
      'Source type',
      'Source identity',
      'Stable reference',
      'Currentness',
      'Protected authority relationship',
      'Public projection relationship',
      'Original external authority',
      'Limitations',
      'Access state',
      'Licensing',
      'Epistemic label',
    ]) {
      expect(componentSource).toContain(`<dt>${label}</dt>`);
    }
  });

  it('omits unavailable field labels and placeholder copy entirely', () => {
    for (const label of [
      'Short title',
      'Verified version',
      'Verified date',
      'SHA-256',
      'Review date',
      'Source destination',
      'Historical or variant notice',
    ]) {
      expect(componentSource).not.toContain(`<dt>${label}</dt>`);
    }
    expect(componentSource).not.toMatch(
      /\bTBD\b|\bN\/A\b|not available|coming soon|placeholder/i,
    );
  });

  it('renders optional projected fields only when present', () => {
    for (const field of [
      'filenameOrExternalIdentity',
      'immutableRefOrStableIdentifier',
      'currentnessStatement',
      'protectedAuthorityRelationship',
      'publicProjectionRelationship',
      'originalExternalAuthority',
      'licensingNotice',
    ]) {
      expect(componentSource).toContain(`record.${field} &&`);
    }
  });

  it('does not expose private governance fields or local absolute paths', () => {
    for (const field of [
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
    ]) {
      expect(componentSource).not.toContain(`record.${field}`);
    }
    expect(componentSource).not.toMatch(/\/Users\/|file:\/\/|~\//i);
    expect(componentSource).not.toMatch(/Part_II_(?:RM0|RC0|RC1|RC2|RC3)/);
  });

  it('preserves structural, integrity, companion, provenance, empirical, and bibliography records', () => {
    const records = new Map(
      COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION.map(
        (record) => [record.stableSourceRecordId, record],
      ),
    );
    expect(records.get('cb-001')?.approvedSourceRole).toBe(
      'Protected structural authority — exact existing scope',
    );
    expect(records.get('cb-003')?.approvedSourceRole).toBe(
      'Manifest or integrity record',
    );
    expect(records.get('cb-006')?.approvedSourceRole).toBe(
      'Primary/original provenance record',
    );
    expect(records.get('cb-024')?.approvedSourceRole).toBe(
      'Local bibliography or index record',
    );
    expect(records.get('cb-e05')?.approvedSourceRole).toBe(
      'Original empirical source',
    );
    expect(records.get('cb-002')?.approvedSourceRole).toBe(
      'Supporting source',
    );
  });

  it('retains governed exclusions and bounded relationship treatment', () => {
    const serialized = JSON.stringify(
      COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION,
    );
    for (const recordId of ['cb-004', 'cb-005', 'cb-058', 'cb-059', 'cb-d01']) {
      expect(serialized).not.toContain(`"${recordId}"`);
    }
    expect(serialized).not.toContain('CU Mathematical Model');
    expect(serialized).not.toContain('Empirical node classification');
    expect(serialized).not.toMatch(/complete marker authority/i);
    expect(serialized).not.toMatch(/final relationship(?:-source)? support/i);
  });

  it('renders no destination or external link because none is projected', () => {
    expect(
      COSMIC_BREATH_PUBLIC_SOURCE_RECORD_PROJECTION.every(
        (record) => record.approvedDestination === undefined,
      ),
    ).toBe(true);
    expect(componentSource).not.toMatch(/<a\b|href=|target=|rel=/);
    expect(componentSource).not.toContain('record.approvedDestination');
  });

  it('keeps the provenance destination separate from the migrated internal action', () => {
    const authority = JSON.parse(observatoryAuthoritySource) as {
      nodes: Array<{
        id: string;
        governingSourceDestination?: Record<string, unknown>;
      }>;
    };
    const breath = authority.nodes.find((node) => node.id === 'cosmic-breath');
    expect(breath?.governingSourceDestination).toEqual({
      kind: 'internal',
      path: 'research#cosmic-breath-provenance',
      label: 'Review Cosmic Breath sources and provenance',
    });
    expect(observatoryAuthoritySource).not.toContain(
      'View the Cosmic Breath source',
    );
    expect(observatoryAuthoritySource).not.toContain(
      'Cosmic_Breath_Calculation.md',
    );
    expect(componentSource.match(/id="cosmic-breath-provenance"/g))
      .toHaveLength(1);
  });

  it('adds no client hydration, browser script, persistence, or telemetry', () => {
    expect(componentSource).not.toMatch(
      /client:|<script|window\.|document\.|localStorage|sessionStorage|document\.cookie|analytics|telemetry/,
    );
    expect(routeSource).not.toContain('client:load');
  });

  it('adds no new route, alias, redirect, or alternate destination', () => {
    expect(routeSource).not.toMatch(
      /redirect|rewrite|cosmic-breath-provenance\.astro/,
    );
    expect(componentSource).not.toMatch(/canonical|alias|alternate destination/i);
  });

  it('keeps the RC-3A-1 and RC-3A-2 authority boundary byte-identical', () => {
    for (const [relativePath, expectedDigest] of protectedRc3Files) {
      expect(digest(readSource(relativePath)), relativePath).toBe(expectedDigest);
    }
  });
});
