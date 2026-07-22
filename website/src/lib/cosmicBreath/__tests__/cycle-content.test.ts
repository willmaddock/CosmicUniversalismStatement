import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import rawContentLedger from '../../../data/cosmic-breath/CB-TOM-CONTENT-1.0.json';
import {
  PUBLIC_EPISTEMIC_LABEL,
  contentLedger,
  getPublicTomContentById,
  getTomContentById,
  orderedPublicTomContent,
  orderedTomContent,
  parseContentLedger,
  parseTransitionContentRecord,
} from '../cycle-content';
import { orderedTomStates } from '../cycle-ledger';

const structuralLedgerPath = fileURLToPath(new URL(
  '../../../data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json',
  import.meta.url,
));
const repositoryContentSourcePath = fileURLToPath(new URL(
  '../../../../../ResearchFiles/Cosmic_Breath_Calculation.md',
  import.meta.url,
));
const expectedStructuralDigest =
  'dcc34d5b7d9e32afb8d0cb97b029a12e0025959095a21b4168d59aff575da825';

const exactSourceDescriptions = [
  'Final compression breath',
  'Recursive core shell',
  'Final ethical cap',
  'Quantum firewall shell',
  'Symbolic/ethical lock',
  'Cosmic checksum layer',
  'Holographic verification',
  'Ethical engagement',
  'Entropy modulation',
  'Memory transfer',
  'Pre-reset bridge',
  'Boundary stabilization',
  'Recursive feedback',
  'Holographic projection',
  'Pre-Big Bang state',
  'Quantum foam rebirth',
  'Black hole age',
  'Spacetime contraction begins',
  'Heat death begins',
  'Quantum encoding phase',
  'Post-biological AI expansion',
  'Alien/civilization stage',
  'Planetary biosphere evolution',
  'Star life cycle era',
  'Supercluster formation',
  'Start of compression',
  'Galactic evolution and contraction',
  'Final stellar formations',
  'Planetary collapse',
  'Human/civilization memory condensation',
  'AI implosion stage',
  'Consciousness holography',
  'Heat death approach',
  'Spacetime wrinkle forming',
  'Collapse threshold',
  'Quantum fog closing',
  'Holographic reversal',
  'Time lattice inversion',
  'Feedback end',
  'Cosmic null stabilization',
  'Reset layering',
  'Final memory imprint',
  'Entropy zero point',
  'Ethical firewall gate',
  'Collapse checksum',
  'Closure sequence initiated',
  'Symbolic compression',
  'Recursive limit breach',
  'Divine fall-off shell',
  'Pre-ZTOM divine echo',
  'ZTOM: full universal reset',
] as const;

type MutableLedger = {
  stateContent: Array<Record<string, unknown>>;
  [key: string]: unknown;
};

const mutableLedger = (): MutableLedger =>
  structuredClone(rawContentLedger) as unknown as MutableLedger;

describe('CB-TOM-CONTENT-1.0 companion content authority', () => {
  it('contains one unique content record for every structural TOM state', () => {
    expect(orderedTomContent).toHaveLength(51);
    expect(new Set(orderedTomContent.map((record) => record.structuralStateId))).toHaveLength(51);
    expect(orderedTomContent.map((record) => record.structuralStateId)).toEqual(
      orderedTomStates.map((state) => state.id),
    );
  });

  it('joins exactly 26 expansion and 25 compression records through structural authority', () => {
    expect(orderedPublicTomContent.filter((record) => record.phase === 'expansion')).toHaveLength(26);
    expect(orderedPublicTomContent.filter((record) => record.phase === 'compression')).toHaveLength(25);
    expect(orderedPublicTomContent.map((record) => record.cycleIndex)).toEqual(
      Array.from({ length: 51 }, (_, index) => index + 1),
    );
  });

  it('preserves representative tetration and ordinary-power strings exactly', () => {
    expect(getTomContentById('expansion-sub-ztom').exactQuantumStateDisplay).toBe('2↑↑65,536');
    expect(getTomContentById('expansion-sub-ytom').exactQuantumStateDisplay).toBe('2↑↑40');
    expect(getTomContentById('expansion-sub-etom').exactQuantumStateDisplay).toBe('2↑↑4');
    expect(getTomContentById('compression-btom').exactQuantumStateDisplay).toBe('2² = 4');
    expect(getTomContentById('compression-ctom').exactQuantumStateDisplay).toBe('2⁴ = 16');
    expect(getTomContentById('compression-dtom').exactQuantumStateDisplay).toBe('2¹⁶ = 65,536');
  });

  it('preserves representative duration strings exactly', () => {
    const durations = new Set(orderedTomContent.map((record) => record.exactDurationEstimateDisplay));
    for (const duration of [
      '2.704e-8 seconds',
      '0.0002704 seconds',
      '4.506 minutes',
      '7.51 hours',
      '0.8547 years',
      '4,273.5 years',
      '28 billion years',
      '280 billion years',
      '2.8 trillion years',
    ]) expect(durations.has(duration)).toBe(true);
  });

  it('preserves all 51 exact original source descriptions in structural order', () => {
    expect(orderedTomContent.map((record) => record.exactSourceCuDescription)).toEqual(
      exactSourceDescriptions,
    );
  });

  it('matches every exact TOM label, notation, duration, and description in the repository source table', () => {
    const sourceRows = readFileSync(repositoryContentSourcePath, 'utf8')
      .split('\n')
      .map((line) => line.match(/^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$/))
      .filter((match): match is RegExpMatchArray => match !== null)
      .map((match) => match.slice(1).map((field) => field.trim()))
      .filter(([sourceTomLabel]) => /^(?:sub-)?[a-z]+tom$|^atom$/.test(sourceTomLabel));

    expect(sourceRows).toHaveLength(51);
    expect(orderedTomContent.map((record) => [
      record.sourceProvenance.sourceTomLabel,
      record.exactQuantumStateDisplay,
      record.exactDurationEstimateDisplay,
      record.exactSourceCuDescription,
    ])).toEqual(sourceRows);
  });

  it('preserves the approved sub-ztom public, canonical, and helper content exactly', () => {
    const subZtom = getTomContentById('expansion-sub-ztom');
    expect(subZtom.publicCuDescription).toBe(
      'Sub-ztom carries the final compression breath into the opening condition of renewed expansion. Within this one-second CU scale, the prior breath’s concentrated memory and structural potential are preserved as the seed condition for the next outward cycle.',
    );
    expect(subZtom.canonicalCuDescription).toBe(
      'Expansion-sub-ztom carries the final compression breath into the opening condition of renewed expansion. It holds the completed continuity, memory, and concentrated potential inherited from compression-ztom without becoming compression-ztom itself. Entry occurs only through the guarded next-breath transition, after which sub-ztom serves as the renewed seed from which the Cosmic Breath begins extending its preserved inheritance toward the recursive core shell.',
    );
    expect(subZtom.quantumStateHelperShort).toBe('Two arrows mean repeated exponentiation.');
    expect(subZtom.quantumStateHelperLong).toBe(
      'The two up arrows indicate tetration, or repeated exponentiation. The expression 2↑↑65,536 represents a power tower of 2s with a height of 65,536. Within Cosmic Universalism, it identifies the assigned quantum-state scale of sub-ztom. It is distinct from elapsed time and does not mean 65,536 seconds, objects, or particles.',
    );
    expect(subZtom.durationHelperShort).toBe('A rapid, concentrated CU transition.');
    expect(subZtom.durationHelperLong).toBe(
      'The approved CU duration estimate for sub-ztom is one second, approximately the time needed to say ‘one.’ Because this is an extremely brief interval, its description concerns concentrated memory, structural inheritance, and the opening of a new breath rather than the development of planets, biological life, civilizations, stars, or galaxies within that second.',
    );
    expect(subZtom.tomLayerExplanation).toBe(
      'Sub-ztom stands at the opening of the expansion sequence while carrying the final compression breath from the prior cycle. The guarded next-breath transition separates it from compression-ztom. Its one-second duration describes a rapid seed condition, while its tetration notation represents extreme CU quantum concentration. Together, these features support its role as the preserved memory and structural potential from which renewed expansion begins.',
    );
  });

  it('uses the approved public description first and exact source description as fallback', () => {
    expect(getPublicTomContentById('expansion-sub-ztom').publicCuDescription).toContain(
      'opening condition of renewed expansion',
    );
    expect(getPublicTomContentById('expansion-sub-ytom').publicCuDescription).toBe(
      'Recursive core shell',
    );
    expect(getPublicTomContentById('compression-ztom').publicCuDescription).toBe(
      'ZTOM: full universal reset',
    );
  });

  it('omits missing helpers and internal governance from the public projection', () => {
    const fallback = getPublicTomContentById('expansion-sub-ytom');
    expect(fallback).not.toHaveProperty('quantumStateHelperShort');
    expect(fallback).not.toHaveProperty('durationHelperLong');
    expect(fallback).not.toHaveProperty('tomLayerExplanation');
    for (const record of orderedPublicTomContent) {
      expect(record).not.toHaveProperty('contentApprovalStatus');
      expect(record).not.toHaveProperty('sourceAlignmentStatus');
      expect(record).not.toHaveProperty('sourceProvenance');
      expect(JSON.stringify(record)).not.toMatch(/under source-alignment review|intentionally withheld/i);
    }
  });

  it('carries the exact public epistemic label on every internal and public record', () => {
    expect(contentLedger.publicEpistemicLabel).toBe(PUBLIC_EPISTEMIC_LABEL);
    expect(orderedTomContent.every((record) => record.publicEpistemicLabel === PUBLIC_EPISTEMIC_LABEL)).toBe(true);
    expect(orderedPublicTomContent.every((record) => record.publicEpistemicLabel === PUBLIC_EPISTEMIC_LABEL)).toBe(true);
  });

  it('preserves separate uploaded and repository provenance and declared phase totals', () => {
    const provenance = getTomContentById('compression-btom').sourceProvenance;
    expect(provenance.uploadedSourceName).toBe('Cosmic_Breath_Calculation(6).md');
    expect(provenance.repositorySourcePath).toBe('ResearchFiles/Cosmic_Breath_Calculation.md');
    expect(provenance.sourceSection).toBe('Compression Phase (btom ➝ ztom)');
    expect(contentLedger.declaredPhaseTotals).toEqual({
      expansion: 'approximately 2.8 trillion years',
      compression: 'approximately 308 billion years',
      completeCosmicBreath: 'approximately 3.108 trillion years',
      derivation: 'Owner-approved declared CU totals; not recomputed from state rows.',
    });
  });

  it('freezes internal records, provenance, optional arrays, and public projections', () => {
    const internal = getTomContentById('expansion-sub-ztom');
    const publicRecord = getPublicTomContentById('expansion-sub-ztom');
    expect(Object.isFrozen(contentLedger)).toBe(true);
    expect(Object.isFrozen(orderedTomContent)).toBe(true);
    expect(Object.isFrozen(orderedPublicTomContent)).toBe(true);
    expect(Object.isFrozen(internal)).toBe(true);
    expect(Object.isFrozen(internal.sourceProvenance)).toBe(true);
    expect(Object.isFrozen(internal.structuralTheme)).toBe(true);
    expect(Object.isFrozen(internal.relatedTransitionContentIds)).toBe(true);
    expect(Object.isFrozen(publicRecord)).toBe(true);
    expect(Object.isFrozen(publicRecord.structuralTheme)).toBe(true);
  });

  it('rejects malformed counts, duplicate IDs, and unknown structural IDs clearly', () => {
    const shortLedger = mutableLedger();
    shortLedger.stateContent.pop();
    expect(() => parseContentLedger(shortLedger)).toThrow(/exactly 51/);

    const duplicateLedger = mutableLedger();
    duplicateLedger.stateContent[50] = structuredClone(duplicateLedger.stateContent[0]);
    expect(() => parseContentLedger(duplicateLedger)).toThrow(/must be unique/);

    const unknownLedger = mutableLedger();
    unknownLedger.stateContent[0].structuralStateId = 'expansion-unknown';
    expect(() => parseContentLedger(unknownLedger)).toThrow(/unknown structuralStateId/);
  });

  it('rejects malformed source fields, provenance, statuses, and epistemic labels', () => {
    const missingSource = mutableLedger();
    delete missingSource.stateContent[0].exactQuantumStateDisplay;
    expect(() => parseContentLedger(missingSource)).toThrow(/exactQuantumStateDisplay/);

    const wrongLabel = mutableLedger();
    wrongLabel.stateContent[0].publicEpistemicLabel = 'Internal review';
    expect(() => parseContentLedger(wrongLabel)).toThrow(/publicEpistemicLabel/);

    const wrongStatus = mutableLedger();
    wrongStatus.stateContent[0].contentApprovalStatus = 'published';
    expect(() => parseContentLedger(wrongStatus)).toThrow(/contentApprovalStatus/);

    const wrongProvenance = mutableLedger();
    const provenance = wrongProvenance.stateContent[0].sourceProvenance as Record<string, unknown>;
    provenance.sourceTomLabel = 'ztom';
    expect(() => parseContentLedger(wrongProvenance)).toThrow(/sourceTomLabel/);
  });

  it('keeps transition content separate from state notation, durations, and positions', () => {
    expect(parseTransitionContentRecord({
      transitionId: 'transition-atom-to-btom',
      canonicalTitle: 'Guarded turn to compression',
      structuralTheme: ['expansion', 'compression'],
      contentApprovalStatus: 'draft',
    })).toEqual({
      transitionId: 'transition-atom-to-btom',
      canonicalTitle: 'Guarded turn to compression',
      structuralTheme: ['expansion', 'compression'],
      contentApprovalStatus: 'draft',
    });
    expect(() => parseTransitionContentRecord({
      transitionId: 'transition-atom-to-btom',
      exactQuantumStateDisplay: '2¹ = 2',
      contentApprovalStatus: 'draft',
    })).toThrow(/must not contain "exactQuantumStateDisplay"/);
    expect(() => parseTransitionContentRecord({
      transitionId: 'transition-ztom-to-next-sub-ztom',
      exactDurationEstimateDisplay: '1 second',
      contentApprovalStatus: 'draft',
    })).toThrow(/must not contain "exactDurationEstimateDisplay"/);
  });

  it('leaves the canonical structural ledger digest unchanged', () => {
    const digest = createHash('sha256').update(readFileSync(structuralLedgerPath)).digest('hex');
    expect(digest).toBe(expectedStructuralDigest);
  });
});
