import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

type TomPhase = 'expansion' | 'compression';
type BoundaryRole = 'new-cosmic-seed' | 'expansion-pause' | 'reset-pause' | null;

interface StructuralTomState {
  id: string;
  label: string;
  phase: TomPhase;
  cycleIndex: number;
  phaseIndex: number;
  previousId: string | null;
  nextId: string | null;
  boundaryRole: BoundaryRole;
  classification: 'cu-theoretical-proposition';
  approvalStatus: 'owner-approved-structural';
  provenance: string[];
}

interface GuardedTransition {
  id: string;
  fromId: string;
  toId: string;
  toCycleOffset: 0 | 1;
  type: 'phase-continuation' | 'next-breath-reset';
  label: string;
  requiresExplicitAction: true;
  selectableTomState: false;
  approvalStatus: 'owner-approved-structural';
  provenance: string[];
}

interface StructuralLedger {
  ledgerId: 'CB-TOM-STRUCTURAL-1.0';
  withheldFields: string[];
  states: StructuralTomState[];
  transitions: GuardedTransition[];
}

const ledgerUrl = new URL(
  '../../../data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json',
  import.meta.url,
);
const digestUrl = new URL(
  '../../../../../documentation/cosmic-breath/ledgers/CB-TOM-STRUCTURAL-1.0.sha256',
  import.meta.url,
);
const ledgerBytes = readFileSync(ledgerUrl);
const ledger = JSON.parse(ledgerBytes.toString('utf8')) as StructuralLedger;
const stateById = new Map(ledger.states.map((state) => [state.id, state]));
const withheldStateFields = [
  'quantumNotation',
  'duration',
  'machineMagnitude',
  'chronologySensitiveDescription',
] as const;

describe('CB-TOM-STRUCTURAL-1.0 canonical ledger', () => {
  it('contains 51 uniquely identified states with consecutive cycle indices', () => {
    expect(ledger.states).toHaveLength(51);
    expect(new Set(ledger.states.map((state) => state.id))).toHaveLength(51);
    expect(ledger.states.map((state) => state.cycleIndex)).toEqual(
      Array.from({ length: 51 }, (_, index) => index + 1),
    );
  });

  it('contains exactly 26 expansion and 25 compression states', () => {
    const expansion = ledger.states.filter((state) => state.phase === 'expansion');
    const compression = ledger.states.filter((state) => state.phase === 'compression');

    expect(expansion).toHaveLength(26);
    expect(compression).toHaveLength(25);
    expect(expansion.map((state) => state.phaseIndex)).toEqual(
      Array.from({ length: 26 }, (_, index) => index + 1),
    );
    expect(compression.map((state) => state.phaseIndex)).toEqual(
      Array.from({ length: 25 }, (_, index) => index + 1),
    );
  });

  it('preserves the approved seed, atom, and ztom phase and boundary roles', () => {
    expect(stateById.get('expansion-sub-ztom')).toMatchObject({
      phase: 'expansion',
      boundaryRole: 'new-cosmic-seed',
      previousId: null,
    });
    expect(stateById.get('expansion-atom')).toMatchObject({
      phase: 'expansion',
      boundaryRole: 'expansion-pause',
      nextId: null,
    });
    expect(stateById.get('compression-btom')).toMatchObject({
      phase: 'compression',
      previousId: null,
    });
    expect(stateById.get('compression-ztom')).toMatchObject({
      phase: 'compression',
      boundaryRole: 'reset-pause',
      nextId: null,
    });
  });

  it('keeps ordinary adjacency inside guarded phase boundaries', () => {
    for (const state of ledger.states) {
      if (state.nextId !== null) {
        const next = stateById.get(state.nextId);
        expect(next, `Missing next state for ${state.id}`).toBeDefined();
        expect(next?.phase).toBe(state.phase);
        expect(next?.previousId).toBe(state.id);
      }

      if (state.previousId !== null) {
        const previous = stateById.get(state.previousId);
        expect(previous, `Missing previous state for ${state.id}`).toBeDefined();
        expect(previous?.phase).toBe(state.phase);
        expect(previous?.nextId).toBe(state.id);
      }
    }
  });

  it('defines exactly two explicit non-selectable guarded transitions', () => {
    expect(ledger.transitions).toHaveLength(2);
    expect(ledger.transitions.every((transition) =>
      transition.requiresExplicitAction && !transition.selectableTomState
    )).toBe(true);

    expect(ledger.transitions).toContainEqual(expect.objectContaining({
      id: 'transition-atom-to-btom',
      fromId: 'expansion-atom',
      toId: 'compression-btom',
      toCycleOffset: 0,
      type: 'phase-continuation',
    }));
    expect(ledger.transitions).toContainEqual(expect.objectContaining({
      id: 'transition-ztom-to-next-sub-ztom',
      fromId: 'compression-ztom',
      toId: 'expansion-sub-ztom',
      toCycleOffset: 1,
      type: 'next-breath-reset',
    }));
  });

  it('omits every withheld field from selectable states', () => {
    expect(ledger.withheldFields).toEqual([...withheldStateFields]);

    for (const state of ledger.states) {
      for (const field of withheldStateFields) {
        expect(state).not.toHaveProperty(field);
      }
    }
  });

  it('matches the recorded SHA-256 digest', () => {
    const recordedLine = readFileSync(digestUrl, 'utf8').trim();
    const [recordedDigest, recordedPath] = recordedLine.split(/\s+/);
    const actualDigest = createHash('sha256').update(ledgerBytes).digest('hex');

    expect(recordedPath).toBe(
      'website/src/data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json',
    );
    expect(actualDigest).toBe(recordedDigest);
  });
});
