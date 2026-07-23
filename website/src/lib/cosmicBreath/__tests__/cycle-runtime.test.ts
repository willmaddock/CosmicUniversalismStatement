import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  serializedStructuralRuntimePayload,
  structuralRuntimeLedger,
} from '../cycle-ledger';
import {
  STRUCTURAL_RUNTIME_STATE_KEYS,
  STRUCTURAL_RUNTIME_TRANSITION_KEYS,
  createCycleDragEngine,
  createCycleStateEngine,
  findNearestAnchor,
  parseStructuralRuntimeLedger,
  parseStructuralRuntimePayload,
  serializeStructuralRuntimePayload,
  type StructuralDragAnchor,
} from '../cycle-runtime';

const explorerComponentPath = fileURLToPath(new URL(
  '../../../components/CosmicBreathCycleExplorer.astro',
  import.meta.url,
));
const contentAuthorityPath = fileURLToPath(new URL(
  '../../../data/cosmic-breath/CB-TOM-CONTENT-1.0.json',
  import.meta.url,
));
const structuralAuthorityPath = fileURLToPath(new URL(
  '../../../data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json',
  import.meta.url,
));

const expectedContentDigest =
  '5eab61c46e1922cdbd52be9c128e2b18a29b6540fdec91a840e3801c580b12be';
const expectedStructuralDigest =
  'dcc34d5b7d9e32afb8d0cb97b029a12e0025959095a21b4168d59aff575da825';

describe('Cosmic Breath structural browser runtime', () => {
  it('projects the complete structural inventory through explicit runtime whitelists', () => {
    expect(structuralRuntimeLedger.states).toHaveLength(51);
    expect(new Set(structuralRuntimeLedger.states.map((state) => state.id))).toHaveLength(51);
    expect(structuralRuntimeLedger.states.filter((state) => state.phase === 'expansion')).toHaveLength(26);
    expect(structuralRuntimeLedger.states.filter((state) => state.phase === 'compression')).toHaveLength(25);
    expect(structuralRuntimeLedger.states.map((state) => state.cycleIndex)).toEqual(
      Array.from({ length: 51 }, (_, index) => index + 1),
    );
    expect(structuralRuntimeLedger.transitions).toHaveLength(2);
    expect(new Set(structuralRuntimeLedger.transitions.map((transition) => transition.id))).toHaveLength(2);

    const stateKeys = new Set<string>(STRUCTURAL_RUNTIME_STATE_KEYS);
    const transitionKeys = new Set<string>(STRUCTURAL_RUNTIME_TRANSITION_KEYS);
    for (const state of structuralRuntimeLedger.states) {
      expect(Object.keys(state).every((key) => stateKeys.has(key))).toBe(true);
    }
    for (const transition of structuralRuntimeLedger.transitions) {
      expect(Object.keys(transition).every((key) => transitionKeys.has(key))).toBe(true);
    }
  });

  it('rejects governance keys, unresolved references, and malformed ordering', () => {
    const governanceState = structuredClone(structuralRuntimeLedger) as unknown as {
      states: Array<Record<string, unknown>>;
      transitions: Array<Record<string, unknown>>;
    };
    governanceState.states[0].ownerDecisionDate = '2026-07-22';
    expect(() => parseStructuralRuntimeLedger(governanceState)).toThrow(/non-runtime key/);

    const governanceTransition = structuredClone(structuralRuntimeLedger) as unknown as {
      states: Array<Record<string, unknown>>;
      transitions: Array<Record<string, unknown>>;
    };
    governanceTransition.transitions[0].provenance = ['internal'];
    expect(() => parseStructuralRuntimeLedger(governanceTransition)).toThrow(/non-runtime key/);

    const unresolved = structuredClone(structuralRuntimeLedger) as unknown as {
      states: Array<Record<string, unknown>>;
      transitions: Array<Record<string, unknown>>;
    };
    unresolved.states[0].nextId = 'expansion-missing';
    expect(() => parseStructuralRuntimeLedger(unresolved)).toThrow(/nextId reference/);

    const reordered = structuredClone(structuralRuntimeLedger) as unknown as {
      states: Array<Record<string, unknown>>;
      transitions: Array<Record<string, unknown>>;
    };
    reordered.states[0].cycleIndex = 2;
    expect(() => parseStructuralRuntimeLedger(reordered)).toThrow(/consecutive/);
  });

  it('serializes safely and reparses all state and transition references', () => {
    const parsed = parseStructuralRuntimePayload(serializedStructuralRuntimePayload);
    const ids = new Set(parsed.states.map((state) => state.id));
    expect(parsed.states.every((state) =>
      (state.previousId === null || ids.has(state.previousId))
      && (state.nextId === null || ids.has(state.nextId))
    )).toBe(true);
    expect(parsed.transitions.every((transition) =>
      ids.has(transition.fromId) && ids.has(transition.toId)
    )).toBe(true);

    const unsafe = structuredClone(structuralRuntimeLedger) as unknown as {
      states: Array<Record<string, unknown>>;
      transitions: Array<Record<string, unknown>>;
    };
    unsafe.states[0].label = '</script><script>alert("unsafe")</script>';
    const serialized = serializeStructuralRuntimePayload(unsafe);
    expect(serialized).not.toContain('</script>');
    expect(parseStructuralRuntimePayload(serialized).states[0].label).toBe(
      '</script><script>alert("unsafe")</script>',
    );
  });

  it('preserves guarded mechanics and direct-inspection behavior in the runtime engine', () => {
    const engine = createCycleStateEngine(structuralRuntimeLedger);
    const atom = engine.selectStateById(engine.INITIAL_CYCLE_STATE, 'expansion-atom');
    expect(engine.moveToNextState(atom)).toMatchObject({ selectedIndex: 25, cycleCount: 0 });
    expect(engine.continueFromAtom(atom)).toMatchObject({ selectedIndex: 26, cycleCount: 0 });

    const ztom = engine.selectStateById(atom, 'compression-ztom');
    expect(engine.moveToNextState(ztom)).toMatchObject({ selectedIndex: 50, cycleCount: 0 });
    expect(engine.beginNextCosmicBreath(ztom)).toMatchObject({ selectedIndex: 0, cycleCount: 1 });
    expect(engine.selectStateById(ztom, 'expansion-sub-ztom')).toMatchObject({
      selectedIndex: 0,
      cycleCount: 0,
    });
  });

  it('preserves all drag anchors, cross-phase selection, and lower-index tie handling', () => {
    const drag = createCycleDragEngine(structuralRuntimeLedger);
    expect(drag.selectableDragAnchors).toHaveLength(51);
    expect(drag.findNearestSelectableAnchor(drag.compressionDragAnchors[0].point).stateId).toBe(
      'compression-btom',
    );

    const anchors: StructuralDragAnchor[] = [
      {
        phase: 'expansion',
        phaseIndex: 2,
        canonicalIndex: 1,
        stateId: 'higher',
        label: 'higher',
        point: { x: 0.4, y: 0.5 },
      },
      {
        phase: 'compression',
        phaseIndex: 1,
        canonicalIndex: 0,
        stateId: 'lower',
        label: 'lower',
        point: { x: 0.6, y: 0.5 },
      },
    ];
    expect(findNearestAnchor({ x: 0.5, y: 0.5 }, anchors).stateId).toBe('lower');
  });

  it('keeps the raw authority server-side and browser imports runtime-only modules', () => {
    const componentSource = readFileSync(explorerComponentPath, 'utf8');
    const clientSource = componentSource.slice(componentSource.lastIndexOf('<script>'));
    expect(componentSource.slice(0, componentSource.lastIndexOf('<script>'))).toContain(
      "from '../lib/cosmicBreath/cycle-ledger'",
    );
    for (const forbiddenImport of [
      "from '../lib/cosmicBreath/cycle-ledger'",
      "from '../lib/cosmicBreath/cycle-state'",
      "from '../lib/cosmicBreath/cycle-drag'",
      "from '../lib/cosmicBreath/cycle-controls'",
    ]) expect(clientSource).not.toContain(forbiddenImport);
    expect(clientSource).toContain("from '../lib/cosmicBreath/cycle-runtime'");
    expect(componentSource).toContain('data-structural-runtime');
  });

  it('leaves both sealed authority digests unchanged', () => {
    const digest = (path: string) =>
      createHash('sha256').update(readFileSync(path)).digest('hex');
    expect(digest(contentAuthorityPath)).toBe(expectedContentDigest);
    expect(digest(structuralAuthorityPath)).toBe(expectedStructuralDigest);
  });
});
