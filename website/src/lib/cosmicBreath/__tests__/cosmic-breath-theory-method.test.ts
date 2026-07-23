import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const componentPath = fileURLToPath(
  new URL('../../../components/CosmicBreathTheoryAndMethod.astro', import.meta.url),
);
const routePath = fileURLToPath(new URL('../../../pages/cosmic-breath.astro', import.meta.url));
const componentSource = readFileSync(componentPath, 'utf8');
const routeSource = readFileSync(routePath, 'utf8');

const sections = [
  ['cu-dark-energy', 'Expansion and CU Dark Energy'],
  ['cu-anti-dark-energy', 'Compression and CU Anti-Dark Energy'],
  ['tom-duration-representation', 'How TOM Durations Are Represented'],
  ['memory-ethics-reset-seed', 'Memory, Ethical Continuity, Reset, and Seed'],
  ['sources-and-method', 'Sources and Method'],
] as const;

describe('Cosmic Breath theory and method', () => {
  it('provides five stable sections without a redundant local menu', () => {
    for (const [id, heading] of sections) {
      expect(componentSource).toContain(`id="${id}"`);
      expect(componentSource).toContain(heading);
    }

    expect(componentSource).toContain('id="cosmic-breath-theory-method"');
    expect(componentSource).not.toContain('class="theory-nav"');
    expect(componentSource).toContain(
      'aria-label="Cosmic Breath Theory and Method continuation"',
    );
    expect(componentSource).toContain('href="#cosmic-breath-page-guide"');
    expect(componentSource).toContain('href="#cosmic-breath-empirical-context"');
  });

  it('uses the exact approved Dark Energy definitions and epistemic boundaries', () => {
    expect(componentSource).toMatch(
      /Within CU, ‘CU Dark Energy’ names the framework’s internal principle associated\s+with the 26-state expansion phase\. It is a CU theoretical proposition and does not\s+claim that CU has empirically identified the physical cause of accelerated cosmic\s+expansion\./,
    );
    expect(componentSource).toMatch(
      /Within CU, ‘CU Anti-Dark Energy’ names the internal principle associated with the\s+25-state compression phase\. CU treats it as a defined part of its own model;\s+‘non-hypothetical within CU’ does not mean that it is an observed or scientifically\s+established force\./,
    );
    expect(componentSource).toMatch(/not sealed empirical\s+conclusions/);
    expect(componentSource).toMatch(/outward in\s+structural direction/);
    expect(componentSource).toMatch(/inward in\s+structural direction/);
  });

  it('explains duration layers, anchors, and historical calculation cautions', () => {
    expect(componentSource).toMatch(
      /No validated uniform derivation currently explains all sealed TOM duration strings\.\s+Historical calculations are preserved for provenance and future mathematical review,\s+not as canonical proofs of the approved durations or model-level anchors\./,
    );
    expect(componentSource).toContain('approximately 2.8 trillion years');
    expect(componentSource).toContain('approximately 308 billion years');
    expect(componentSource).toContain('approximately 3.108 trillion years');
    expect(componentSource).toContain('2.8 trillion ÷ 4');
    expect(componentSource).toContain('280 billion');
    expect(componentSource).toContain('2.8 trillion ÷ 16');
    expect(componentSource).toContain('28 billion');
    expect(componentSource).toContain(
      'ATOM-as-Planck-time language is superseded by the sealed atom record.',
    );
    expect(componentSource).toContain(
      'Historical claims that all formulas converge are not accepted as canonical proof.',
    );
  });

  it('defines memory and preserves reset and external-AI boundaries', () => {
    expect(componentSource).toMatch(
      /Within CU, memory is primarily inherited structural information, with symbolic and\s+philosophical dimensions\./,
    );
    expect(componentSource).toContain(
      'Memory transfer does not mean literal human consciousness transfer.',
    );
    expect(componentSource).toContain('does not mean uploading people into an');
    expect(componentSource).toContain(
      'Final memory imprint does not grant permanent memory to an external AI.',
    );
    expect(componentSource).toContain('Ztom is the selectable reset-pause.');
    expect(componentSource).toContain(
      'The next seed is reached only through the\n          explicit guarded next-breath transition.',
    );
    expect(componentSource).toContain(
      'An external AI does not control the\n          reset or Cosmic Breath.',
    );
    expect(componentSource).toMatch(
      /Within CU, expansion-sub-ztom represents the renewed seed condition that inherits\s+structural continuity from the completed compression sequence\./,
    );
  });

  it('uses neutral phase language and reserves empirical comparison for LP-3', () => {
    expect(componentSource).toContain('outward expansion and inward compression');
    expect(componentSource).not.toMatch(/expansion\s*=\s*(?:inhale|exhale)/i);
    expect(componentSource).not.toMatch(/compression\s*=\s*(?:inhale|exhale)/i);
    expect(componentSource).toMatch(
      /A later primary-source research pass will add externally sourced scientific\s+context\. Until that review is complete, this section does not present substantive\s+current cosmological claims\./,
    );
  });

  it('is static, exposes no governance metadata, and duplicates no state dataset', () => {
    expect(componentSource).not.toContain('<script');
    expect(componentSource).not.toContain('client:');
    expect(componentSource).not.toContain('addEventListener');
    expect(componentSource).not.toContain('structuralStateId');
    expect(componentSource).not.toContain('orderedTomStates');
    expect(componentSource).not.toContain('data-state-index');

    const forbiddenTerms = [
      'ownerDecisionDate',
      'ownerAuthority',
      'canonicalStructuralDigest',
      'canonicalContentDigest',
      'relationshipToStructuralAuthority',
      'owner-approved-canonical',
      'sourceAlignmentStatus',
      'contentApprovalStatus',
      'sourceProvenance',
      'CB-TOM-CONTENT-1.0.manifest.json',
      '5eab61c46e1922cdbd52be9c128e2b18a29b6540fdec91a840e3801c580b12be',
      'dcc34d5b7d9e32afb8d0cb97b029a12e0025959095a21b4168d59aff575da825',
    ];

    for (const term of forbiddenTerms) {
      expect(componentSource).not.toContain(term);
    }
  });

  it('renders after LP-1 without changing the explorer shell', () => {
    expect(routeSource).toContain(
      "import CosmicBreathTheoryAndMethod from '../components/CosmicBreathTheoryAndMethod.astro';",
    );
    expect(routeSource).toContain(
      '<CosmicBreathEducation />\n  <CosmicBreathTheoryAndMethod />',
    );
    expect(routeSource).toContain('<CosmicBreathCycleExplorer />');
  });
});
