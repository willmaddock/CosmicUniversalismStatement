import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const componentPath = fileURLToPath(
  new URL('../../../components/CosmicBreathEducation.astro', import.meta.url),
);
const routePath = fileURLToPath(new URL('../../../pages/cosmic-breath.astro', import.meta.url));
const componentSource = readFileSync(componentPath, 'utf8');
const routeSource = readFileSync(routePath, 'utf8');

const sections = [
  ['cosmic-breath-at-a-glance', 'Cosmic Breath at a Glance'],
  ['reading-the-structural-diagram', 'Reading the Structural Diagram'],
  ['tom-names-and-positions', 'TOM Names and Positions'],
  ['exponents-tetration-and-duration', 'Exponents, Tetration, and Duration'],
  [
    'expansion-compression-and-guarded-boundaries',
    'Expansion, Compression, and Guarded Boundaries',
  ],
  ['observable-universe-marker', 'Observable Universe Marker'],
  ['structural-and-epistemic-boundaries', 'Structural and Epistemic Boundaries'],
] as const;

describe('Cosmic Breath educational foundation', () => {
  it('provides seven stable sections without a redundant local menu', () => {
    for (const [id, heading] of sections) {
      expect(componentSource).toContain(`id="${id}"`);
      expect(componentSource).toContain(heading);
    }

    expect(componentSource).toContain('id="cosmic-breath-field-guide"');
    expect(componentSource).not.toContain('class="topic-nav"');
    expect(componentSource).toContain(
      'aria-label="Cosmic Breath Field Guide continuation"',
    );
    expect(componentSource).toContain('href="#cosmic-breath-page-guide"');
    expect(componentSource).toContain('href="#cosmic-breath-theory-method"');
  });

  it('states the structural counts and declared model-level anchors', () => {
    expect(componentSource).toMatch(/Selectable TOM states<\/dt><dd>51<\/dd>/);
    expect(componentSource).toMatch(/Expansion states<\/dt><dd>26<\/dd>/);
    expect(componentSource).toMatch(/Compression states<\/dt><dd>25<\/dd>/);
    expect(componentSource).toMatch(/Explicit guarded transitions<\/dt><dd>2<\/dd>/);
    expect(componentSource).toContain('Approximately 2.8 trillion years');
    expect(componentSource).toContain('Approximately 308 billion years');
    expect(componentSource).toContain('Approximately 3.108 trillion years');
    expect(componentSource).toMatch(
      /They should not be\s+reconstructed by silently summing or normalizing the individual TOM duration estimates\./,
    );
  });

  it('explains the two guarded boundaries and explicit cycle offsets', () => {
    expect(componentSource).toContain('Begin Compression');
    expect(componentSource).toContain('Begin the Next Cosmic Breath');
    expect(componentSource.match(/cycle offset/g)).toHaveLength(2);
    expect(componentSource).toContain('<strong>0</strong>');
    expect(componentSource).toContain('<strong>1</strong>');
    expect(componentSource.match(/Selectable state/g)).toHaveLength(2);
    expect(componentSource.match(/Guarded transition action/g)).toHaveLength(2);
  });

  it('distinguishes exponentiation, tetration, notation, and duration', () => {
    expect(componentSource).toContain('Ordinary exponentiation');
    expect(componentSource).toContain('2¹ = 2 · 2² = 4 · 2⁴ = 16 · 2¹⁶ = 65,536');
    expect(componentSource).toContain('2↑↑4');
    expect(componentSource).toContain('2↑↑65,536');
    expect(componentSource).toContain('does not mean n seconds');
    expect(componentSource).toContain('ztom’s declared one-second duration');
  });

  it('includes the exact geometry, marker, and epistemic language', () => {
    expect(componentSource).toContain(
      'This field is structural and index-based—not drawn to physical or temporal scale.',
    );
    expect(componentSource).toMatch(
      /Within the CU diagram, the non-selectable Observable Universe marker is positioned\s+structurally between compression-btom and compression-ctom\. It is not ctom, not\s+sub-ctom, and not one of the 51 selectable TOM states\./,
    );
    expect(componentSource).toMatch(
      /This placement is a CU structural proposition — not an empirical measurement\. The\s+diagram is structural and is not drawn to physical or temporal scale\./,
    );
    expect(componentSource).toContain(
      'CU structural proposition — not an empirical measurement',
    );
  });

  it('remains static without duplicating the complete state authority', () => {
    expect(componentSource).not.toContain('<script');
    expect(componentSource).not.toContain('client:');
    expect(componentSource).not.toContain('addEventListener');
    expect(componentSource).not.toContain('structuralStateId');
    expect(componentSource).not.toContain('orderedTomStates');
    expect(componentSource).not.toContain('data-state-index');
  });

  it('does not expose internal governance fields or artifact identifiers', () => {
    const forbiddenTerms = [
      'ownerDecisionDate',
      'ownerAuthority',
      'canonicalStructuralDigest',
      'relationshipToStructuralAuthority',
      'owner-approved-canonical',
      'sourceAlignmentStatus',
      'contentApprovalStatus',
      'sourceProvenance',
      'companion-content-authority-in-development',
      'CB-TOM-CONTENT-1.0.manifest.json',
    ];

    for (const term of forbiddenTerms) {
      expect(componentSource).not.toContain(term);
    }
  });

  it('replaces the sparse route details without changing the explorer shell', () => {
    expect(routeSource).toContain(
      "import CosmicBreathEducation from '../components/CosmicBreathEducation.astro';",
    );
    expect(routeSource).toContain('<CosmicBreathCycleExplorer />');
    expect(routeSource).toContain('<CosmicBreathEducation />');
    expect(routeSource).not.toContain('class="breath-details"');
    expect(routeSource).not.toContain('<ContentSection');
  });
});
