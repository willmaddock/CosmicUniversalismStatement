import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  cosmicBreathEmpiricalSources,
  empiricalSourceClasses,
} from '../../../data/cosmic-breath/cosmic-breath-empirical-sources';

const componentPath = fileURLToPath(
  new URL('../../../components/CosmicBreathEmpiricalScience.astro', import.meta.url),
);
const routePath = fileURLToPath(new URL('../../../pages/cosmic-breath.astro', import.meta.url));
const componentSource = readFileSync(componentPath, 'utf8');
const routeSource = readFileSync(routePath, 'utf8');

const sections = [
  ['empirical-observations', 'What Observations Establish'],
  ['empirical-open-questions', 'What Remains Unknown'],
  ['scientific-cosmic-futures', 'Scientific Models of the Far Future'],
  ['scientific-units-notation', 'Units, Scales, and Mathematical Notation'],
  ['quantum-holography-information', 'Quantum Gravity, Holography, and Information'],
  ['where-cu-begins', 'Where CU Begins'],
  ['scientific-sources', 'Sources and Scientific Boundaries'],
] as const;

const compact = (value: string) => value.replace(/\s+/g, ' ').trim();
const compactComponent = compact(componentSource);

describe('Cosmic Breath empirical science', () => {
  it('exists, renders after LP-2, and adds no client JavaScript', () => {
    expect(routeSource).toContain(
      "import CosmicBreathEmpiricalScience from '../components/CosmicBreathEmpiricalScience.astro';",
    );
    expect(routeSource).toContain(
      '<CosmicBreathTheoryAndMethod />\n  <CosmicBreathEmpiricalScience />',
    );
    expect(componentSource).not.toContain('<script');
    expect(componentSource).not.toContain('client:');
    expect(componentSource).not.toContain('addEventListener');
  });

  it('provides all seven stable sections without a redundant local menu', () => {
    for (const [id, heading] of sections) {
      expect(componentSource).toContain(`id="${id}"`);
      expect(componentSource).toContain(heading);
    }

    expect(componentSource).toContain('id="cosmic-breath-empirical-context"');
    expect(componentSource).not.toContain('class="empirical-nav"');
    expect(componentSource).toContain(
      'aria-label="Primary-Source Empirical Context continuation"',
    );
    expect(componentSource).toContain('href="#cosmic-breath-page-guide"');
    expect(componentSource).toContain('href="#cosmic-breath-page-top"');
  });

  it('states the approved observation, age, and observable-universe boundaries', () => {
    expect(compactComponent).toContain(
      'Observations of distant supernovae and other cosmological probes support the conclusion that cosmic expansion is accelerating.',
    );
    expect(compactComponent).toContain(
      'The universe is approximately 13.8 billion years old under the standard cosmological model.',
    );
    expect(compactComponent).toContain(
      'The observable universe is roughly 92 billion light-years across in present comoving-distance terms. This is not the known size of the entire universe and is not interchangeable with its age.',
    );
    expect(compactComponent).toContain(
      'Neither its age nor its comoving extent defines the placement or scale of the CU Observable Universe marker.',
    );
  });

  it('keeps dark energy unknown and preserves the exact DESI qualification', () => {
    expect(compactComponent).toContain(
      'Dark energy is the name used for the still-unknown cause or model component associated with accelerated expansion.',
    );
    expect(compactComponent).toContain(
      'DESI’s current results provide model- and dataset-dependent hints that dark energy may evolve, but they do not establish that conclusion.',
    );
    expect(compactComponent).toContain('Evolving evidence');
    expect(compactComponent).toContain(
      'DESI is not evidence for CU Dark Energy, CU Anti-Dark Energy, compression, a complete Cosmic Breath, or a future universal reset.',
    );
  });

  it('keeps cosmic futures conditional and CU Anti-Dark Energy internal', () => {
    expect(compactComponent).toContain(
      'Heat death is one conditional long-term scenario, not a directly observed or inevitable future event.',
    );
    expect(compactComponent).toContain(
      'CU-defined Anti-Dark Energy is an internal CU proposition. No standard observed scientific component equivalent to CU Anti-Dark Energy was identified in the reviewed authoritative literature.',
    );
    expect(compactComponent).not.toContain(
      'CU Anti-Dark Energy is a scientifically discovered force',
    );
  });

  it('states the SI, Planck-time, and tetration definitions and limits', () => {
    expect(compactComponent).toContain(
      'The SI second is defined by fixing the cesium-133 unperturbed ground-state hyperfine-transition frequency at exactly 9,192,631,770 hertz.',
    );
    expect(compactComponent).toContain(
      'Planck time is approximately 5.391247(60) × 10⁻⁴⁴ seconds and is a derived natural unit—not an experimentally proven smallest interval.',
    );
    expect(compactComponent).toContain('CU ATOM is not identified with Planck time.');
    expect(compactComponent).toContain(
      'In the convention used here, 2↑↑n denotes a right-associated power tower of 2s with height n. The expression has no inherent unit of time.',
    );
    expect(compactComponent).toContain(
      'The full value of 2↑↑65,536 is not calculated on this page.',
    );
  });

  it('preserves quantum-foam, holography, black-hole, and memory boundaries', () => {
    expect(compactComponent).toContain(
      'quantum foam has not been directly detected as a universal chronological stage.',
    );
    expect(compactComponent).toContain(
      'This does not establish that the observed universe is literally a hologram or that consciousness is holographic.',
    );
    expect(compactComponent).toContain(
      'Evaporation time depends strongly on mass, spin, particle content, accretion, and environment; there is no universal scientific “black-hole age.”',
    );
    expect(compactComponent).toContain(
      'Holography, black-hole information research, and cyclic cosmological models do not currently establish semantic memory, civilization records, consciousness, or ethical learning surviving between cosmic cycles.',
    );
  });

  it('draws an explicit Where CU Begins boundary', () => {
    expect(compactComponent).toContain(
      'Cosmic Universalism uses empirical science as a point of comparison, while the complete Cosmic Breath remains a CU theoretical proposition.',
    );
    expect(compactComponent).toContain(
      'Any continuity or memory across Cosmic Breaths is a CU proposition, not a conclusion currently established by physics.',
    );
    expect(compactComponent).toContain(
      'the declared 3.108-trillion-year Cosmic Breath as an empirical cycle;',
    );
    expect(compactComponent).toContain('CU ATOM as Planck time.');
  });

  it('uses numbered citations and a semantic ordered source list', () => {
    expect(componentSource).toContain('<ol class="source-list">');
    expect(componentSource).toContain('id={`science-source-${source.id}`}');
    expect(componentSource).toContain('aria-label={`Return to citation ${source.id}`}');

    const referencedIds = [
      ...componentSource.matchAll(/href="#science-source-(\d+)"/g),
    ].map((match) => Number(match[1]));
    const sourceIds = cosmicBreathEmpiricalSources.map((source) => source.id);

    expect(new Set(referencedIds)).toEqual(new Set(sourceIds));
    expect(new Set(sourceIds).size).toBe(sourceIds.length);
  });

  it('opens only external scientific sources in new tabs with accessible disclosure', () => {
    expect(componentSource).toContain(
      '<a href={source.url} target="_blank" rel="noopener noreferrer">',
    );
    expect(componentSource).toContain(
      '<span class="visually-hidden"> (opens in a new tab)</span>',
    );
    expect(componentSource).toContain('.visually-hidden {');

    for (const internalHref of [
      '#science-source-1',
      '#cosmic-breath-page-guide',
      '#cosmic-breath-page-top',
    ]) {
      const matchingLines = componentSource
        .split('\n')
        .filter((line) => line.includes(internalHref));
      expect(matchingLines.length).toBeGreaterThan(0);
      expect(matchingLines.every((line) => !line.includes('target="_blank"'))).toBe(true);
    }

    expect(componentSource).toContain('href={`#${sourceReturnTargets[source.id]}`}');
    expect(componentSource).not.toMatch(
      /href=\{`#\$\{sourceReturnTargets\[source\.id\]\}`\}[^>]*target="_blank"/,
    );
  });

  it('uses a responsive two-column source grid with source cards filling their cells', () => {
    expect(componentSource).toMatch(
      /\.source-list\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/s,
    );
    expect(componentSource).toMatch(
      /\.source-list > li:last-child\s*\{[^}]*grid-column:\s*1 \/ -1/s,
    );
    expect(componentSource).toMatch(
      /@media \(max-width: 48rem\)[\s\S]*?\.source-list\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/,
    );
    const sourceCardRule = componentSource.match(/\.source-list > li\s*\{([^}]*)\}/s)?.[1];
    expect(sourceCardRule).toBeDefined();
    expect(sourceCardRule).toMatch(/max-width:\s*none/);
    expect(sourceCardRule).not.toMatch(/(?:^|;)\s*width\s*:/);
  });

  it('keeps public source metadata complete, unique, and valid', () => {
    const urls = new Set<string>();
    const validClasses = new Set<string>(empiricalSourceClasses);

    expect(cosmicBreathEmpiricalSources).toHaveLength(19);

    for (const source of cosmicBreathEmpiricalSources) {
      expect(source.id).toBeGreaterThan(0);
      expect(source.title.trim()).not.toBe('');
      expect(source.authors.trim()).not.toBe('');
      expect(source.organization.trim()).not.toBe('');
      expect(source.url).toMatch(/^https:\/\//);
      expect(validClasses.has(source.sourceClass)).toBe(true);
      expect(urls.has(source.url)).toBe(false);
      urls.add(source.url);
    }
  });

  it('displays every approved public source class', () => {
    for (const sourceClass of empiricalSourceClasses) {
      expect(cosmicBreathEmpiricalSources.some((source) => source.sourceClass === sourceClass)).toBe(
        true,
      );
    }
  });

  it('contains no governance metadata, sealed digests, or repository-local paths', () => {
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
      '/Users/cosmos/',
      'website/src/',
    ];

    for (const term of forbiddenTerms) {
      expect(componentSource).not.toContain(term);
    }
  });

  it('contains none of the prohibited unsupported public claims', () => {
    const prohibitedClaims = [
      'Dark energy has been directly detected as a substance.',
      'DESI proved that dark energy changes with time.',
      'Anti-Dark Energy is a scientifically discovered force.',
      'Science confirms that the universe will contract.',
      'The universe is known to repeat in cycles.',
      'The universe is exactly 13.8 billion years old.',
      'The entire universe is 92 billion light-years wide.',
      'Planck time is the smallest possible unit of time.',
      'CU ATOM is Planck time.',
      'Tetration is a physical clock.',
      'Quantum foam has been observed.',
      'The universe will certainly end in heat death.',
      'Entropy becomes exactly zero at a universal reset.',
      'Physics proves the universe is a hologram.',
      'Holography proves consciousness survives.',
      'All black holes have the same lifespan.',
      'Civilizations retain memories across cosmic cycles.',
      'Empirical science validates the complete Cosmic Breath chronology.',
    ];

    for (const claim of prohibitedClaims) {
      expect(compactComponent).not.toContain(claim);
    }
  });

  it('does not duplicate the 51-state authority', () => {
    expect(componentSource).not.toContain('structuralStateId');
    expect(componentSource).not.toContain('orderedTomStates');
    expect(componentSource).not.toContain('data-state-index');
    expect(componentSource.match(/51-state/g)).toHaveLength(1);
  });
});
