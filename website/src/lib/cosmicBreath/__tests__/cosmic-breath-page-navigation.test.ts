import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(fileURLToPath(new URL(relativePath, import.meta.url)), 'utf8');

const routeSource = readSource('../../../pages/cosmic-breath.astro');
const explorerSource = readSource('../../../components/CosmicBreathCycleExplorer.astro');
const guideSource = readSource('../../../components/CosmicBreathPageGuide.astro');
const educationSource = readSource('../../../components/CosmicBreathEducation.astro');
const theorySource = readSource('../../../components/CosmicBreathTheoryAndMethod.astro');
const empiricalSource = readSource('../../../components/CosmicBreathEmpiricalScience.astro');

const groups = [
  {
    title: 'Cosmic Breath Field Guide',
    label: 'Explore the framework',
    container: 'cosmic-breath-field-guide',
    targets: [
      'cosmic-breath-at-a-glance',
      'reading-the-structural-diagram',
      'tom-names-and-positions',
      'exponents-tetration-and-duration',
      'expansion-compression-and-guarded-boundaries',
      'observable-universe-marker',
      'structural-and-epistemic-boundaries',
    ],
  },
  {
    title: 'Cosmic Breath Theory and Method',
    label: 'Inside the CU model',
    container: 'cosmic-breath-theory-method',
    targets: [
      'cu-dark-energy',
      'cu-anti-dark-energy',
      'tom-duration-representation',
      'memory-ethics-reset-seed',
      'sources-and-method',
    ],
  },
  {
    title: 'Primary-Source Empirical Context',
    label: 'Empirical Science and Cosmic Universalism',
    container: 'cosmic-breath-empirical-context',
    targets: [
      'empirical-observations',
      'empirical-open-questions',
      'scientific-cosmic-futures',
      'scientific-units-notation',
      'quantum-holography-information',
      'where-cu-begins',
      'scientific-sources',
    ],
  },
] as const;

const targetIds = groups.flatMap((group) => [...group.targets]);
const renderedSources = [routeSource, educationSource, theorySource, empiricalSource];

describe('Cosmic Breath page navigation', () => {
  it('starts the route and explorer content with an H1-to-H2 heading progression', () => {
    expect(explorerSource).toContain(
      '<h2 id={`${explorerId}-title`}>Explore the structural Cosmic Breath cycle</h2>',
    );
    expect(explorerSource).not.toContain(
      '<h3 id={`${explorerId}-title`}>Explore the structural Cosmic Breath cycle</h3>',
    );
  });

  it('renders one unified guide immediately after the introduction and before LP-1', () => {
    expect(routeSource).toContain(
      "import CosmicBreathPageGuide from '../components/CosmicBreathPageGuide.astro';",
    );
    expect(routeSource).toContain(
      '</section>\n\n  <CosmicBreathPageGuide />\n  <CosmicBreathEducation />',
    );
    expect(routeSource.match(/<CosmicBreathPageGuide \/>/g)).toHaveLength(1);
    expect(guideSource).toContain('id="cosmic-breath-page-guide"');
    expect(guideSource).toContain('aria-label="Cosmic Breath page guide"');
    expect(guideSource).toContain('Explore the complete page');
  });

  it('contains three labeled groups and exactly 19 unique section links', () => {
    for (const group of groups) {
      expect(guideSource).toContain(`title: '${group.title}'`);
      expect(guideSource).toContain(`label: '${group.label}'`);
    }

    expect(targetIds).toHaveLength(19);
    expect(new Set(targetIds).size).toBe(19);

    for (const target of targetIds) {
      expect(guideSource).toContain(`['${target}',`);
    }
  });

  it('resolves every guide target exactly once in the rendered component set', () => {
    for (const target of targetIds) {
      const occurrences = renderedSources.reduce(
        (count, source) => count + (source.match(new RegExp(`id="${target}"`, 'g'))?.length ?? 0),
        0,
      );
      expect(occurrences, target).toBe(1);
    }
  });

  it('removes all three redundant component-local navigation menus', () => {
    expect(educationSource).not.toContain('class="topic-nav"');
    expect(theorySource).not.toContain('class="theory-nav"');
    expect(empiricalSource).not.toContain('class="empirical-nav"');

    expect(educationSource).toContain('Cosmic Breath field guide');
    expect(educationSource).toContain('Explore the framework');
    expect(theorySource).toContain('Cosmic Breath theory and method');
    expect(theorySource).toContain('Inside the CU model');
    expect(empiricalSource).toContain('Primary-source empirical context');
    expect(empiricalSource).toContain('Empirical Science and Cosmic Universalism');
  });

  it('adds unique major-group targets and complete continuation navigation', () => {
    for (const group of groups) {
      expect(renderedSources.filter((source) => source.includes(`id="${group.container}"`))).toHaveLength(
        1,
      );
    }

    expect(educationSource).toContain(
      'aria-label="Cosmic Breath Field Guide continuation"',
    );
    expect(educationSource).toContain('href="#cosmic-breath-page-guide"');
    expect(educationSource).toContain('href="#cosmic-breath-theory-method"');

    expect(theorySource).toContain(
      'aria-label="Cosmic Breath Theory and Method continuation"',
    );
    expect(theorySource).toContain('href="#cosmic-breath-page-guide"');
    expect(theorySource).toContain('href="#cosmic-breath-empirical-context"');

    expect(empiricalSource).toContain(
      'aria-label="Primary-Source Empirical Context continuation"',
    );
    expect(empiricalSource).toContain('href="#cosmic-breath-page-guide"');
    expect(empiricalSource).toContain('href="#cosmic-breath-page-top"');
  });

  it('provides a progressively enhanced same-tab Back to top anchor', () => {
    expect(routeSource).toContain('id="cosmic-breath-page-top"');
    expect(guideSource).toContain(
      '<a class="back-to-top" href="#cosmic-breath-page-top" data-cosmic-breath-back-to-top>',
    );
    expect(guideSource).toContain('Back to top');
    expect(guideSource).toContain(
      'backToTop.hidden = window.scrollY <= pageTop.offsetTop',
    );
    expect(guideSource).toContain(
      "window.addEventListener('scroll', updateBackToTop, { passive: true })",
    );
    expect(guideSource).not.toMatch(
      /href="#cosmic-breath-page-top"[^>]*target="_blank"/,
    );
    expect(guideSource).not.toContain('scroll-behavior: smooth');
    expect(routeSource).toMatch(
      /\.page-top-sentinel\s*\{[^}]*width:\s*1px;[^}]*height:\s*1px;/s,
    );
  });

  it('keeps all master-guide and continuation links in the same tab', () => {
    expect(guideSource).not.toContain('target="_blank"');

    for (const source of [educationSource, theorySource, empiricalSource]) {
      const internalAnchors = source.match(/<a[^>]+href="#[^"]+"[^>]*>/g) ?? [];
      expect(internalAnchors.length).toBeGreaterThan(0);
      expect(internalAnchors.every((anchor) => !anchor.includes('target="_blank"'))).toBe(
        true,
      );
    }
  });

  it('introduces no duplicate static IDs across route and lower-page components', () => {
    const staticIds = renderedSources.flatMap((source) =>
      [...source.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]),
    );
    const duplicates = staticIds.filter((id, index) => staticIds.indexOf(id) !== index);

    expect(duplicates).toEqual([]);
  });

  it('preserves representative approved CU and scientific boundary wording', () => {
    expect(educationSource).toContain(
      'CU structural proposition — not an empirical measurement',
    );
    expect(theorySource).toContain(
      'ATOM-as-Planck-time language is superseded by the sealed atom record.',
    );
    expect(empiricalSource.replace(/\s+/g, ' ')).toContain(
      'DESI’s current results provide model- and dataset-dependent hints that dark energy may evolve, but they do not establish that conclusion.',
    );
    expect(empiricalSource).toContain(
      'Cosmic Universalism uses empirical science as a point of comparison',
    );
  });
});
