import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const digest = (source: string) =>
  createHash('sha256').update(source).digest('hex');

const headerSource = readSource('../../../components/Header.astro');
const cosmicBreathSource = readSource('../../../pages/cosmic-breath.astro');
const empiricalScienceSource = readSource(
  '../../../components/CosmicBreathEmpiricalScience.astro',
);
const cuTimeSource = readSource('../../../components/CUTimeConverter.astro');
const cuIntelligenceSource = readSource(
  '../../../pages/cu-intelligence.astro',
);
const researchSource = readSource(
  '../../../components/research/ResearchObservatory.astro',
);
const legacyConverterSource = readSource(
  '../../../../public/cosmic_converter/v3_0_0/cosmic_breath_time_converter_v3_0_1.html',
);

const scriptBody = (source: string) =>
  source.match(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/)?.[1] ?? '';

describe('VH-6A confirmed defect corrections', () => {
  it('advances every authorized changed source to its reviewed digest', () => {
    expect(
      new Map([
        ['Header.astro', digest(headerSource)],
        ['cosmic-breath.astro', digest(cosmicBreathSource)],
        ['CosmicBreathEmpiricalScience.astro', digest(empiricalScienceSource)],
        ['CUTimeConverter.astro', digest(cuTimeSource)],
        ['cu-intelligence.astro', digest(cuIntelligenceSource)],
        ['ResearchObservatory.astro', digest(researchSource)],
        ['legacy converter', digest(legacyConverterSource)],
      ]),
    ).toEqual(
      new Map([
        ['Header.astro', '2b1e7cf1dfebfc1e98ba43c6539c5006fff5b96704a14856996994fa6909956f'],
        ['cosmic-breath.astro', 'a0ed45ef3aff558f297bff50893dc1b3ca82e75cc7756b16be050bf7a2a511bc'],
        ['CosmicBreathEmpiricalScience.astro', '960a1bb7f3ec0e8f5bd5f58351d4b96ff4cc1b658355cdacd82b07b7ad632370'],
        ['CUTimeConverter.astro', '771db5fef68427b3c95b2f29036225f47be7e7792d5e0817f456f37639de8bef'],
        ['cu-intelligence.astro', '74470433295ab4e7082db998000e5e923f6c311e1b1c9cec1fa22c44f669570e'],
        ['ResearchObservatory.astro', 'a9a3b147ba0520379663d53981630899757672b4091274a1cdac09714429f1a0'],
        ['legacy converter', 'ac5f7dbc028389281864f9d3cbe5fe9e524a5dbe9e55d56a7b9a6745d0adb344'],
      ]),
    );
  });

  it('completes the eight-route continuation sequence at Cosmic Breath', () => {
    expect(cosmicBreathSource.match(/<ContinuationNavigation\b/g)).toHaveLength(1);
    expect(cosmicBreathSource).toContain('title="Continue to CU-Time"');
    expect(cosmicBreathSource).toContain(
      'description="A proposed deep-time coordinate system"',
    );
    expect(cosmicBreathSource).toContain('path="cu-time/"');
    expect(cosmicBreathSource.indexOf('<CosmicBreathEmpiricalScience />')).toBeLessThan(
      cosmicBreathSource.indexOf('<ContinuationNavigation'),
    );
  });

  it('keeps both tools inert in static output and reveals them only after initialization', () => {
    expect(cuTimeSource).toContain(
      'JavaScript is required to use the CU-Time converter. The explanatory material',
    );
    expect(cuTimeSource).toMatch(
      /<section[\s\S]*?hidden[\s\S]*?data-cu-time-interface/,
    );
    expect(cuTimeSource).toContain('converterInterface.hidden = false;');
    expect(cuTimeSource).toContain('converterFallback.hidden = true;');
    expect(cuTimeSource.indexOf('converterInterface.hidden = false;')).toBeGreaterThan(
      cuTimeSource.indexOf("Object.values(reverseReference).every(Boolean)"),
    );

    expect(cuIntelligenceSource).toContain(
      'JavaScript is required to generate a prompt in the Prompt Studio. The',
    );
    expect(cuIntelligenceSource).toMatch(
      /<section[\s\S]*?id="prompt-studio"[\s\S]*?hidden[\s\S]*?data-cucii-studio-interface/,
    );
    expect(cuIntelligenceSource).toContain(
      'if (form && studioInterface && studioFallback)',
    );
    expect(cuIntelligenceSource).toContain('studioInterface.hidden = false;');
    expect(cuIntelligenceSource).toContain('studioFallback.hidden = true;');
  });

  it('adds only initialization behavior to the protected CU-Time and CUCII scripts', () => {
    const normalizedCuTimeScript = scriptBody(cuTimeSource)
      .replace(
        "\n  const converterInterface = document.querySelector<HTMLElement>('[data-cu-time-interface]');\n  const converterFallback = document.querySelector<HTMLElement>('[data-cu-time-fallback]');",
        '',
      )
      .replace(
        /\n\n  if \(\n    converterInterface[\s\S]*?    converterFallback\.hidden = true;\n  \}\n$/,
        '\n',
      );
    expect(digest(normalizedCuTimeScript)).toBe(
      'c2618e30a1d47e9f4e87de8d666ba48db37353fc3dca2c81278c4fafa7c305bc',
    );

    const normalizedCuciiScript = scriptBody(cuIntelligenceSource)
      .replace(
        "\n  const studioInterface = document.querySelector<HTMLElement>('[data-cucii-studio-interface]');\n  const studioFallback = document.querySelector<HTMLElement>('[data-cucii-studio-fallback]');\n  if (form && studioInterface && studioFallback) {",
        '\n  if (form) {',
      )
      .replace(
        '\n    studioInterface.hidden = false;\n    studioFallback.hidden = true;',
        '',
      );
    expect(digest(normalizedCuciiScript)).toBe(
      'a9bc063dd2f9e6bcfab199bda664eb9b13176a58f7c3bd70d8d87a69f4d73562',
    );
  });

  it('uses shrink-safe CU-Time controls and single-column narrow tracks', () => {
    expect(cuTimeSource).toContain('box-sizing: border-box');
    expect(cuTimeSource).toContain('max-width: 100%');
    expect(cuTimeSource).toContain('overflow-wrap: anywhere');
    expect(cuTimeSource).toMatch(
      /@media \(max-width: 42rem\)[\s\S]*?grid-template-columns: minmax\(0, 1fr\)/,
    );
    expect(cuTimeSource).not.toMatch(
      /@media \(max-width: 42rem\)[\s\S]*?grid-template-columns: 1fr;/,
    );
  });

  it('moves focus off the Menu control before hiding it on desktop', () => {
    expect(headerSource).toContain(
      "header.querySelector<HTMLAnchorElement>('.site-identity')",
    );
    expect(headerSource).toContain(
      'const focusWasOnToggle = document.activeElement === toggle;',
    );
    expect(headerSource).toContain('identity.focus({ preventScroll: true });');
    expect(headerSource).toContain('toggle.hidden = true;');
    expect(headerSource.indexOf('identity.focus({ preventScroll: true });')).toBeLessThan(
      headerSource.indexOf('toggle.hidden = true;'),
    );
    expect(headerSource).toContain('toggle.hidden = false;');
    expect(headerSource).not.toContain('.site-header__menu-toggle:focus {');
  });

  it('corrects the platform heading and labels the two confirmed landmarks', () => {
    expect(cuIntelligenceSource).toContain(
      '<h4 id="platform-links-title">Choose an AI Platform</h4>',
    );
    expect(cuIntelligenceSource).not.toContain(
      '<h5 id="platform-links-title">Choose an AI Platform</h5>',
    );
    expect(empiricalScienceSource).toContain(
      '<aside class="cu-boundary-inline" aria-label="CU proposition">',
    );
    expect(researchSource).toMatch(
      /<template data-detail-template=\{overview\.id\}>[\s\S]*?aria-label=\{`\$\{overview\.title\} epistemic boundary`\}/,
    );
    expect(researchSource).toMatch(
      /<template data-detail-template=\{node\.id\}>[\s\S]*?aria-label=\{`\$\{node\.title\} epistemic boundary`\}/,
    );
  });

  it('removes only the parsed-CSV debug statement from the legacy asset', () => {
    expect(legacyConverterSource).not.toContain(
      "console.log('Parsed CSV data:', results.data);",
    );
    expect(legacyConverterSource).toContain('Papa.parse(csvText, {');
    expect(legacyConverterSource).toContain('setResults(results.data);');

    const restored = legacyConverterSource.replace(
      '                        complete: (results) => {\n',
      "                        complete: (results) => {\n                            console.log('Parsed CSV data:', results.data);\n",
    );
    expect(digest(restored)).toBe(
      'f7c71928d65cad54147e6f04fa2d3e0af0c5d84ecde192c0b86d09da864c0999',
    );
  });
});
