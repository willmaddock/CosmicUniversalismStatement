import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import rawStructuralLedger from '../../../data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json';

const diagramPath = fileURLToPath(new URL(
  '../../../components/CosmicBreathDiagram.astro',
  import.meta.url,
));
const routePath = fileURLToPath(new URL(
  '../../../pages/cosmic-breath.astro',
  import.meta.url,
));
const educationPath = fileURLToPath(new URL(
  '../../../components/CosmicBreathEducation.astro',
  import.meta.url,
));
const structuralLedgerPath = fileURLToPath(new URL(
  '../../../data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json',
  import.meta.url,
));
const expectedStructuralDigest =
  'dcc34d5b7d9e32afb8d0cb97b029a12e0025959095a21b4168d59aff575da825';

const diagramSource = readFileSync(diagramPath, 'utf8');
const routeSource = readFileSync(routePath, 'utf8');
const educationSource = readFileSync(educationPath, 'utf8');
const publicPageSource = `${routeSource}\n${educationSource}`;
const markerSource = diagramSource.match(
  /<g\s+[\s\S]*?class="cosmic-breath-diagram__observable-marker"[\s\S]*?<\/g>/,
)?.[0] ?? '';

describe('shared Observable Universe marker', () => {
  it('describes the marker as a non-selectable reference between compression states', () => {
    expect(diagramSource).toContain(
      'marker positioned between compression-btom and compression-ctom',
    );
    expect(diagramSource).toContain('The marker is not');
    expect(diagramSource).toContain('one of the 51 TOM states');
    expect(`${diagramSource}\n${routeSource}`).not.toMatch(
      /marker (?:is |belongs |identifies the Observable Universe )?(?:within|inside|occupying) ctom/i,
    );
  });

  it('keeps the route wording explicit and epistemically bounded', () => {
    expect(publicPageSource).toContain('Observable Universe Marker');
    expect(publicPageSource.match(/non-selectable Observable Universe marker/g)).toHaveLength(2);
    expect(publicPageSource.match(/not one of the 51 selectable TOM states/g)).toHaveLength(2);
    expect(publicPageSource).toContain(
      'This placement is a CU structural proposition — not an empirical measurement.',
    );
  });

  it('derives the marker from unchanged btom and ctom radii', () => {
    expect(diagramSource).toContain('const compressionBtomRadius = 142;');
    expect(diagramSource).toContain('const compressionCtomRadius = 196;');
    expect(diagramSource).toContain(
      'const observableMarkerRadius = (compressionBtomRadius + compressionCtomRadius) / 2;',
    );
    expect(diagramSource).toContain(
      'cosmic-breath-diagram__boundary--btom" cx="320" cy="300" r="142"',
    );
    expect(diagramSource).toContain(
      'cosmic-breath-diagram__boundary--ctom" cx="320" cy="300" r="196"',
    );
  });

  it('gives the marker no selectable or structural semantics', () => {
    expect(markerSource).not.toBe('');
    expect(markerSource).not.toMatch(
      /structuralStateId|cycleIndex|phaseIndex|data-state|data-select|tabindex|<button|<a\s|role=/i,
    );
    expect(diagramSource).toContain('pointer-events: none;');
  });

  it('preserves the canonical 51-state structure and digest', () => {
    expect(rawStructuralLedger.states).toHaveLength(51);
    const digest = createHash('sha256').update(readFileSync(structuralLedgerPath)).digest('hex');
    expect(digest).toBe(expectedStructuralDigest);
    expect(rawStructuralLedger.transitions).toHaveLength(2);
  });
});
