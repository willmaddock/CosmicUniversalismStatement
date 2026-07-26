import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const headingSource = readSource(
  '../../../components/page-system/EditorialSectionHeading.astro',
);
const calloutSource = readSource(
  '../../../components/page-system/EpistemicCallout.astro',
);
const provenanceSource = readSource(
  '../../../components/page-system/SourceProvenancePanel.astro',
);
const continuationSource = readSource(
  '../../../components/page-system/ContinuationNavigation.astro',
);
const contentSectionSource = readSource(
  '../../../components/ContentSection.astro',
);
const tokensSource = readSource('../../../styles/tokens.css');

describe('shared editorial page-system primitives', () => {
  it('keeps the editorial heading semantic and configurable', () => {
    expect(headingSource).toContain(
      "type HeadingLevel = 'h2' | 'h3' | 'h4';",
    );
    expect(headingSource).toContain("headingLevel = 'h2'");
    expect(headingSource).toContain('const Heading = headingLevel;');
    expect(headingSource).toContain('<Heading id={headingId}>{title}</Heading>');
    expect(headingSource).toContain('sectionNumber');
    expect(headingSource).toContain('eyebrow');
    expect(headingSource).toContain('headingId');
    expect(headingSource).toContain('overflow-wrap: anywhere');
    expect(headingSource).not.toContain('<script');
  });

  it('renders callout meaning as visible text for every supported kind', () => {
    expect(calloutSource).toContain(
      "type CalloutKind = 'boundary' | 'method' | 'caution' | 'context';",
    );
    expect(calloutSource).toContain(
      "type CalloutElement = 'aside' | 'section';",
    );
    expect(calloutSource).toContain(
      '<p class="epistemic-callout__label">{label}</p>',
    );
    expect(calloutSource).toContain('aria-label={label}');
    expect(calloutSource).toContain('<slot />');
    expect(calloutSource).not.toContain('<script');
  });

  it('provides route-owned provenance content and safe destinations', () => {
    expect(provenanceSource).toContain('<section');
    expect(provenanceSource).toContain('aria-labelledby={headingId}');
    expect(provenanceSource).toContain('<slot />');
    expect(provenanceSource).toContain("kind: 'internal'");
    expect(provenanceSource).toContain('href={sitePath(link.path)}');
    expect(provenanceSource).toContain("kind: 'external'");
    expect(provenanceSource).toContain('target="_blank"');
    expect(provenanceSource).toContain('rel="noopener noreferrer"');
    expect(provenanceSource).toContain('{link.disclosure}');
    expect(provenanceSource).not.toContain('<script');
  });

  it('renders a semantic, base-path-safe same-tab continuation', () => {
    expect(continuationSource).toContain(
      '<nav class="continuation-navigation" aria-label={ariaLabel}>',
    );
    expect(continuationSource).toContain('<a href={sitePath(path)}>');
    expect(continuationSource).toContain('<strong>{title}</strong>');
    expect(continuationSource).toContain('<span>{description}</span>');
    expect(continuationSource).toContain('Forward →');
    expect(continuationSource).not.toContain('target=');
    expect(continuationSource).not.toContain('<script');
  });

  it('adds semantic tokens without changing established token values', () => {
    const establishedTokens = [
      '--color-deep-navy-black: #060b17;',
      '--color-white-gold: #f4e8c1;',
      '--color-cyan: #58d9ec;',
      '--color-translucent-blue: rgb(62 124 189 / 24%);',
      '--color-amber: #e8a84a;',
      '--color-muted-gray: #8f99a8;',
      '--measure-readable: 70ch;',
    ];
    establishedTokens.forEach((token) => {
      expect(tokensSource).toContain(token);
    });

    [
      '--color-panel-quiet:',
      '--color-panel-feature:',
      '--color-border-quiet:',
      '--color-border-section:',
      '--color-border-feature:',
      '--color-callout-boundary-accent:',
      '--color-callout-method-accent:',
      '--color-callout-caution-accent:',
      '--color-callout-context-accent:',
      '--space-editorial-section:',
    ].forEach((token) => {
      expect(tokensSource).toContain(token);
    });
  });

  it('keeps ContentSection variants backward-compatible', () => {
    expect(contentSectionSource).toContain(
      "variant?: 'plain' | 'panel' | 'notice' | 'card';",
    );
    expect(contentSectionSource).toContain("variant = 'plain'");
    expect(contentSectionSource).toContain("headingLevel = 'h2'");
    expect(contentSectionSource).toContain('const Heading = headingLevel;');
    for (const variant of ['panel', 'notice', 'card']) {
      expect(contentSectionSource).toContain(`.content-section--${variant}`);
    }
  });
});
