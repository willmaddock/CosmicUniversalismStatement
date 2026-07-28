import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const featureSource = readSource(
  '../../../components/page-system/FullWidthFeaturePanel.astro',
);
const guideSource = readSource(
  '../../../components/page-system/PageGuide.astro',
);
const backToTopSource = readSource(
  '../../../components/page-system/BackToTop.astro',
);

describe('long-page shared primitives', () => {
  it('provides a route-owned, semantic, contained feature shell', () => {
    expect(featureSource).toContain('<section');
    expect(featureSource).toContain('aria-label={title ? undefined : ariaLabel}');
    expect(featureSource).toContain('aria-labelledby={title ? headingId : undefined}');
    expect(featureSource).toContain("type HeadingLevel = 'h2' | 'h3' | 'h4';");
    expect(featureSource).toContain('<Heading id={headingId}>{title}</Heading>');
    expect(featureSource).toContain('if (title && !headingId)');
    expect(featureSource).toContain("if (!title && !ariaLabel?.trim())");
    expect(featureSource).toContain('<slot />');
    expect(featureSource).toContain('width: min(112rem, calc(100vw');
    expect(featureSource).toContain('min-width: 0');
    expect(featureSource).toContain('overflow: hidden');
    expect(featureSource).toContain('@media (max-width: 36rem)');
    expect(featureSource).not.toContain('<script');
    expect(featureSource).not.toMatch(/CU-Time|Cosmic Breath|converter/i);
  });

  it('renders grouped, route-owned fragment navigation without JavaScript', () => {
    expect(guideSource).toContain('<nav aria-label={ariaLabel}>');
    expect(guideSource).toContain('groups.map((group)');
    expect(guideSource).toContain('group.links.map((link)');
    expect(guideSource).toContain('href={`#${link.targetId}`}');
    expect(guideSource).toContain('fragmentIdPattern');
    expect(guideSource).toContain('!headingId');
    expect(guideSource).toContain('link.label.trim()');
    expect(guideSource).toContain('min-height: 2.75rem');
    expect(guideSource).toContain('repeat(auto-fit');
    expect(guideSource).not.toContain('<script');
    expect(guideSource).not.toMatch(/CU-Time|Cosmic Breath|converter/i);
  });

  it('keeps Back to top a real anchor with progressive enhancement', () => {
    expect(backToTopSource).toContain('href={`#${targetId}`}');
    expect(backToTopSource).toContain("label = 'Back to top'");
    expect(backToTopSource).toContain('control.hidden = true');
    expect(backToTopSource).toContain('control.hidden = !isVisible');
    expect(backToTopSource).toContain('.back-to-top[hidden]');
    expect(backToTopSource).toContain('display: none');
    expect(backToTopSource).toContain("window.addEventListener('scroll'");
    expect(backToTopSource).toContain('requestAnimationFrame');
    expect(backToTopSource).toContain(
      'revealAfter.getBoundingClientRect().bottom < 0',
    );
    expect(backToTopSource).toContain('document.activeElement === control');
    expect(backToTopSource).toContain('env(safe-area-inset-right)');
    expect(backToTopSource).toContain('env(safe-area-inset-bottom)');
    expect(backToTopSource).toContain('@media (prefers-reduced-motion: reduce)');
    expect(backToTopSource).toContain('scroll-behavior: auto');
    expect(backToTopSource).not.toContain('scrollIntoView');
    expect(backToTopSource).not.toContain('history.');
    expect(backToTopSource).not.toContain('.focus(');
    expect(backToTopSource).not.toMatch(/CU-Time|Cosmic Breath|converter/i);
  });
});
