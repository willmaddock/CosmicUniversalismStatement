import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const backToTopSource = readFileSync(
  fileURLToPath(
    new URL(
      '../../../components/page-system/BackToTop.astro',
      import.meta.url,
    ),
  ),
  'utf8',
);

const cuciiSource = readFileSync(
  fileURLToPath(
    new URL('../../../pages/cu-intelligence.astro', import.meta.url),
  ),
  'utf8',
);

describe('Back-to-top enhanced behavior', () => {
  it('scrolls the real window to the true document top', () => {
    expect(backToTopSource).toContain(
      "control.addEventListener('click', returnToTop)",
    );
    expect(backToTopSource).toContain('event.preventDefault()');
    expect(backToTopSource).toContain('window.scrollTo({');
    expect(backToTopSource).toContain('top: 0');
    expect(backToTopSource).toContain('left: 0');
  });

  it('respects reduced motion', () => {
    expect(backToTopSource).toContain(
      "'(prefers-reduced-motion: reduce)'",
    );
    expect(backToTopSource).toContain(
      "behavior: reducedMotionQuery.matches ? 'auto' : 'smooth'",
    );
  });

  it('moves focus to the main content without causing another scroll', () => {
    expect(backToTopSource).toContain(
      "document.getElementById('main-content')",
    );
    expect(backToTopSource).toContain(
      'mainContent?.focus({ preventScroll: true })',
    );
  });

  it('supports Space activation in addition to native anchor Enter activation', () => {
    expect(backToTopSource).toContain(
      "control.addEventListener('keydown'",
    );
    expect(backToTopSource).toContain("event.key !== ' '");
    expect(backToTopSource).toContain('control.click()');
  });

  it('retains the fragment link as a no-JavaScript fallback', () => {
    expect(backToTopSource).toContain(
      '<a class="back-to-top" href={`#${targetId}`}>{label}</a>',
    );
  });

  it('reveals the CUCII control after the page guide rather than the full Prompt Studio', () => {
    expect(cuciiSource).toContain(
      'revealAfterId="cucii-page-guide"',
    );
    expect(cuciiSource).not.toContain(
      'revealAfterId="prompt-studio-feature"',
    );
  });
});
