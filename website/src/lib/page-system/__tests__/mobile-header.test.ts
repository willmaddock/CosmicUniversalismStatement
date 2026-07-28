import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const digest = (source: string) =>
  createHash('sha256').update(source).digest('hex');

const headerSource = readSource('../../../components/Header.astro');
const siteSource = readSource('../../../config/site.ts');
const globalSource = readSource('../../../styles/global.css');

const protectedDigests = new Map([
  ['../../../layouts/BaseLayout.astro', 'acd647a19004cd5843047e11970a6d364f8a2e935c0d9302158c173820f25582'],
  ['../../../components/Footer.astro', '685a88746cc80f6bfc97a405886be084ac8dae5b9c91f3637b8f0bfdf4244e63'],
  ['../../../styles/global.css', 'f5bc581e43533441fe2e35ced4420e80d1f59b8ae746a02f27d6d181f94e8e4a'],
  ['../../../styles/tokens.css', 'f8c2260dc066db1fdd05d05777494c180c74c54e9342879d3d8b5d37939ddc54'],
  ['../../../pages/index.astro', 'f78dc57be535dd5bd83eac1aaa98d8882d7b8d04f5df5a304435ce6e6b60fcff'],
  ['../../../pages/framework.astro', '16fcffe3ce0fb7632756626221d3a8a984f85e28c61df717aec3f850dbf3e8ac'],
  ['../../../pages/cosmic-breath.astro', 'a0ed45ef3aff558f297bff50893dc1b3ca82e75cc7756b16be050bf7a2a511bc'],
  ['../../../pages/cu-time.astro', 'dc557e235883cb18b33766d2d818139b58eb202fcd5d0be3daa1909bbae4c0d3'],
  ['../../../pages/cu-intelligence.astro', '6f371ba4beffa261e5183db7795b649354760f9939da323df2f29a280fdc90c7'],
  ['../../../pages/research/index.astro', '01c875f6b5e41a9e0807ced06fa08735e8465bb1fe6f39120cd30d59c7d6d47b'],
  ['../../../pages/media.astro', '00d48d13e4f36382d57620d169b3d265fe06dc76c75d52540149e409aecf28ff'],
  ['../../../pages/about.astro', '1de1e76e3e27c940af0a09ad0923771b9b1b7b5891060ea6c02022812c27b4d5'],
] as const);

describe('compact accessible mobile Header', () => {
  it('preserves one identity, one primary navigation, and one destination tree', () => {
    expect(headerSource.match(/class="site-identity"/g)).toHaveLength(1);
    expect(headerSource.match(/aria-label="Primary navigation"/g)).toHaveLength(1);
    expect(headerSource.match(/primaryNavigation\.map/g)).toHaveLength(1);
    expect(headerSource.match(/class="primary-navigation__list"/g)).toHaveLength(1);
    expect(headerSource.match(/class="site-header__action"/g)).toHaveLength(1);
  });

  it('preserves exact destination labels, registry order, and link contracts', () => {
    const labels = [
      "'Framework'",
      "'Cosmic Breath'",
      "'CU-Time'",
      "'Research'",
      "'Media'",
      "'About'",
      "'GitHub'",
      "'Explore CUCII'",
    ];
    const positions = labels.map((label) => siteSource.indexOf(label));

    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(headerSource).toContain("target={external ? '_blank' : undefined}");
    expect(headerSource).toContain(
      "rel={external ? 'noopener noreferrer' : undefined}",
    );
  });

  it('renders one hidden-until-enhanced native disclosure control', () => {
    expect(headerSource.match(/<button\b/g)).toHaveLength(1);
    expect(headerSource).toContain('type="button"');
    expect(headerSource).toContain('aria-expanded="false"');
    expect(headerSource).toContain('aria-controls="site-navigation"');
    expect(headerSource).toContain('id="site-navigation"');
    expect(headerSource).toContain('<span>Menu</span>');
    expect(headerSource).toContain('hidden');
    expect(headerSource).toContain("header.dataset.enhanced = 'true'");
    expect(headerSource).toContain('toggle.hidden = false');
  });

  it('keeps complete navigation server-rendered and available without JavaScript', () => {
    expect(headerSource).toContain('primaryNavigation.map');
    expect(headerSource).toContain('{primaryAction.label}');
    expect(headerSource).not.toMatch(/innerHTML|insertAdjacentHTML|createElement/);
    expect(headerSource).not.toMatch(
      /<div[^>]+data-site-navigation[^>]+hidden/,
    );
    expect(headerSource).toContain(
      ".site-header[data-enhanced='true'] .site-header__navigation-region[hidden]",
    );
  });

  it('removes closed links from interaction and exposes them when open', () => {
    expect(headerSource).toContain(
      'navigation.hidden = mobileHeaderQuery.matches ? !open : false;',
    );
    expect(headerSource).toContain(
      "toggle.setAttribute('aria-expanded', String(open));",
    );
    expect(headerSource).toContain('setMenuOpen(!menuIsOpen(), true);');
    expect(headerSource).not.toContain('inert');
  });

  it('supports Escape closure and safe focus restoration without focusing on open', () => {
    expect(headerSource).toContain("event.key !== 'Escape'");
    expect(headerSource).toContain('setMenuOpen(false, true);');
    expect(headerSource).toContain('toggle.focus();');
    expect(headerSource).toContain(
      'if (!open && restoreFocusIfNeeded && focusWasInside)',
    );
    expect(headerSource).not.toMatch(
      /querySelector[^;]+(?:primary-navigation|\\ba\\b)[^;]*\.focus\(/,
    );
    expect(headerSource).not.toMatch(/focus-trap|tabindex=["']-1["']/i);
  });

  it('uses the established breakpoint with desktop restoration and deterministic collapse', () => {
    expect(headerSource).toContain(
      "window.matchMedia('(max-width: 64rem)')",
    );
    expect(headerSource).toContain('@media (max-width: 64rem)');
    expect(headerSource).toContain('navigation.hidden = false;');
    expect(headerSource).toContain('setMenuOpen(false);');
    expect(headerSource).toContain(
      "mobileHeaderQuery.addEventListener('change', synchronizeForViewport)",
    );
    expect(headerSource).toContain(
      'const focusWasOnToggle = document.activeElement === toggle;',
    );
    expect(headerSource).toContain('identity.focus({ preventScroll: true });');
    expect(headerSource).toContain('toggle.hidden = true;');
    expect(headerSource).not.toContain('.site-header__menu-toggle:focus {');
  });

  it('provides touch sizing, containment, wrapping, and non-color active cues', () => {
    expect(headerSource).toContain('min-width: 2.75rem');
    expect(headerSource).toContain('min-height: 2.75rem');
    expect(headerSource).toContain('grid-template-columns: minmax(0, 1fr) auto');
    expect(headerSource).toContain('overflow-wrap: anywhere');
    expect(headerSource).not.toMatch(/overflow-x:\s*(?:auto|scroll)/);
    expect(headerSource).not.toMatch(/position:\s*(?:fixed|sticky)/);
    expect(headerSource).toContain("a[aria-current='page']");
    expect(headerSource).toContain('text-decoration: underline');
  });

  it('inherits reduced-motion behavior and adds no prohibited client capability', () => {
    expect(globalSource).toContain('@media (prefers-reduced-motion: reduce)');
    expect(headerSource).not.toMatch(
      /history\.|pushState|replaceState|localStorage|sessionStorage|MutationObserver|dispatchEvent|CustomEvent|scroll-lock|overflow:\s*hidden/i,
    );
  });

  it('keeps protected layouts, routes, Footer, and global styles byte-identical', () => {
    for (const [relativePath, expectedDigest] of protectedDigests) {
      expect(digest(readSource(relativePath))).toBe(expectedDigest);
    }
  });
});
