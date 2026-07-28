import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  isCurrentPath,
  normalizePathname,
} from '../../../config/site';

const basePath = '/CosmicUniversalismStatement';
const internalDestinations = [
  '/',
  `${basePath}/framework`,
  `${basePath}/cosmic-breath`,
  `${basePath}/cu-time`,
  `${basePath}/research`,
  `${basePath}/media`,
  `${basePath}/about`,
  `${basePath}/cu-intelligence`,
] as const;

const routeCases = [
  [`${basePath}/`, '/'],
  [`${basePath}/framework/`, `${basePath}/framework`],
  [`${basePath}/cosmic-breath/`, `${basePath}/cosmic-breath`],
  [`${basePath}/cu-time/`, `${basePath}/cu-time`],
  [`${basePath}/cu-intelligence/`, `${basePath}/cu-intelligence`],
  [`${basePath}/research/`, `${basePath}/research`],
  [`${basePath}/media/`, `${basePath}/media`],
  [`${basePath}/about/`, `${basePath}/about`],
] as const;

const readSource = (relativePath: string) =>
  readFileSync(new URL(relativePath, import.meta.url), 'utf8');

const siteSource = readSource('../../../config/site.ts');
const headerSource = readSource('../../../components/Header.astro');
const footerSource = readSource('../../../components/Footer.astro');

describe('base-path-aware current route matching', () => {
  it('matches every configured public route in the same base-path coordinate system', () => {
    for (const [pathname, href] of routeCases) {
      expect(isCurrentPath(pathname, href, basePath)).toBe(true);
    }
  });

  it('treats trailing slashes, query strings, and fragments consistently', () => {
    expect(normalizePathname('/media/?view=cards#selected')).toBe('/media');
    expect(
      isCurrentPath(
        `${basePath}/media/?view=cards#selected`,
        `${basePath}/media/`,
        `${basePath}/`,
      ),
    ).toBe(true);
  });

  it('keeps Home exact and does not activate it on another route', () => {
    expect(isCurrentPath(`${basePath}/`, '/', basePath)).toBe(true);
    expect(isCurrentPath(`${basePath}/research/`, '/', basePath)).toBe(false);
  });

  it('rejects unrelated prefix paths and external destinations', () => {
    expect(
      isCurrentPath(
        `${basePath}/media-library/`,
        `${basePath}/media`,
        basePath,
      ),
    ).toBe(false);
    expect(
      isCurrentPath(
        `${basePath}/aboutness/`,
        `${basePath}/about`,
        basePath,
      ),
    ).toBe(false);
    expect(
      isCurrentPath(
        `${basePath}/research/`,
        'https://github.com/willmaddock/CosmicUniversalismStatement',
        basePath,
      ),
    ).toBe(false);
  });

  it('emits exactly one current Header destination for each public route', () => {
    for (const [pathname, expectedHref] of routeCases) {
      const matches = internalDestinations.filter((href) =>
        isCurrentPath(pathname, href, basePath),
      );

      expect(matches).toEqual([expectedHref]);
    }
  });

  it('preserves the complete navigation registry byte-for-byte outside matching logic', () => {
    const registry = siteSource.match(
      /export const primaryNavigation:[\s\S]*?export const primaryAction:[\s\S]*?};/,
    )?.[0];

    expect(registry).toContain("{ label: 'Framework', href: sitePath('framework') }");
    expect(registry).toContain("{ label: 'Cosmic Breath', href: sitePath('cosmic-breath') }");
    expect(registry).toContain("{ label: 'CU-Time', href: sitePath('cu-time') }");
    expect(registry).toContain("{ label: 'Research', href: sitePath('research') }");
    expect(registry).toContain("{ label: 'Media', href: sitePath('media') }");
    expect(registry).toContain("{ label: 'About', href: sitePath('about') }");
    expect(registry).toContain("label: 'GitHub'");
    expect(registry).toContain("label: 'Explore CUCII'");
  });

  it('keeps both existing consumers on the shared matching authority', () => {
    expect(headerSource.match(/isCurrentPath\(/g)).toHaveLength(3);
    expect(footerSource.match(/isCurrentPath\(/g)).toHaveLength(2);
    expect(headerSource).not.toMatch(/location\.pathname|window\.location/);
    expect(footerSource).not.toMatch(/location\.pathname|window\.location/);
  });
});
