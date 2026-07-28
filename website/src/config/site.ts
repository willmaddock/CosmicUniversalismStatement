export const siteMetadata = {
  name: 'Cosmic Universalism',
  defaultDescription:
    'Explore the principles and research of Cosmic Universalism.',
  repositoryUrl:
    'https://github.com/willmaddock/CosmicUniversalismStatement',
} as const;

export interface NavigationItem {
  label: string;
  href: string;
  external?: boolean;
}

/**
 * Creates a URL that respects Astro's configured base path.
 *
 * Example:
 * BASE_URL = "/CosmicUniversalismStatement/"
 *
 * sitePath("framework")
 * returns:
 * "/CosmicUniversalismStatement/framework/"
 */
export function sitePath(path: string): string {
  const base = import.meta.env.BASE_URL;

  const cleanBase = base.endsWith('/')
    ? base
    : `${base}/`;

  const cleanPath = path.replace(/^\/+/, '');

  return `${cleanBase}${cleanPath}`;
}

export const primaryNavigation: readonly NavigationItem[] = [
  { label: 'Framework', href: sitePath('framework') },
  { label: 'Cosmic Breath', href: sitePath('cosmic-breath') },
  { label: 'CU-Time', href: sitePath('cu-time') },
  { label: 'Research', href: sitePath('research') },
  { label: 'Media', href: sitePath('media') },
  { label: 'About', href: sitePath('about') },
  {
    label: 'GitHub',
    href: siteMetadata.repositoryUrl,
    external: true,
  },
];

export const primaryAction: NavigationItem = {
  label: 'Explore CUCII',
  href: sitePath('cu-intelligence'),
};

export function normalizePathname(pathname: string): string {
  const pathnameOnly = pathname.split(/[?#]/, 1)[0] ?? '';
  const normalizedPathname = pathnameOnly.replace(/\/+$/, '');

  return normalizedPathname || '/';
}

export function isCurrentPath(
  pathname: string,
  href: string,
  basePath = import.meta.env.BASE_URL,
): boolean {
  if (!href.startsWith('/')) return false;

  const normalizeFromBase = (value: string): string => {
    const normalizedValue = normalizePathname(value);
    const normalizedBase = normalizePathname(basePath);

    if (normalizedBase === '/') return normalizedValue;
    if (normalizedValue === normalizedBase) return '/';
    if (normalizedValue.startsWith(`${normalizedBase}/`)) {
      return normalizedValue.slice(normalizedBase.length) || '/';
    }

    return normalizedValue;
  };

  const currentPathname = normalizeFromBase(pathname);
  const targetPathname = normalizeFromBase(href);

  if (targetPathname === '/') {
    return currentPathname === targetPathname;
  }

  return (
    currentPathname === targetPathname ||
    currentPathname.startsWith(`${targetPathname}/`)
  );
}

export function formatPageTitle(title?: string): string {
  if (!title || title === siteMetadata.name) {
    return siteMetadata.name;
  }

  if (title.endsWith(`| ${siteMetadata.name}`)) {
    return title;
  }

  return `${title} | ${siteMetadata.name}`;
}
