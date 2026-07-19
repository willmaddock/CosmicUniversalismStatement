const basePath = import.meta.env.BASE_URL;

export const siteMetadata = {
  name: 'Cosmic Universalism',
  defaultDescription: 'Explore the principles and research of Cosmic Universalism.',
  repositoryUrl: 'https://github.com/willmaddock/CosmicUniversalismStatement',
} as const;

export interface NavigationItem {
  label: string;
  href: string;
  external?: boolean;
}

export const primaryNavigation: readonly NavigationItem[] = [
  { label: 'Framework', href: `${basePath}framework/` },
  { label: 'Cosmic Breath', href: `${basePath}cosmic-breath/` },
  { label: 'CU-Time', href: `${basePath}cu-time/` },
  { label: 'Research', href: `${basePath}research/` },
  { label: 'Media', href: `${basePath}media/` },
  { label: 'About', href: `${basePath}about/` },
  { label: 'GitHub', href: siteMetadata.repositoryUrl, external: true },
];

export const primaryAction: NavigationItem = {
  label: 'Explore CU-Time',
  href: `${basePath}cu-time/`,
};

export function normalizePathname(pathname: string): string {
  const normalizedPathname = pathname.replace(/\/+$/, '');
  return normalizedPathname || '/';
}

export function isCurrentPath(pathname: string, href: string): boolean {
  if (!href.startsWith('/')) return false;

  const currentPathname = normalizePathname(pathname);
  const targetPathname = normalizePathname(href);

  if (targetPathname === '/') return currentPathname === '/';

  return (
    currentPathname === targetPathname ||
    currentPathname.startsWith(`${targetPathname}/`)
  );
}

export function formatPageTitle(title?: string): string {
  if (!title || title === siteMetadata.name) return siteMetadata.name;
  if (title.endsWith(`| ${siteMetadata.name}`)) return title;

  return `${title} | ${siteMetadata.name}`;
}