import rawAuthority from '../../data/research/CU-RESEARCH-OBSERVATORY-1.0.json';
import { sitePath } from '../../config/site';
import {
  researchClassifications,
  researchStatuses,
  type ResearchClassification,
  type ResearchStatus,
} from '../../data/research-taxonomy';

export type ResearchNodeRole = 'overview' | 'primary' | 'supporting';
export type ResearchRelationshipKind = 'strong' | 'related' | 'support';
export type ResearchKeyboardKey =
  | 'ArrowUp'
  | 'ArrowRight'
  | 'ArrowDown'
  | 'ArrowLeft';

export interface InternalResearchDestination {
  readonly kind: 'internal';
  readonly path: string;
  readonly label: string;
}

export interface ExternalResearchDestination {
  readonly kind: 'external';
  readonly href: string;
  readonly label: string;
  readonly externalLabel: string;
}

export type ResearchDestination =
  | InternalResearchDestination
  | ExternalResearchDestination;

export interface ResearchNodeGovernance {
  readonly decisionIds: readonly string[];
  readonly reviewStatus: 'owner-approved-part-i';
}

export interface ResearchNodeRecord {
  readonly id: string;
  readonly title: string;
  readonly shortTitle: string;
  readonly role: ResearchNodeRole;
  readonly status: ResearchStatus;
  readonly classifications: readonly ResearchClassification[];
  readonly summary: string;
  readonly epistemicBoundary?: string;
  readonly primaryDestination?: ResearchDestination;
  readonly governingSourceDestination?: ResearchDestination;
  readonly governance: ResearchNodeGovernance;
}

export interface ResearchFilterGroup {
  readonly id: string;
  readonly label: string;
  readonly classifications: readonly ResearchClassification[];
}

export interface ResearchRelationship {
  readonly sourceId: string;
  readonly targetId: string;
  readonly kind: ResearchRelationshipKind;
  readonly publicExplanation: string;
}

export type ResearchKeyboardNavigation = Readonly<
  Record<string, Readonly<Partial<Record<ResearchKeyboardKey, string>>>>
>;

export interface ResearchObservatoryGovernance {
  readonly owner: 'William Maddock';
  readonly decisionIds: readonly string[];
  readonly contentBoundary: string;
}

export interface ResearchObservatoryAuthority {
  readonly authorityId: 'CU-RESEARCH-OBSERVATORY-1.0';
  readonly version: '1.0';
  readonly targetRoute: string;
  readonly overviewNodeId: string;
  readonly governance: ResearchObservatoryGovernance;
  readonly filterGroups: readonly ResearchFilterGroup[];
  readonly nodes: readonly ResearchNodeRecord[];
  readonly relationships: readonly ResearchRelationship[];
  readonly keyboardNavigation: ResearchKeyboardNavigation;
}

export interface PublicResearchDestination {
  readonly kind: 'internal' | 'external';
  readonly href: string;
  readonly label: string;
  readonly external: boolean;
  readonly externalLabel?: string;
  readonly opensInNewTab?: true;
  readonly rel?: 'noopener noreferrer';
}

export interface PublicResearchNode {
  readonly id: string;
  readonly title: string;
  readonly shortTitle: string;
  readonly role: ResearchNodeRole;
  readonly status: ResearchStatus;
  readonly classifications: readonly ResearchClassification[];
  readonly summary: string;
  readonly epistemicBoundary?: string;
  readonly primaryDestination?: PublicResearchDestination;
  readonly governingSourceDestination?: PublicResearchDestination;
}

export interface PublicResearchRegistry {
  readonly authorityId: 'CU-RESEARCH-OBSERVATORY-1.0';
  readonly version: '1.0';
  readonly targetRoute: string;
  readonly overviewNodeId: string;
  readonly filterGroups: readonly ResearchFilterGroup[];
  readonly nodes: readonly PublicResearchNode[];
  readonly relationships: readonly ResearchRelationship[];
  readonly keyboardNavigation: ResearchKeyboardNavigation;
}

export const PUBLIC_RESEARCH_REGISTRY_KEYS = Object.freeze([
  'authorityId',
  'version',
  'targetRoute',
  'overviewNodeId',
  'filterGroups',
  'nodes',
  'relationships',
  'keyboardNavigation',
] as const satisfies readonly (keyof PublicResearchRegistry)[]);

export const PUBLIC_RESEARCH_NODE_KEYS = Object.freeze([
  'id',
  'title',
  'shortTitle',
  'role',
  'status',
  'classifications',
  'summary',
  'epistemicBoundary',
  'primaryDestination',
  'governingSourceDestination',
] as const satisfies readonly (keyof PublicResearchNode)[]);

export const PUBLIC_RESEARCH_DESTINATION_KEYS = Object.freeze([
  'kind',
  'href',
  'label',
  'external',
  'externalLabel',
  'opensInNewTab',
  'rel',
] as const satisfies readonly (keyof PublicResearchDestination)[]);

const AUTHORITY_KEYS = new Set([
  'authorityId',
  'version',
  'targetRoute',
  'overviewNodeId',
  'governance',
  'filterGroups',
  'nodes',
  'relationships',
  'keyboardNavigation',
]);
const AUTHORITY_GOVERNANCE_KEYS = new Set([
  'owner',
  'decisionIds',
  'contentBoundary',
]);
const FILTER_KEYS = new Set(['id', 'label', 'classifications']);
const NODE_KEYS = new Set([
  'id',
  'title',
  'shortTitle',
  'role',
  'status',
  'classifications',
  'summary',
  'epistemicBoundary',
  'primaryDestination',
  'governingSourceDestination',
  'governance',
]);
const NODE_GOVERNANCE_KEYS = new Set([
  'decisionIds',
  'reviewStatus',
]);
const INTERNAL_DESTINATION_KEYS = new Set(['kind', 'path', 'label']);
const EXTERNAL_DESTINATION_KEYS = new Set([
  'kind',
  'href',
  'label',
  'externalLabel',
]);
const RELATIONSHIP_KEYS = new Set([
  'sourceId',
  'targetId',
  'kind',
  'publicExplanation',
]);
const KEYBOARD_KEYS = Object.freeze([
  'ArrowUp',
  'ArrowRight',
  'ArrowDown',
  'ArrowLeft',
] as const);
const KEYBOARD_KEY_SET = new Set<string>(KEYBOARD_KEYS);
const NODE_ROLES = Object.freeze([
  'overview',
  'primary',
  'supporting',
] as const);
const RELATIONSHIP_KINDS = Object.freeze([
  'strong',
  'related',
  'support',
] as const);
const PUBLIC_REGISTRY_KEY_SET = new Set<string>(PUBLIC_RESEARCH_REGISTRY_KEYS);
const PUBLIC_NODE_KEY_SET = new Set<string>(PUBLIC_RESEARCH_NODE_KEYS);
const PUBLIC_DESTINATION_KEY_SET = new Set<string>(
  PUBLIC_RESEARCH_DESTINATION_KEYS,
);
const PUBLIC_FILTER_KEY_SET = FILTER_KEYS;
const PUBLIC_RELATIONSHIP_KEY_SET = RELATIONSHIP_KEYS;
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const INTERNAL_PATH_PATTERN =
  /^[a-z0-9]+(?:-[a-z0-9]+)*(?:\/[a-z0-9]+(?:-[a-z0-9]+)*)*(?:#[a-z0-9]+(?:-[a-z0-9]+)*)?$/;

const fail = (message: string): never => {
  throw new Error(`Invalid Research Observatory authority: ${message}`);
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const requireRecord = (
  value: unknown,
  name: string,
): Record<string, unknown> =>
  isRecord(value) ? value : fail(`${name} must be an object`);

const requireArray = (value: unknown, name: string): readonly unknown[] =>
  Array.isArray(value) ? value : fail(`${name} must be an array`);

const requireString = (value: unknown, name: string): string =>
  typeof value === 'string' && value.trim().length > 0
    ? value
    : fail(`${name} must be a non-empty string`);

const requireExact = <T extends string>(
  value: unknown,
  expected: T,
  name: string,
): T =>
  value === expected
    ? expected
    : fail(`${name} must equal "${expected}"`);

const rejectUnexpectedKeys = (
  record: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  name: string,
): void => {
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) {
      fail(`${name} contains unexpected field "${key}"`);
    }
  }
};

const requireUniqueStrings = (
  value: unknown,
  name: string,
): readonly string[] => {
  const entries = requireArray(value, name).map((entry, index) =>
    requireString(entry, `${name}[${index}]`));
  if (new Set(entries).size !== entries.length) {
    fail(`${name} must not contain duplicates`);
  }
  return Object.freeze(entries);
};

const requireId = (value: unknown, name: string): string => {
  const id = requireString(value, name);
  return ID_PATTERN.test(id)
    ? id
    : fail(`${name} must be a lowercase kebab-case ID`);
};

const requireClassification = (
  value: unknown,
  name: string,
): ResearchClassification => {
  const classification = requireString(value, name);
  return classification in researchClassifications
    ? classification as ResearchClassification
    : fail(`${name} is not an existing research classification`);
};

const requireClassifications = (
  value: unknown,
  name: string,
): readonly ResearchClassification[] => {
  const classifications = requireArray(value, name).map((entry, index) =>
    requireClassification(entry, `${name}[${index}]`));
  if (new Set(classifications).size !== classifications.length) {
    fail(`${name} must not contain duplicates`);
  }
  return Object.freeze(classifications);
};

const requireStatus = (value: unknown, name: string): ResearchStatus => {
  const status = requireString(value, name);
  return status in researchStatuses
    ? status as ResearchStatus
    : fail(`${name} is not an existing research status`);
};

const requireNodeRole = (
  value: unknown,
  name: string,
): ResearchNodeRole => {
  const role = requireString(value, name);
  return NODE_ROLES.includes(role as ResearchNodeRole)
    ? role as ResearchNodeRole
    : fail(`${name} is not an approved node role`);
};

const requireRelationshipKind = (
  value: unknown,
  name: string,
): ResearchRelationshipKind => {
  const kind = requireString(value, name);
  return RELATIONSHIP_KINDS.includes(kind as ResearchRelationshipKind)
    ? kind as ResearchRelationshipKind
    : fail(`${name} is not an approved relationship kind`);
};

const parseDestination = (
  value: unknown,
  name: string,
): ResearchDestination => {
  const record = requireRecord(value, name);
  const kind = requireString(record.kind, `${name}.kind`);
  if (kind === 'internal') {
    rejectUnexpectedKeys(record, INTERNAL_DESTINATION_KEYS, name);
    const path = requireString(record.path, `${name}.path`);
    if (
      path.startsWith('/')
      || path.includes('..')
      || !INTERNAL_PATH_PATTERN.test(path)
    ) {
      fail(`${name}.path must be a base-path-safe sitePath argument`);
    }
    return Object.freeze({
      kind,
      path,
      label: requireString(record.label, `${name}.label`),
    });
  }
  if (kind === 'external') {
    rejectUnexpectedKeys(record, EXTERNAL_DESTINATION_KEYS, name);
    const href = requireString(record.href, `${name}.href`);
    let url: URL;
    try {
      url = new URL(href);
    } catch {
      return fail(`${name}.href must be a valid URL`);
    }
    if (url.protocol !== 'https:') {
      fail(`${name}.href must use HTTPS`);
    }
    const externalLabel = requireString(
      record.externalLabel,
      `${name}.externalLabel`,
    );
    if (!/\bexternal\b/i.test(externalLabel)) {
      fail(`${name}.externalLabel must disclose external navigation`);
    }
    return Object.freeze({
      kind,
      href,
      label: requireString(record.label, `${name}.label`),
      externalLabel,
    });
  }
  return fail(`${name}.kind must be internal or external`);
};

const parseFilter = (
  value: unknown,
  index: number,
): ResearchFilterGroup => {
  const name = `filterGroups[${index}]`;
  const record = requireRecord(value, name);
  rejectUnexpectedKeys(record, FILTER_KEYS, name);
  return Object.freeze({
    id: requireId(record.id, `${name}.id`),
    label: requireString(record.label, `${name}.label`),
    classifications: requireClassifications(
      record.classifications,
      `${name}.classifications`,
    ),
  });
};

const parseNode = (value: unknown, index: number): ResearchNodeRecord => {
  const name = `nodes[${index}]`;
  const record = requireRecord(value, name);
  rejectUnexpectedKeys(record, NODE_KEYS, name);
  const governanceRecord = requireRecord(
    record.governance,
    `${name}.governance`,
  );
  rejectUnexpectedKeys(
    governanceRecord,
    NODE_GOVERNANCE_KEYS,
    `${name}.governance`,
  );
  const epistemicBoundary =
    record.epistemicBoundary === undefined
      ? undefined
      : requireString(
        record.epistemicBoundary,
        `${name}.epistemicBoundary`,
      );
  const role = requireNodeRole(record.role, `${name}.role`);
  const primaryDestination =
    record.primaryDestination === undefined
      ? undefined
      : parseDestination(
        record.primaryDestination,
        `${name}.primaryDestination`,
      );
  const governingSourceDestination =
    record.governingSourceDestination === undefined
      ? undefined
      : parseDestination(
        record.governingSourceDestination,
        `${name}.governingSourceDestination`,
      );
  if (
    role !== 'overview'
    && primaryDestination === undefined
    && governingSourceDestination === undefined
  ) {
    fail(`${name} must contain at least one public action`);
  }
  return Object.freeze({
    id: requireId(record.id, `${name}.id`),
    title: requireString(record.title, `${name}.title`),
    shortTitle: requireString(record.shortTitle, `${name}.shortTitle`),
    role,
    status: requireStatus(record.status, `${name}.status`),
    classifications: requireClassifications(
      record.classifications,
      `${name}.classifications`,
    ),
    summary: requireString(record.summary, `${name}.summary`),
    ...(epistemicBoundary === undefined ? {} : { epistemicBoundary }),
    ...(primaryDestination === undefined ? {} : { primaryDestination }),
    ...(governingSourceDestination === undefined
      ? {}
      : { governingSourceDestination }),
    governance: Object.freeze({
      decisionIds: requireUniqueStrings(
        governanceRecord.decisionIds,
        `${name}.governance.decisionIds`,
      ),
      reviewStatus: requireExact(
        governanceRecord.reviewStatus,
        'owner-approved-part-i',
        `${name}.governance.reviewStatus`,
      ),
    }),
  });
};

const parseRelationship = (
  value: unknown,
  index: number,
): ResearchRelationship => {
  const name = `relationships[${index}]`;
  const record = requireRecord(value, name);
  rejectUnexpectedKeys(record, RELATIONSHIP_KEYS, name);
  return Object.freeze({
    sourceId: requireId(record.sourceId, `${name}.sourceId`),
    targetId: requireId(record.targetId, `${name}.targetId`),
    kind: requireRelationshipKind(record.kind, `${name}.kind`),
    publicExplanation: requireString(
      record.publicExplanation,
      `${name}.publicExplanation`,
    ),
  });
};

const parseKeyboardNavigation = (
  value: unknown,
  nodeIds: ReadonlySet<string>,
): ResearchKeyboardNavigation => {
  const record = requireRecord(value, 'keyboardNavigation');
  if (
    Object.keys(record).length !== nodeIds.size
    || Object.keys(record).some((id) => !nodeIds.has(id))
  ) {
    fail('keyboardNavigation must contain exactly one entry for every node');
  }
  const result: Record<
    string,
    Readonly<Partial<Record<ResearchKeyboardKey, string>>>
  > = {};
  for (const [nodeId, rawNeighbors] of Object.entries(record)) {
    const neighbors = requireRecord(
      rawNeighbors,
      `keyboardNavigation.${nodeId}`,
    );
    for (const key of Object.keys(neighbors)) {
      if (!KEYBOARD_KEY_SET.has(key)) {
        fail(`keyboardNavigation.${nodeId} contains unsupported key "${key}"`);
      }
    }
    const parsed: Partial<Record<ResearchKeyboardKey, string>> = {};
    for (const key of KEYBOARD_KEYS) {
      if (neighbors[key] === undefined) continue;
      const targetId = requireId(
        neighbors[key],
        `keyboardNavigation.${nodeId}.${key}`,
      );
      if (!nodeIds.has(targetId)) {
        fail(`keyboardNavigation.${nodeId}.${key} has an unknown target`);
      }
      if (targetId === nodeId) {
        fail(`keyboardNavigation.${nodeId}.${key} must not target itself`);
      }
      parsed[key] = targetId;
    }
    result[nodeId] = Object.freeze(parsed);
  }
  return Object.freeze(result);
};

const validateAuthorityInvariants = (
  authority: ResearchObservatoryAuthority,
): void => {
  const { nodes, filterGroups, relationships, overviewNodeId } = authority;
  const nodeIds = nodes.map((node) => node.id);
  if (nodes.length !== 11 || new Set(nodeIds).size !== nodes.length) {
    fail('nodes must contain exactly eleven unique records');
  }
  if (nodes.filter((node) => node.role === 'overview').length !== 1) {
    fail('nodes must contain exactly one overview record');
  }
  if (nodes.filter((node) => node.role === 'primary').length !== 6) {
    fail('nodes must contain exactly six primary records');
  }
  if (nodes.filter((node) => node.role === 'supporting').length !== 4) {
    fail('nodes must contain exactly four supporting records');
  }
  if (
    !nodes.some(
      (node) => node.id === overviewNodeId && node.role === 'overview',
    )
  ) {
    fail('overviewNodeId must identify the overview record');
  }

  const filterIds = filterGroups.map((filter) => filter.id);
  if (
    filterGroups.length !== 6
    || new Set(filterIds).size !== filterGroups.length
  ) {
    fail('filterGroups must contain exactly six unique records');
  }
  const allFilter = filterGroups.find((filter) => filter.id === 'all');
  if (!allFilter || allFilter.classifications.length !== 0) {
    fail('the all filter must exist and have no classification restriction');
  }
  if (
    filterGroups
      .filter((filter) => filter.id !== 'all')
      .some((filter) => filter.classifications.length === 0)
  ) {
    fail('every non-all filter must contain classifications');
  }

  const knownNodeIds = new Set(nodeIds);
  for (const node of nodes) {
    const primary = node.primaryDestination;
    const source = node.governingSourceDestination;
    if (primary && source) {
      const primaryKey = primary.kind === 'internal'
        ? `internal:${primary.path}`
        : `external:${primary.href}`;
      const sourceKey = source.kind === 'internal'
        ? `internal:${source.path}`
        : `external:${source.href}`;
      if (primaryKey === sourceKey) {
        fail(`${node.id} must not contain duplicate public actions`);
      }
    }
  }

  const relationshipPairs = new Set<string>();
  for (const relationship of relationships) {
    if (
      !knownNodeIds.has(relationship.sourceId)
      || !knownNodeIds.has(relationship.targetId)
    ) {
      fail('relationships must reference existing nodes');
    }
    if (relationship.sourceId === relationship.targetId) {
      fail('relationships must not reference the same node twice');
    }
    const pair = [relationship.sourceId, relationship.targetId]
      .sort()
      .join('::');
    if (relationshipPairs.has(pair)) {
      fail(`duplicate relationship for ${pair}`);
    }
    relationshipPairs.add(pair);
  }
};

export const parseResearchObservatoryAuthority = (
  value: unknown,
): ResearchObservatoryAuthority => {
  const record = requireRecord(value, 'authority');
  rejectUnexpectedKeys(record, AUTHORITY_KEYS, 'authority');
  const governanceRecord = requireRecord(
    record.governance,
    'authority.governance',
  );
  rejectUnexpectedKeys(
    governanceRecord,
    AUTHORITY_GOVERNANCE_KEYS,
    'authority.governance',
  );
  const nodes = Object.freeze(requireArray(record.nodes, 'nodes').map(parseNode));
  const nodeIds = new Set(nodes.map((node) => node.id));
  if (nodeIds.size !== nodes.length) {
    fail('nodes must contain unique IDs');
  }
  const authority: ResearchObservatoryAuthority = Object.freeze({
    authorityId: requireExact(
      record.authorityId,
      'CU-RESEARCH-OBSERVATORY-1.0',
      'authority.authorityId',
    ),
    version: requireExact(record.version, '1.0', 'authority.version'),
    targetRoute: (() => {
      const route = requireString(record.targetRoute, 'authority.targetRoute');
      if (!INTERNAL_PATH_PATTERN.test(route) || route.includes('#')) {
        fail('authority.targetRoute must be a base-path-safe route');
      }
      return route;
    })(),
    overviewNodeId: requireId(
      record.overviewNodeId,
      'authority.overviewNodeId',
    ),
    governance: Object.freeze({
      owner: requireExact(
        governanceRecord.owner,
        'William Maddock',
        'authority.governance.owner',
      ),
      decisionIds: requireUniqueStrings(
        governanceRecord.decisionIds,
        'authority.governance.decisionIds',
      ),
      contentBoundary: requireString(
        governanceRecord.contentBoundary,
        'authority.governance.contentBoundary',
      ),
    }),
    filterGroups: Object.freeze(
      requireArray(record.filterGroups, 'filterGroups').map(parseFilter),
    ),
    nodes,
    relationships: Object.freeze(
      requireArray(record.relationships, 'relationships').map(
        parseRelationship,
      ),
    ),
    keyboardNavigation: parseKeyboardNavigation(
      record.keyboardNavigation,
      nodeIds,
    ),
  });
  validateAuthorityInvariants(authority);
  return authority;
};

export const researchObservatoryAuthority =
  parseResearchObservatoryAuthority(rawAuthority as unknown);

const projectDestination = (
  destination: ResearchDestination,
  resolveInternal: (path: string) => string,
): PublicResearchDestination =>
  destination.kind === 'internal'
    ? Object.freeze({
      kind: destination.kind,
      href: resolveInternal(destination.path),
      label: destination.label,
      external: false,
    })
    : Object.freeze({
      kind: destination.kind,
      href: destination.href,
      label: destination.label,
      external: true,
      externalLabel: destination.externalLabel,
      opensInNewTab: true,
      rel: 'noopener noreferrer',
    });

export const createPublicResearchRegistry = (
  resolveInternal: (path: string) => string = sitePath,
): PublicResearchRegistry =>
  Object.freeze({
    authorityId: researchObservatoryAuthority.authorityId,
    version: researchObservatoryAuthority.version,
    targetRoute: resolveInternal(researchObservatoryAuthority.targetRoute),
    overviewNodeId: researchObservatoryAuthority.overviewNodeId,
    filterGroups: Object.freeze(
      researchObservatoryAuthority.filterGroups.map((filter) =>
        Object.freeze({
          id: filter.id,
          label: filter.label,
          classifications: Object.freeze([...filter.classifications]),
        })),
    ),
    nodes: Object.freeze(
      researchObservatoryAuthority.nodes.map((node) =>
        Object.freeze({
          id: node.id,
          title: node.title,
          shortTitle: node.shortTitle,
          role: node.role,
          status: node.status,
          classifications: Object.freeze([...node.classifications]),
          summary: node.summary,
          ...(node.epistemicBoundary === undefined
            ? {}
            : { epistemicBoundary: node.epistemicBoundary }),
          ...(node.primaryDestination === undefined
            ? {}
            : {
              primaryDestination: projectDestination(
                node.primaryDestination,
                resolveInternal,
              ),
            }),
          ...(node.governingSourceDestination === undefined
            ? {}
            : {
              governingSourceDestination: projectDestination(
                node.governingSourceDestination,
                resolveInternal,
              ),
            }),
        })),
    ),
    relationships: Object.freeze(
      researchObservatoryAuthority.relationships.map((relationship) =>
        Object.freeze({ ...relationship })),
    ),
    keyboardNavigation: Object.freeze(
      Object.fromEntries(
        Object.entries(researchObservatoryAuthority.keyboardNavigation).map(
          ([nodeId, neighbors]) => [
            nodeId,
            Object.freeze({ ...neighbors }),
          ],
        ),
      ),
    ),
  });

export const publicResearchRegistry = createPublicResearchRegistry();

const validatePublicDestination = (
  value: unknown,
  name: string,
): void => {
  const record = requireRecord(value, name);
  rejectUnexpectedKeys(record, PUBLIC_DESTINATION_KEY_SET, name);
  const kind = requireString(record.kind, `${name}.kind`);
  requireString(record.href, `${name}.href`);
  requireString(record.label, `${name}.label`);
  if (kind === 'internal') {
    if (record.external !== false) {
      fail(`${name}.external must be false for an internal destination`);
    }
    const href = requireString(record.href, `${name}.href`);
    if (!href.startsWith('/') || href.includes('..')) {
      fail(`${name}.href must be a resolved internal URL`);
    }
    for (const forbidden of [
      'externalLabel',
      'opensInNewTab',
      'rel',
    ]) {
      if (record[forbidden] !== undefined) {
        fail(`${name} contains external-only field "${forbidden}"`);
      }
    }
    return;
  }
  if (kind !== 'external' || record.external !== true) {
    fail(`${name} must contain approved external metadata`);
  }
  const href = requireString(record.href, `${name}.href`);
  let url: URL;
  try {
    url = new URL(href);
  } catch {
    return fail(`${name}.href must be a valid URL`);
  }
  if (url.protocol !== 'https:') {
    fail(`${name}.href must use HTTPS`);
  }
  const externalLabel = requireString(
    record.externalLabel,
    `${name}.externalLabel`,
  );
  if (!/\bexternal\b/i.test(externalLabel)) {
    fail(`${name}.externalLabel must disclose external navigation`);
  }
  if (
    record.opensInNewTab !== true
    || record.rel !== 'noopener noreferrer'
  ) {
    fail(`${name} must contain safe new-tab metadata`);
  }
};

const validatePublicProjection = (value: unknown): void => {
  const record = requireRecord(value, 'publicRegistry');
  rejectUnexpectedKeys(record, PUBLIC_REGISTRY_KEY_SET, 'publicRegistry');
  requireExact(
    record.authorityId,
    'CU-RESEARCH-OBSERVATORY-1.0',
    'publicRegistry.authorityId',
  );
  requireExact(record.version, '1.0', 'publicRegistry.version');
  const targetRoute = requireString(
    record.targetRoute,
    'publicRegistry.targetRoute',
  );
  if (!targetRoute.startsWith('/') || targetRoute.includes('..')) {
    fail('publicRegistry.targetRoute must be a resolved internal URL');
  }
  requireId(record.overviewNodeId, 'publicRegistry.overviewNodeId');

  const filterIds: string[] = [];
  requireArray(record.filterGroups, 'publicRegistry.filterGroups').forEach(
    (value, index) => {
      const item = requireRecord(
        value,
        `publicRegistry.filterGroups[${index}]`,
      );
      rejectUnexpectedKeys(
        item,
        PUBLIC_FILTER_KEY_SET,
        `publicRegistry.filterGroups[${index}]`,
      );
      filterIds.push(
        requireId(item.id, `publicRegistry.filterGroups[${index}].id`),
      );
      requireString(
        item.label,
        `publicRegistry.filterGroups[${index}].label`,
      );
      requireClassifications(
        item.classifications,
        `publicRegistry.filterGroups[${index}].classifications`,
      );
    },
  );
  if (new Set(filterIds).size !== filterIds.length) {
    fail('publicRegistry.filterGroups must have unique IDs');
  }

  const nodeIds: string[] = [];
  const nodeRoles: ResearchNodeRole[] = [];
  requireArray(record.nodes, 'publicRegistry.nodes').forEach(
    (value, index) => {
      const name = `publicRegistry.nodes[${index}]`;
      const node = requireRecord(value, name);
      rejectUnexpectedKeys(node, PUBLIC_NODE_KEY_SET, name);
      nodeIds.push(requireId(node.id, `${name}.id`));
      requireString(node.title, `${name}.title`);
      requireString(node.shortTitle, `${name}.shortTitle`);
      const role = requireNodeRole(node.role, `${name}.role`);
      nodeRoles.push(role);
      requireStatus(node.status, `${name}.status`);
      requireClassifications(node.classifications, `${name}.classifications`);
      requireString(node.summary, `${name}.summary`);
      if (node.epistemicBoundary !== undefined) {
        requireString(node.epistemicBoundary, `${name}.epistemicBoundary`);
      }
      if (node.primaryDestination !== undefined) {
        validatePublicDestination(
          node.primaryDestination,
          `${name}.primaryDestination`,
        );
      }
      if (node.governingSourceDestination !== undefined) {
        validatePublicDestination(
          node.governingSourceDestination,
          `${name}.governingSourceDestination`,
        );
      }
      if (
        role !== 'overview'
        && node.primaryDestination === undefined
        && node.governingSourceDestination === undefined
      ) {
        fail(`${name} must contain at least one public action`);
      }
      if (
        node.primaryDestination !== undefined
        && node.governingSourceDestination !== undefined
      ) {
        const primary = requireRecord(
          node.primaryDestination,
          `${name}.primaryDestination`,
        );
        const source = requireRecord(
          node.governingSourceDestination,
          `${name}.governingSourceDestination`,
        );
        if (primary.href === source.href) {
          fail(`${name} must not contain duplicate public actions`);
        }
      }
    },
  );
  if (nodeIds.length !== 11 || new Set(nodeIds).size !== nodeIds.length) {
    fail('publicRegistry.nodes must contain eleven unique IDs');
  }
  if (
    nodeRoles.filter((role) => role === 'overview').length !== 1
    || nodeRoles.filter((role) => role === 'primary').length !== 6
    || nodeRoles.filter((role) => role === 'supporting').length !== 4
  ) {
    fail('publicRegistry.nodes must preserve the approved role counts');
  }

  const knownNodeIds = new Set(nodeIds);
  const relationshipPairs = new Set<string>();
  requireArray(record.relationships, 'publicRegistry.relationships').forEach(
    (value, index) => {
      const name = `publicRegistry.relationships[${index}]`;
      const relationship = requireRecord(value, name);
      rejectUnexpectedKeys(
        relationship,
        PUBLIC_RELATIONSHIP_KEY_SET,
        name,
      );
      const sourceId = requireId(
        relationship.sourceId,
        `${name}.sourceId`,
      );
      const targetId = requireId(
        relationship.targetId,
        `${name}.targetId`,
      );
      if (
        !knownNodeIds.has(sourceId)
        || !knownNodeIds.has(targetId)
        || sourceId === targetId
      ) {
        fail(`${name} must connect two existing distinct nodes`);
      }
      const pair = [sourceId, targetId].sort().join('::');
      if (relationshipPairs.has(pair)) {
        fail(`publicRegistry contains duplicate relationship ${pair}`);
      }
      relationshipPairs.add(pair);
      requireRelationshipKind(relationship.kind, `${name}.kind`);
      requireString(
        relationship.publicExplanation,
        `${name}.publicExplanation`,
      );
    },
  );

  const keyboardNavigation = requireRecord(
    record.keyboardNavigation,
    'publicRegistry.keyboardNavigation',
  );
  if (
    Object.keys(keyboardNavigation).length !== knownNodeIds.size
    || Object.keys(keyboardNavigation).some((id) => !knownNodeIds.has(id))
  ) {
    fail('publicRegistry.keyboardNavigation must cover every node');
  }
  for (const [nodeId, rawNeighbors] of Object.entries(keyboardNavigation)) {
    const neighbors = requireRecord(
      rawNeighbors,
      `publicRegistry.keyboardNavigation.${nodeId}`,
    );
    for (const [key, rawTarget] of Object.entries(neighbors)) {
      if (!KEYBOARD_KEY_SET.has(key)) {
        fail(
          `publicRegistry.keyboardNavigation.${nodeId} contains unsupported key "${key}"`,
        );
      }
      const targetId = requireId(
        rawTarget,
        `publicRegistry.keyboardNavigation.${nodeId}.${key}`,
      );
      if (!knownNodeIds.has(targetId) || targetId === nodeId) {
        fail(
          `publicRegistry.keyboardNavigation.${nodeId}.${key} has an invalid target`,
        );
      }
    }
  }
};

export const serializePublicResearchRegistry = (
  value: unknown = publicResearchRegistry,
): string => {
  validatePublicProjection(value);
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
};
