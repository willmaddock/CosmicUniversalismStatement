import type {
  ResearchClassification,
  ResearchStatus,
} from './research-taxonomy';

export type IsoDateString = `${number}-${number}-${number}`;

export interface CuRepositorySource {
  kind: 'cu-repository-file';
  label: string;
  repositoryPath: string;
  url?: string;
}

export interface ExternalEmpiricalReference {
  kind: 'empirical-reference';
  title: string;
  citation: string;
  url?: string;
}

export interface ResearchRevision {
  version: string;
  date: IsoDateString;
  summary: string;
}

export interface ResearchOpenQuestion {
  id: string;
  question: string;
}

type CurrentResearchStatus = Exclude<ResearchStatus, 'superseded' | 'archived'>;

export type ResearchLifecycle =
  | {
      status: CurrentResearchStatus;
      note?: string;
    }
  | {
      status: 'superseded';
      note: string;
      supersededBy: string;
    }
  | {
      status: 'archived';
      archiveNote: string;
    };

export interface ResearchArticleMetadata {
  slug: string;
  title: string;
  summary: string;
  classifications: readonly [ResearchClassification, ...ResearchClassification[]];
  lifecycle: ResearchLifecycle;
  version: string;
  lastReviewed: IsoDateString;
  originalSource: CuRepositorySource;
  empiricalReferences: readonly ExternalEmpiricalReference[];
  assumptions?: readonly string[];
  limitations?: readonly string[];
  openQuestions?: readonly ResearchOpenQuestion[];
  revisionHistory?: readonly ResearchRevision[];
}

export const researchProvenanceFields = [
  {
    key: 'original-file',
    label: 'Original CU repository file',
    description: 'The project file from which a future research entry is prepared.',
  },
  {
    key: 'external-references',
    label: 'External empirical references',
    description:
      'Observational or established scientific material cited separately from CU interpretation.',
  },
  {
    key: 'version',
    label: 'Version',
    description: 'The identified revision of the research entry being presented.',
  },
  {
    key: 'last-reviewed',
    label: 'Last reviewed',
    description: 'The date on which the entry most recently received editorial review.',
  },
  {
    key: 'revision-history',
    label: 'Revision history',
    description: 'A record of material changes, replacements, and preserved context.',
  },
] as const;
