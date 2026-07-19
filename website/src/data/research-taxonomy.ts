export type ResearchClassificationTone =
  | 'empirical'
  | 'mathematical'
  | 'theoretical'
  | 'philosophical'
  | 'narrative'
  | 'open'
  | 'historical';

export interface ResearchClassificationDefinition {
  label: string;
  description: string;
  marker: string;
  tone: ResearchClassificationTone;
}

export const researchClassifications = {
  'empirical-reference': {
    label: 'Empirical Reference',
    description:
      'References to observational evidence, established measurements, standards, or conventional scientific material. This label does not validate a CU interpretation.',
    marker: '●',
    tone: 'empirical',
  },
  'cu-mathematical-model': {
    label: 'CU Mathematical Model',
    description:
      'CU-specific mathematical constructions derived from CU constants, formulas, ledgers, or converter logic. This label does not establish empirical validity.',
    marker: '■',
    tone: 'mathematical',
  },
  'cu-theoretical-proposition': {
    label: 'CU Theoretical Proposition',
    description:
      'CU framework claims or structural interpretations that are not established empirical cosmology.',
    marker: '◆',
    tone: 'theoretical',
  },
  'philosophical-interpretation': {
    label: 'Philosophical Interpretation',
    description:
      'Interpretive ideas concerning meaning, consciousness, authorship, free will, or related philosophical questions.',
    marker: '◇',
    tone: 'philosophical',
  },
  'in-world-narrative': {
    label: 'In-World Narrative',
    description:
      'Material presented through the project’s narrative voice rather than as empirical evidence or a scientific conclusion.',
    marker: '▰',
    tone: 'narrative',
  },
  'open-question': {
    label: 'Open Question',
    description:
      'An unresolved contradiction, definition, research problem, or mathematical tension.',
    marker: '?',
    tone: 'open',
  },
  'historical-superseded': {
    label: 'Historical / Superseded',
    description:
      'An earlier interpretation preserved for the project record but not treated as a current conclusion.',
    marker: '↺',
    tone: 'historical',
  },
} as const satisfies Record<string, ResearchClassificationDefinition>;

export type ResearchClassification = keyof typeof researchClassifications;

export const coreResearchClassifications = [
  'empirical-reference',
  'cu-mathematical-model',
  'cu-theoretical-proposition',
  'open-question',
] as const satisfies readonly ResearchClassification[];

export const allResearchClassifications = Object.keys(
  researchClassifications,
) as ResearchClassification[];

export type ResearchStatusTone =
  | 'foundational'
  | 'active'
  | 'provisional'
  | 'review'
  | 'open'
  | 'superseded'
  | 'archived';

export interface ResearchStatusDefinition {
  label: string;
  description: string;
  tone: ResearchStatusTone;
}

export const researchStatuses = {
  foundational: {
    label: 'Foundational',
    description: 'A current basis for other work in the framework.',
    tone: 'foundational',
  },
  'active-research': {
    label: 'Active Research',
    description: 'Work that is currently being developed or extended.',
    tone: 'active',
  },
  provisional: {
    label: 'Provisional',
    description: 'A tentative position retained for evaluation and revision.',
    tone: 'provisional',
  },
  'under-review': {
    label: 'Under Review',
    description: 'Material undergoing structured examination or technical review.',
    tone: 'review',
  },
  'open-problem': {
    label: 'Open Problem',
    description: 'A recognized unresolved problem requiring further work.',
    tone: 'open',
  },
  superseded: {
    label: 'Superseded',
    description: 'A position replaced by a later interpretation but preserved in history.',
    tone: 'superseded',
  },
  archived: {
    label: 'Archived',
    description: 'Material retained for reference but no longer under active development.',
    tone: 'archived',
  },
} as const satisfies Record<string, ResearchStatusDefinition>;

export type ResearchStatus = keyof typeof researchStatuses;

export const allResearchStatuses = Object.keys(researchStatuses) as ResearchStatus[];
