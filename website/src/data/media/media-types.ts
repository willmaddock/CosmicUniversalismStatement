import type { ResearchClassification } from '../research-taxonomy';

export const mediaFormats = [
  'video',
  'short',
  'written-explainer',
  'diagram-walkthrough',
  'recorded-conversation',
  'audio',
] as const;

export type MediaFormat = (typeof mediaFormats)[number];

export const mediaPublicationStatuses = [
  'staging',
  'published',
  'archived',
  'superseded',
] as const;

export type MediaPublicationStatus = (typeof mediaPublicationStatuses)[number];

export const mediaProviderAvailabilities = ['available', 'unavailable'] as const;

export type MediaProviderAvailability =
  (typeof mediaProviderAvailabilities)[number];

export type MediaSource =
  | {
      kind: 'internal';
      label: string;
      path: string;
      establishes: string;
    }
  | {
      kind: 'external';
      label: string;
      href: string;
      disclosure: string;
      establishes: string;
    };

export interface MediaEntry {
  id: string;
  slug: string;
  youtubeId: string;
  youtubeUrl: string;
  title: string;
  summary: string;
  runtime: string;
  format: MediaFormat;
  classifications: readonly ResearchClassification[];
  publicationStatus: MediaPublicationStatus;
  providerAvailability: MediaProviderAvailability;
  playlist?: string;
  transcriptPath: string;
  captionPath: string;
  captionSha256: string;
  posterPath: string;
  posterAlt: string;
  sourceBasis: readonly MediaSource[];
  revision: string;
}
