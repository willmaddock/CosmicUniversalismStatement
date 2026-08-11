import type { ResearchClassification } from '../research-taxonomy';

export type MediaFormat =
  | 'video'
  | 'short'
  | 'written-explainer'
  | 'diagram-walkthrough'
  | 'recorded-conversation'
  | 'audio';

export type MediaPublicationStatus =
  | 'staging'
  | 'published'
  | 'archived'
  | 'superseded';

export type MediaProviderAvailability = 'available' | 'unavailable';

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
