# Future Media Publication Workflow

This is the repository-local maintainer checklist for extending the owner-curated Cosmic Universalism Media library. The governing Media Page Implementation Manual v1.2 remains authoritative. Live repository state and a bounded owner authorization must be verified before every change.

## Governing model

The website registry is the editorial authority. YouTube delivers playback but does not add, update, publish, archive, or remove a CU Media entry automatically.

The controlled path is:

1. owner approval;
2. approved `MediaEntry` values;
3. approved local publication assets;
4. focused tests, full tests, build, and review;
5. owner-controlled publication status;
6. authorized deployment and release.

Do not add a YouTube Data API integration, Google Cloud credential, API key, OAuth flow, channel scraper, provider probe, metadata cache, polling process, scheduled Media workflow, or client/build-time metadata fetch. A public or unlisted YouTube upload remains channel-only until the owner approves a website record.

## Stop conditions

Stop and request owner direction when any required editorial value is missing, ambiguous, inconsistent with the source package, or not explicitly approved. An implementation agent must not infer or invent titles, summaries, slugs, classifications, lifecycle state, availability, source claims, alt text, or release facts.

Do not change an existing record or asset merely because provider metadata changed. Do not create a demonstration record in production authority.

## Owner-controlled record

Add one object to `website/src/data/media/media-library.ts` that satisfies the centralized contract in `media-types.ts`. Obtain owner-reviewed values for every applicable field:

| Field | Requirement |
| --- | --- |
| `id` | Stable unique kebab-case identity. |
| `slug` | Unique permanent kebab-case CU route segment. Preserve it for revisions of the same publication. |
| `youtubeId` | Exact 11-character provider ID. |
| `youtubeUrl` | Normal `https://youtu.be/<youtubeId>` watch URL. |
| `title` | Exact owner-approved title. |
| `summary` | Concise owner-approved CU summary and boundary context. |
| `runtime` | Owner-verified ISO-8601 duration such as `PT6M45S`. |
| `format` | One value from `mediaFormats`; do not infer `short` from runtime alone. |
| `classifications` | One or more canonical research taxonomy keys, without duplicates. |
| `publicationStatus` | One value from `mediaPublicationStatuses`, changed only at an authorized lifecycle checkpoint. |
| `providerAvailability` | Owner-controlled `available` or `unavailable`; never determined by a probe. |
| `playlist` | Optional owner-approved label. |
| `posterPath` | `media/<slug>/thumbnail.png`. |
| `posterAlt` | Meaningful owner-approved alternative text. |
| `transcriptPath` | `media/<slug>/transcript-en.txt`. |
| `captionPath` | `media/<slug>/captions-en.vtt`. |
| `captionSha256` | Lowercase SHA-256 of the final publication VTT. |
| `sourceBasis` | Nonempty typed internal/external sources with truthful `establishes` statements. |
| `revision` | Owner-approved publication revision. |

The registry order is the deliberate landing-page order unless the owner approves a different ordering contract.

## Lifecycle contract

- `staging`: permitted on a feature branch or in local review; never describe it as publicly released.
- `published`: permitted only after explicit owner authorization at the release checkpoint.
- `archived`: the record remains historical but is no longer current; preserve its CU-owned publication material according to the approved archival decision.
- `superseded`: a replacement exists; preserve the prior record and correction/source history according to the approved revision decision.

Provider visibility and CU publication status are separate decisions. Provider-side Public status does not publish the website record.

## Provider availability

Record availability only from an owner decision. Do not probe YouTube from tests, builds, browsers, workflows, or server code.

When `providerAvailability` is `unavailable`, preserve the permanent CU page, summary, classifications, transcript, captions link, provenance, and boundary context. Do not delete CU-owned material because playback is unavailable.

## Publication assets

Create exactly this permanent directory pattern for an approved entry:

```text
website/public/media/<slug>/
  thumbnail.png
  transcript-en.txt
  captions-en.vtt
```

Do not commit MP4 masters. Keep masters and provider exports in the owner-controlled external Media archive.

The plain-text transcript is the readable canonical narration asset used during the static build. It must preserve the reviewed narration and approved terminology.

Before accepting a WebVTT candidate, verify:

- the first header is `WEBVTT`;
- every cue has a forward, nonzero time range;
- cue order never regresses and cues do not overlap;
- wording and CU terminology match the approved transcript review;
- the file uses LF endings and ends with exactly one LF newline;
- its SHA-256 exactly matches `captionSha256`.

The thumbnail and `posterAlt` require owner approval. Do not derive alt text automatically from the image filename or provider title.

## Source and epistemic review

Every `sourceBasis` item must have a nonempty label and a specific, truthful `establishes` statement.

- Internal links use a base-path-safe repository-relative `path`.
- External links use HTTPS `href`, a visible `disclosure`, and the existing safe external-link presentation.
- An empirical source does not validate a CU mathematical model or proposition.
- A CU theoretical document is not empirical proof.
- Philosophical or in-world material is not demonstrated engineering capability.

For CUCII or AI entries, preserve the boundary that structured prompt context does not by itself establish model retraining, unlocking, permanent alignment, machine consciousness, divine authority, supernatural memory, or cosmic access.

## Registry extension behavior

Do not edit presentation components merely because a record is added. The existing architecture is data-driven:

- `/media/` maps every `mediaLibrary` entry through `MediaCard`;
- `media/[slug].astro` maps every entry through `getStaticPaths`;
- poster, title, summary, runtime, classifications, status, transcript, captions, sources, and detail links derive from `MediaEntry`.

If a future record requires route-specific markup or a second registry, stop and request an architecture decision. Do not add a JSON mirror, database, remote content source, generated metadata authority, or fake production entry.

## Validation and review

From `website/`, run:

```sh
npm test -- src/lib/media/__tests__/media-authority.test.ts
npm test
npm run build
```

From the repository root, run:

```sh
git diff --check
git status --short --branch
git diff --stat
```

Confirm before requesting a Git-write checkpoint:

- IDs, slugs, and YouTube IDs are unique;
- all required fields are approved and valid;
- all three local assets exist under the exact slug directory;
- caption structure, timing, terminology, newline, and sealed digest pass;
- classifications and source statements pass owner review;
- the landing card and permanent detail route are generated from the registry;
- staging is not presented as public release;
- unavailable playback leaves the CU-owned page useful;
- no API, provider discovery, cache, polling, synchronization, credential, or Media workflow was introduced;
- no current record or sealed asset changed outside the approved scope.

Browser and release-readiness review remains a separately authorized checkpoint. Staging, committing, pushing, merging, tagging, deployment, provider visibility changes, and publication-status changes each require explicit owner authorization.

## Revision or replacement

For a revised upload representing the same publication, preserve the canonical slug, update only owner-approved provider fields and assets, advance the revision, and preserve prior correction/source history. For substantially different content, create a new approved entry and mark the prior entry `superseded` only when the owner authorizes that lifecycle change.
