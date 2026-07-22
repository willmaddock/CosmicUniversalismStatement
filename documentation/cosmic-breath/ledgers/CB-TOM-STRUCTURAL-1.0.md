# CB-TOM-STRUCTURAL-1.0

## Approval record

- **Ledger identifier:** `CB-TOM-STRUCTURAL-1.0`
- **Owner and final approval authority:** William Maddock
- **Owner approval date:** 2026-07-22
- **Status:** Owner-approved structural decisions; numerical duration, formula, quantum-notation, magnitude, and chronology-sensitive fields withheld
- **Canonical JSON SHA-256:** `dcc34d5b7d9e32afb8d0cb97b029a12e0025959095a21b4168d59aff575da825`

## Approved structural decisions

The canonical structural ledger contains exactly 51 selectable TOM states: 26 expansion states followed by 25 compression states. Selectable-state `phase` is limited to `expansion` and `compression`. Boundary semantics are represented independently by `boundaryRole`.

The approved expansion order is sub-ztom, sub-ytom, sub-xtom, sub-wtom, sub-vtom, sub-utom, sub-ttom, sub-stom, sub-rtom, sub-qtom, sub-ptom, sub-otom, sub-ntom, sub-mtom, sub-ltom, sub-ktom, sub-jtom, sub-itom, sub-htom, sub-gtom, sub-ftom, sub-etom, sub-dtom, sub-ctom, sub-btom, and atom.

The approved compression order is btom, ctom, dtom, etom, ftom, gtom, htom, itom, jtom, ktom, ltom, mtom, ntom, otom, ptom, qtom, rtom, stom, ttom, utom, vtom, wtom, xtom, ytom, and ztom.

Approved boundary roles:

- `expansion-sub-ztom`: `new-cosmic-seed`
- `expansion-atom`: `expansion-pause`
- `compression-ztom`: `reset-pause`
- All other selectable states: `null`

State order, labels, phase membership, complete-cycle indices 1–51, phase indices, adjacency, classification, approval status, and provenance are owner-approved structural fields. The structural timeline and any native range control use indices 0–50, not time-proportional spacing.

## Declared phase anchors

The following are approved only as declared CU model-level anchors:

- Expansion: approximately 2.8 trillion years
- Compression: approximately 308 billion years
- Complete Cosmic Breath: approximately 3.108 trillion years

These values are not derived from, and must not be described as sums of, individual TOM rows.

## Guarded transitions

Two transitions are approved as explicit, non-selectable records:

1. `transition-atom-to-btom` pauses at atom and requires the deliberate **Begin Compression** action before entering `compression-btom`.
2. `transition-ztom-to-next-sub-ztom` pauses at ztom and requires the deliberate **Begin the Next Cosmic Breath** action. It targets `expansion-sub-ztom` with a cycle offset of 1.

Ordinary `previousId` and `nextId` adjacency does not cross either guarded boundary.

## Withheld fields and unresolved questions

The initial production explorer must not include per-state:

- quantum notation or tetration heights;
- exponential or tetration formula indices;
- durations or duration interpretations;
- machine-readable magnitudes or year conversions;
- time-proportional positions or duration-derived percentages;
- chronology-sensitive educational descriptions.

The displayed 280-billion-year sub-btom value, displayed 28-billion-year sub-ctom value, ATOM-as-Planck-time mapping, one-second ztom/sub-ztom interpretations, factor-two disputes, tetration-index disputes, and omitted-letter disputes remain withheld for later mathematical review.

Original descriptions remain provisional source annotations. The known sub-ftom chronology contradiction remains an unresolved CU research question and is not corrected by reordering states.

## Source provenance

- **CBC Phase I:** `ResearchFiles/Cosmic_Breath_Calculation.md`, expansion source inventory.
- **CBC Phase II:** `ResearchFiles/Cosmic_Breath_Calculation.md`, compression source inventory.
- **BC:** `ResearchFiles/Cosmic_Breathing_Cycle.md`, Cosmic Breath narrative and sub-ztom seed interpretation.
- **TC:** `ResearchFiles/Time_Calculation.md`, disputed formulas and numerical mappings retained only as review provenance.
- **FM:** Cosmic Breath Interactive Cycle Explorer Implementation and Operations Manual v1.1, especially §§3.6, 8.9, and Appendix B.
- **P0B:** William Maddock's Phase 0B owner decisions and subsequent phase-membership correction.
- **P0C:** William Maddock's final approval dated 2026-07-22.

The uploaded/project research files were previously verified byte-for-byte against their repository copies.

## Canonical artifact

The canonical machine-readable artifact is:

`website/src/data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json`

Its standard SHA-256 record is:

`dcc34d5b7d9e32afb8d0cb97b029a12e0025959095a21b4168d59aff575da825  website/src/data/cosmic-breath/CB-TOM-STRUCTURAL-1.0.json`
