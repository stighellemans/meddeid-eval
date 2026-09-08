# Changelog

All notable user-visible changes are recorded here. This project follows
semantic versioning while pre-1.0 versions may still refine public contracts.

## [Unreleased]

## [0.5.0] - 2026-09-08

- Added a paper-style evaluation battery that selects the registered public
  baseline from the data language profile and emits reproducible reports,
  tables, figures, confusion matrices, subannotation coverage, and paired
  document-bootstrap intervals.
- Added matched-span label accuracy and explicit missed/spurious outcomes,
  expanded primary-label and subannotation plots, and strengthened score-input
  validation and regression coverage.

## [0.4.0] - 2026-09-05

- Added aggregate-only end-to-end date/age pseudonymization evaluation from
  saved predicted spans, including protocol failure and residual-exposure rates.

## [0.3.0] - 2026-08-27

- Refactored stability name/date helpers around a selected locale provider and
  added exact `en-GB` and `en-US` support while rejecting bare `en`.

## [0.2.2] - 2026-08-18

- Added privacy-safe detailed score tables and non-PII redaction metrics.
- Added comparison plotting with accessible recall, non-PII, and exact-label
  confusion heatmaps, runtime plots, and PNG/PDF/SVG export.
- Upgraded stability inference to note-cluster bootstrap and paired
  note-cluster permutation statistics.
- Added cross-scope Benjamini-Hochberg adjustment with immutable raw analyses
  and audit exports.
- Replaced quick-look stability plots with role-safe, missing-safe,
  publication-grade figures and vector output.

## [0.2.1] - 2026-08-17

- Published the first externally supported MedDeID evaluation release.
- Added public installation, compatibility, licensing, and verification
  metadata.
- Established independent CI and immutable release artifacts.

For earlier migration history, consult the repository's Git history.
