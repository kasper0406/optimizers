# criteria/ — HUMAN-AUTHORED ONLY

This directory holds pre-registered gates, tolerances, and success criteria as
versioned files (e.g. `airbench_tolerance.yaml`, `nanogpt_tolerance.yaml`,
`phase1_preregistration.md`, `phase2_success.yaml`).

Rules (see CLAUDE.md):

- All files here are authored by the human. Agents must not create or modify
  criteria content.
- Agents may at most draft `*.template` files with blank/TODO thresholds,
  clearly headed "HUMAN MUST AUTHOR".
- Changing a metric definition, tolerance, or success criterion requires an
  explicit human instruction referencing the file.
- WP1.2 (Phase-1 measurement runs) is blocked until the human has committed
  `criteria/phase1_preregistration.md`, and launch tooling verifies that the
  file predates the first run.
