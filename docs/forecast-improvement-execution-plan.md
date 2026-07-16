# Forecast Improvement Execution Plan

This branch establishes the safe foundation for the first four improvement
workstreams. It does not promote a model or claim an accuracy gain before a
completed point-in-time comparison exists.

## Branch and delivery strategy

Use one integration branch for the shared contracts:

```text
codex/forecast-improvement-foundation
```

The evidence contract, experiment runner, Chronos ablation controls, and
fundamentals schema depend on one another and are reviewed together here.
Long-running experiment outputs remain outside Git under
`results/experiments/`. A model promotion must happen later in a separate,
small branch after its completed run has been verified and reviewed.

## Local prerequisites

All A0-A4 specs require the ignored, locally built price panel, its QA report,
and the historical weather-feature parquet at the paths declared in each spec.
The runner hashes those inputs, but it does not build or download missing
prerequisites.

A0 and A1 additionally require the exact production artifact at:

```text
artifacts/models/chronos_weather/sha256-55814a7fd0d36973
```

That artifact is not stored in Git. Its authoritative bootstrap source is the
private artifact store configured for production: deployment resolves
Terraform's `model_artifact_uri`, while the runtime derives the release-relative
path from `config/production.json`. The production workflow synchronizes that
URI before validating the artifact contract. Supplying the same artifact at
the local path above is therefore an operator prerequisite for A0/A1; these
experiment specs intentionally do not invent a credential- or
environment-specific acquisition command.

## Workstreams

### 1. Trustworthy model-selection evidence

- Enforce `information_cutoff_utc <= forecast_origin_utc`.
- Classify evidence as strict point-in-time, reconstructed proxy, or verified
  saved-live.
- Bind each evaluated model release and origin to a hashed availability audit.
- Reject post-outcome saved-live forecasts using the target price-availability
  timestamp, not merely the delivery timestamp.

### 2. Reproducible non-blocking experiments

- Resolve smoke/full profiles before computing experiment identity.
- Hash code, declared inputs, and environment.
- Write per-stage receipts and verify every declared output.
- Resume only from verified receipts.
- Launch workers detached; never wait in the interactive development session
  for training to finish.
- Write `COMPLETED.json` only after all stages and earlier outputs have been
  reverified.

### 3. Low-cost Chronos ablations

Prepare A0-A4 as research-only experiments:

```text
A0  ensemble mean, independent DK1/DK2
A1  A0 weights, cross_learning=True at inference
A2  A0 plus derived weather features
A3  ensemble mean plus ensemble spread
A4  cross-learning plus derived features plus ensemble spread
```

Smoke profiles prove the pipeline and artifact contracts with minimal work.
Full profiles may run detached only from a clean committed source identity.
Recent diagnostics are useful operational checks but are not substitutes for
frozen-origin, point-in-time model-selection evidence.

### 4. Point-in-time fundamentals v1

- Preserve provider documents, revisions, vintages, publication/retrieval
  timestamps, and availability semantics.
- Build exact UTC hours without mixing revisions.
- Encode provider, zone or directed border, parameter, and unit in every
  feature identity.
- Expose Danish and neighbouring-zone features to each DK target row.
- Derive provider/zone residual load and topology-validated complete
  directional capacity totals.

The current branch defines the offline contract only. ENTSO-E/Energinet API
ingestion and a production feature release remain separate follow-up work.

## Autonomous execution loop

For each experiment:

1. Run lint and focused unit tests.
2. Prepare a smoke run and verify its recorded identity.
3. Launch the smoke worker detached and return immediately.
4. Continue with code review, documentation, or the next independent task.
5. On the next development pass, inspect status and logs; verify completed
   receipts before using outputs.
6. Launch a full run only from a clean commit.
7. Never promote automatically. Compare the candidate on frozen,
   evidence-qualified origins and open a separate promotion change.

This keeps expensive compute off the interactive critical path and makes a
failed or interrupted run resumable without repeating already verified work.

## Deferred work

- Historical source ingestion and an expanded availability-audited replay.
- Capacity-weighted weather and expected-renewable-MW transforms.
- Chronos/CatBoost residual correction and convex ensembles.
- Neighbouring price histories and broader joint-market modelling.
- Structural flow-based or graph modelling.
