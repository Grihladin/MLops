# Forklift Dagster Pipeline

This package turns the ad-hoc telemetry notebook into a reproducible Dagster
asset graph. It ingests data from LakeFS, reuses the existing cleaning scripts,
and trains a carry-state classifier that is versioned alongside metrics.

## Layout

```
pipeline/
├─ dagster.yaml                # Workspace definition (``dagster dev``)
├─ assets/                     # Dagster assets (ingest, prep, training, reports)
├─ resources/                  # Configurable LakeFS + model registry helpers
├─ io_managers/                # Custom IO managers for artifacts and DataFrames
├─ sensors/                    # LakeFS commit sensor -> asset job
├─ utils/                      # Shared helpers / dataclasses
├─ artifacts/                  # Created at runtime; stores intermediate CSVs/parquet
├─ models/                     # Stored pickled estimators
└─ metrics/                    # JSON performance reports per run
```

## Configuring LakeFS

The `LakeFSResource` is wired as the `lakefs` resource in `repo.py`. Provide the
connection parameters via Dagster run config or environment variables:

```yaml
resources:
  lakefs:
    config:
      endpoint_url: "https://lakefs.example.com"
      repo: forklift-data
      branch: main
      access_key: ${LAKEFS_ACCESS_KEY}
      secret_key: ${LAKEFS_SECRET_KEY}
      prefix: real_data
```

For local development without a LakeFS server you can set
`fallback_local_path` to a directory containing `real_data/`. The resource will
copy files from that location instead of making API calls.

## Running the pipeline

1. Install dependencies (handled via `uv pip sync` or your preferred tool).
2. Start Dagster: `uv run dagster dev -m pipeline`.
3. Trigger `forklift_training_job` manually or turn on the
   `lakefs_new_commit_sensor` to launch runs on new commits.

Artifacts land under `pipeline/artifacts/` and are run-scoped so multiple runs
can coexist. Models are serialized into `pipeline/models/` with filenames tied to
`run_id`. Metrics are emitted as JSON under `pipeline/metrics/` for downstream
reporting.

## Quality gates

* Ruff + Pyright cover the repository via VS Code / CLI.
* Custom IO managers persist every asset output so runs are fully replayable.
* The training asset enforces deterministic splits (configurable `random_state`).
* LakeFS sensors keep orchestration aligned with new data arriving on the
  configured branch.
