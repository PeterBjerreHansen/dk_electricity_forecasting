# Evidence provenance

This directory records the reviewed descriptive comparison for production
release `sha256-55814a7fd0d36973`.

The prediction input was an exported snapshot of the private, versioned
dashboard history:

```text
S3 object: dk-energy-forecasts/dashboard/forecast_history.parquet
S3 version: 0vv66NNCfN3lHxmqz10WYQLrJdDF7g8A
SHA-256: 55f45dfeae5546a7623963400edb4afe79089f2b277149ebc224ca258221e01e
```

The selected rows are historical-origin forecasts for the exact configured
Chronos release and the fixed weighted-median reference. They are initial
evaluation evidence, not a claim of 24 live production days. The JSON report
contains the complete settings, paired-origin results, stratification, and
input hash; the Markdown report is its concise review view. Because that
historical report uses the older `model_comparison_v1` schema, its evidence
classification is stored separately in `evidence.json` instead of retrofitting
new fields into the old report schema. The sidecar records the report's relative
path, schema, and SHA-256. The
`dkenergy_forecast.evaluation.evidence_sidecar.validate_model_evidence_sidecar`
validator fails if the report is moved, changed, or replaced without reviewing
and updating the declaration.

Evidence tier: `reconstructed_proxy`. Price availability is a project
convention, price and weather source revisions use latest-retrieved snapshots,
and weather availability is synthetic. This report must not be described as a
strict point-in-time replay.

The report is descriptive and cannot change `config/production.json` or deploy
a model.
