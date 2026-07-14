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
input hash; the Markdown report is its concise review view.

The report is descriptive and cannot change `config/production.json` or deploy
a model.
