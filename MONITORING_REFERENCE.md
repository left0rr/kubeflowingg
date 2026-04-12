# Monitoring Reference

This file documents the monitoring workflow on `feature/monitoring-stack` for:

- `simulate_trafic.py`
- `monitoring/metrics_exporter.py`
- `monitoring/drift_detection.py`

Examples below assume you run commands from the repo root.

## 1. Traffic Simulator

Script:

```bash
python simulate_trafic.py [options]
```

Parameters:

- `--kserve-url`
  - Default: `http://localhost:8085/v1/models/gpon-failure-predictor:predict`
  - KServe prediction endpoint.
- `--baseline`
  - Default: `data/processed/processed.csv`
  - Baseline processed dataset used for replay-style traffic generation.
- `--output`
  - Default: `data/predictions/latest.csv`
  - Output CSV file that stores sent feature rows and predictions.
- `--sleep-seconds`
  - Default: `2.0`
  - Delay between requests sent to KServe.
- `--mode`
  - Choices: `auto`, `baseline-replay`, `synthetic`
  - Default: `auto`
  - `auto` uses `baseline-replay` when the baseline file is usable, otherwise falls back to `synthetic`.
- `--drift-profile`
  - Choices: `none`, `gradual-stress`, `temperature-shift`, `error-burst`
  - Default: `none`
  - Applies controlled drift after `--drift-start-after`.
- `--drift-start-after`
  - Default: `120`
  - Number of successful requests to send before drift starts.
- `--drift-rate`
  - Default: `0.35`
  - Probability that a replayed row gets drift applied after drift starts.
- `--drift-strength`
  - Default: `1.0`
  - Intensity multiplier for the drift profile.

Recommended commands:

Baseline-like traffic, no drift:

```bash
python simulate_trafic.py \
  --mode baseline-replay \
  --drift-profile none \
  --sleep-seconds 1
```

Gradual realistic drift:

```bash
python simulate_trafic.py \
  --mode baseline-replay \
  --drift-profile gradual-stress \
  --drift-start-after 60 \
  --drift-rate 0.4 \
  --drift-strength 1.0 \
  --sleep-seconds 1
```

Temperature-focused drift:

```bash
python simulate_trafic.py \
  --mode baseline-replay \
  --drift-profile temperature-shift \
  --drift-start-after 60 \
  --drift-rate 0.5 \
  --drift-strength 1.2
```

Error-count burst drift:

```bash
python simulate_trafic.py \
  --mode baseline-replay \
  --drift-profile error-burst \
  --drift-start-after 60 \
  --drift-rate 0.5 \
  --drift-strength 1.2
```

Synthetic fallback mode:

```bash
python simulate_trafic.py \
  --mode synthetic \
  --drift-profile none
```

Notes:

- If `data/processed/processed.csv` is missing or missing required features, `auto` will fall back to `synthetic`.
- For cleaner drift tests, clear old predictions before a new run:

```bash
rm -f data/predictions/latest.csv
```

## 2. Metrics Exporter

Script:

```bash
python -m monitoring.metrics_exporter [options]
```

Parameters:

- `--predictions`
  - Default: `data/predictions/latest.csv`
  - Predictions CSV to monitor.
- `--port`
  - Default: `8000`
  - HTTP port for the Prometheus `/metrics` endpoint.
- `--interval`
  - Default: `30`
  - Refresh interval in seconds.
- `--window-rows`
  - Default: `300`
  - Use only the most recent `N` rows.
- `--window-minutes`
  - Default: unset
  - Use only rows from the last `N` minutes when timestamps are present and parseable.

Exported metrics:

- `prediction_failure_ratio`
- `prediction_window_sample_size`

The exporter uses `predicted_failure_label` first, then falls back to `Failure_In_7_Days`.

Recommended commands:

Basic row-window exporter:

```bash
python -m monitoring.metrics_exporter \
  --predictions data/predictions/latest.csv \
  --port 8000 \
  --interval 10 \
  --window-rows 300
```

Time-window exporter:

```bash
python -m monitoring.metrics_exporter \
  --predictions data/predictions/latest.csv \
  --port 8000 \
  --interval 10 \
  --window-minutes 15
```

Quick check in browser or terminal:

```bash
curl http://localhost:8000/metrics
```

## 3. Prometheus Queries

The scrape job in `monitoring/prometheus.yml` is `gpon_model_metrics`.

Useful queries:

- `up{job="gpon_model_metrics"}`
- `prediction_failure_ratio`
- `prediction_failure_ratio{job="gpon_model_metrics"}`
- `prediction_window_sample_size`
- `avg_over_time(prediction_failure_ratio[5m])`
- `max_over_time(prediction_failure_ratio[15m])`
- `min_over_time(prediction_failure_ratio[15m])`
- `avg_over_time(prediction_window_sample_size[10m])`
- `(prediction_failure_ratio > 0.2) and (prediction_window_sample_size >= 100)`

## 4. Drift Detection

Script:

```bash
python -m monitoring.drift_detection [options]
```

Parameters:

- `--baseline`
  - Default: `data/processed/processed.csv`
  - Baseline reference dataset, usually your processed training data.
- `--current`
  - Required
  - Current production-style dataset, usually `data/predictions/latest.csv`.
- `--output`
  - Default: `monitoring/reports/drift_report.html`
  - Output HTML report path.
- `--include-target`
  - Default: off
  - Includes target and metadata columns in the drift analysis.
- `--current-window-rows`
  - Default: `500`
  - Uses the most recent `N` rows from the current dataset.
- `--current-window-minutes`
  - Default: unset
  - Uses only rows from the last `N` minutes when timestamps are present.
- `--min-current-rows`
  - Default: `100`
  - Minimum number of rows required before a drift decision is made.

Important behavior:

- By default, the script excludes:
  - `Failure_In_7_Days`
  - `predicted_failure_label`
  - `timestamp`
  - `prediction_score`
  - `true_status`
  - `source_mode`
  - `drift_profile`
  - `drift_applied`
- Drift is calculated only on overlapping feature columns between baseline and current data.
- The script exits with code `1` if dataset drift is detected.

Recommended commands:

Basic drift report using recent rows:

```bash
python -m monitoring.drift_detection \
  --baseline data/processed/processed.csv \
  --current data/predictions/latest.csv \
  --output monitoring/reports/drift_report.html \
  --current-window-rows 300 \
  --min-current-rows 100
```

Time-window drift report:

```bash
python -m monitoring.drift_detection \
  --baseline data/processed/processed.csv \
  --current data/predictions/latest.csv \
  --output monitoring/reports/drift_report.html \
  --current-window-minutes 15 \
  --min-current-rows 100
```

Include target and metadata columns:

```bash
python -m monitoring.drift_detection \
  --baseline data/processed/processed.csv \
  --current data/predictions/latest.csv \
  --output monitoring/reports/drift_report.html \
  --current-window-rows 300 \
  --min-current-rows 100 \
  --include-target
```

## 5. Suggested Test Flow

Open a KServe port-forward first if needed:

```bash
kubectl port-forward -n kserve svc/gpon-failure-predictor-predictor 8085:80
```

Then run:

```bash
rm -f data/predictions/latest.csv
```

```bash
python simulate_trafic.py \
  --mode baseline-replay \
  --drift-profile none \
  --sleep-seconds 1
```

In another terminal:

```bash
python -m monitoring.metrics_exporter \
  --predictions data/predictions/latest.csv \
  --port 8000 \
  --interval 10 \
  --window-rows 300
```

In another terminal:

```bash
python -m monitoring.drift_detection \
  --baseline data/processed/processed.csv \
  --current data/predictions/latest.csv \
  --output monitoring/reports/drift_report.html \
  --current-window-rows 300 \
  --min-current-rows 100
```

To test realistic drift later, rerun the simulator with:

```bash
python simulate_trafic.py \
  --mode baseline-replay \
  --drift-profile gradual-stress \
  --drift-start-after 60 \
  --drift-rate 0.4 \
  --drift-strength 1.0 \
  --sleep-seconds 1
```
