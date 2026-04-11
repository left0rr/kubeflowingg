# Retraining Foundations

This folder documents the next natural step after monitoring and Grafana:

- detect meaningful drift
- decide whether retraining should happen
- submit a new Kubeflow run only when the trigger rules pass

The goal is to make retraining controlled and explainable, not “fully automatic at any cost.”

## Why Retraining Is The Natural Next Step

Your platform already has:

- KServe deployment
- traffic simulation
- metrics exporter
- drift detection
- Grafana dashboards
- model promotion tooling

That means the next missing link is the decision layer:

```
Drift observed -> Should we retrain? -> If yes, start pipeline run
```

Without this step, the platform can observe problems but cannot respond to them.

## What This Branch Adds

- [retraining_trigger.py](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/monitoring/retraining_trigger.py)
  - evaluates drift metrics and decides whether retraining should be triggered
  - runs in dry-run mode by default
  - can optionally submit a KFP run
  - writes its own Prometheus textfile metrics
- [retraining_config.example.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/retraining/retraining_config.example.yaml)
  - example config file for thresholds and pipeline parameters

## High-Level Flow

```text
1. drift_detection.py runs
2. it writes HTML report + Prometheus metrics
3. retraining_trigger.py reads those metrics
4. it applies trigger rules
5. if rules pass and --submit is enabled:
   - a new Kubeflow pipeline run is submitted
6. after the training run succeeds:
   - you can promote the new champion and redeploy
```

## Important Design Choice

This branch does **not** automatically redeploy a new model the moment drift is detected.

That is intentional.

A safer order is:

1. detect drift
2. trigger retraining
3. let the pipeline enforce the quality gate
4. compare candidate vs champion
5. promote/deploy only if the candidate is truly better

This avoids “drift happened -> redeploy anything” behavior.

## Trigger Rules

The starter trigger logic uses these signals from `drift_detection.prom`:

- `gpon_drift_check_success`
- `gpon_data_drift_detected`
- `gpon_drifted_columns_count`
- `gpon_drifted_columns_fraction`

Default thresholds:

- require dataset drift: `true`
- minimum drifted columns: `3`
- minimum drifted fraction: `0.30`
- cooldown: `120` minutes

## Dry Run First

The safest way to use the trigger is:

```bash
python -m monitoring.retraining_trigger
```

That evaluates the decision and writes retraining metrics without starting a pipeline run.

Only when you are happy with the thresholds should you submit a run:

```bash
python -m monitoring.retraining_trigger --submit
```

## Example Commands

### 1. Refresh drift metrics

```bash
python -m monitoring.drift_detection \
  --baseline data/processed/processed.csv \
  --current data/predictions/latest.csv \
  --output monitoring/reports/drift_report.html
```

### 2. Evaluate retraining trigger in dry-run mode

```bash
python -m monitoring.retraining_trigger \
  --config retraining/retraining_config.example.yaml
```

### 3. Submit a retraining run when ready

```bash
python -m monitoring.retraining_trigger \
  --config retraining/retraining_config.example.yaml \
  --submit
```

## Output Metrics

The trigger writes `monitoring/prometheus_textfile/retraining_trigger.prom`.

That file contains metrics such as:

- `gpon_retraining_should_trigger`
- `gpon_retraining_submission_attempted`
- `gpon_retraining_submission_success`
- `gpon_retraining_last_decision_timestamp_seconds`
- `gpon_retraining_last_submission_timestamp_seconds`
- `gpon_retraining_drifted_columns_count`
- `gpon_retraining_drifted_columns_fraction`
- `gpon_retraining_cooldown_active`

These metrics are suitable for future Grafana panels.

## What Comes After This Foundation

After you test this branch successfully, the next retraining improvements should be:

1. add Grafana panels for retraining trigger status
2. connect the trigger to a scheduler or CronJob
3. automatically run `promote_champion.py` after successful retraining and validation
4. add notification hooks for “triggered / not triggered / failed”
