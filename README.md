# Forklift Telemetry Cleanup

This directory contains the complete toolkit for taking raw forklift telemetry
drops, adding canonical headers, cleaning the height signal, generating a
debounced carrying flag, and plotting or diagnosing the results. The typical
flow is:

1. `assign_columns.py` → write standardized headers + forklift/truck suffixes to `real_data_with_headers/`.
2. `clean_forklift_height.py` → bound or rescale the height channel into `real_data_with_headers_height_cleaned/`.
3. `add_carrying_feature.py` → append a stable `Carrying` column and store the outputs in `real_data_with_carrying/`.
4. Plot or score the resulting datasets with the visualization and analytics scripts (PNGs land under `plots/`).

Every script understands `uv run <script>.py --help`, so you can tweak paths or
thresholds as needed.

## Standardize the raw telemetry

### `assign_columns.py`
Adds the agreed header order (`FFFID, Height, Load, OnDuty, Timestamp, Latitude, Longitude, Speed`) to every raw CSV in `real_data/`, sorts by timestamp, drops redundant `OnDuty == 0` rows, infers whether the file is a forklift (any non-zero height) or truck, and writes the result to `real_data_with_headers/` as `<stem>_{forklift|truck}.csv`. Run `uv run assign_columns.py --data-dir … --output-dir …` to process another location.

## Height QA utilities

### `clean_forklift_height.py`
Scans a header-enriched directory (defaults to `real_data_with_headers/`), keeps only `_forklift.csv` files, enforces believable height bounds, and writes the cleaned rows to `real_data_with_headers_height_cleaned/`. Pick how outliers are handled with `--mode drop|clip|scale` (the scaling option linearly maps the provided sensor range into the target `--min-height/--max-height`). Use `--source-min`, `--source-max`, and `--overwrite` for finer control.

### `plot_forklift_height.py`
Reads each `_forklift.csv` from `real_data_with_headers/` and charts `Height` vs. `Timestamp` as vertical lollipop lines, saving PNGs to `plots/forklift_heights/`. This is the quickest way to spot unscaled sensors before you run the cleaning step.

### `plot_cleaned_forklift_height.py`
Identical plotter, but it targets `real_data_with_headers_height_cleaned/` and writes to `plots/forklift_heights_cleaned/`. Use it to confirm `clean_forklift_height.py` produced smooth trajectories.

## Load and carrying features

### `plot_forklift_load.py`
Turns every `_forklift.csv` in `real_data_with_headers/` into a two-panel PNG in `plots/forklift_loads/`. The top timeline shows contiguous load intervals, while the bottom bar chart summarizes the share of time carrying per resampled bucket (`--daily-frequency` defaults to `1D`). Perfect for a quick utilization audit right after header assignment.

### `add_carrying_feature.py`
Creates a debounced boolean `Carrying` column by combining the raw `Load` flag with fork-height and max-speed heuristics, rejecting short pulses (default 45 s), optionally closing tiny gaps, and skipping off-duty rows unless `--allow-off-duty` is provided. Outputs land in `real_data_with_carrying/`, and every threshold (`--min-duration`, `--fill-gap`, `--height-threshold`, `--load-threshold`, `--max-speed`) is configurable.

### `plot_carrying_feature.py`
Consumes the annotated files from `real_data_with_carrying/` and renders the carrying timeline plus resampled share-of-time bars into `plots/carrying/`. Use `--resample` to change the aggregation window (daily by default).

### `plot_carrying_diagnostics.py`
Builds richer diagnostics for each carrying file and writes them to `plots/carrying_diagnostics/`. Every figure stacks: (1) raw load vs. `Carrying` timeline, (2) a histogram of carrying durations, and (3) daily carrying share with segment counts. Customize resolution (`--dpi`) and histogram bins (`--duration-bins`) as needed.

## Sensor reliability & anomaly detection

### `evaluate_load_sensor.py`
Summarizes load-sensor health for every `_forklift.csv` in `real_data_with_headers/`. It computes total logged hours, duty cycle, toggle rate, how many load segments shorter than `--short-threshold`, plus median loaded/unloaded durations. The table is printed and saved to `plots/load_sensor_summary.csv` unless you override `--output`.

### `detect_load_anomalies.py`
Extends the above metrics with fast-load-flip analysis (how often the load state switches while speed exceeds `--speed-threshold`) and flags machines whose short-load ratio, toggle rate, or fast-flip rate breach the supplied limits. Results are reported in the console and optionally exported via `--output path/to/anomalies.csv`.

### `plot_load_anomalies.py`
Visual companion to the anomaly detector. It recomputes the metrics, overlays the short-load and fast-flip thresholds on a scatter plot (color = toggles/hour), then draws a ranking bar chart that highlights flagged machines. The combined figure defaults to `plots/load_anomaly_visualization.png`, and all threshold knobs mirror the detector CLI.

## Safety & wear monitoring

### `monitor_safety.py`
Analyzes each `_forklift.csv` inside `real_data_with_headers_height_cleaned/` with only the cleaned `Height` and `Speed` columns, so flaky load sensors do not matter. For every file it:

1. Converts timestamps to durations so we know how long each sample lasts.
2. Marks a sample as “raised” when `Height > --raised-threshold` (0.5 m default) and as “tall” when `Height > --tall-threshold` (3.5 m default).
3. Flags overspeed events when raised forks coincide with `Speed > --overspeed-limit` (8 km/h default) and tall-lift-while-moving events when tall forks coincide with `Speed > --tall-move-speed` (1 km/h default).
4. Counts height oscillations by measuring how often the raised state flips per hour.

For each behavior it reports total events, percent of logged time, and per-hour rates, then compares them against user-provided alert limits (`--overspeed-limit-rate`, `--tall-move-limit-rate`, `--toggle-limit`). Any breaches are listed in the `alerts` column, and the full CSV saves to `plots/safety_monitor.csv` unless `--output` overrides it. Run `uv run monitor_safety.py --help` to tailor thresholds to your site’s policy.

### `plot_safety_monitor.py`
Recomputes the same metrics and renders them as a scatter plot (overspeed/hr vs. tall-lift/hr, color-coded by toggle rate with threshold lines) plus a bar chart ranking machines by height oscillations. Alerts are highlighted in red so risky forklifts stand out immediately. Run `uv run plot_safety_monitor.py` to save `plots/safety_monitor.png`, overriding thresholds or figure DPI with the usual CLI flags.
