30-deg step load comparison package

Experiments
- No load: 2026-09-15 00:55:51
- 400 g: 2026-09-15 01:42:15
- 800 g: 2026-09-15 01:46:31
- 1200 g: 2026-09-15 01:50:51

Common conditions
- External target: 30 deg
- Policy target scale: 1.0
- Policy output: direct raw command
- Timing: 1 s zero-target hold followed by 4 s of step control
- Processed time origin: the first recorded 30-deg target sample
- Processed data use recorded samples directly; no interpolation was applied
- The no-load run completed and saved valid data, but its source MAT report records passed=0 for the combined gate; the other three runs record passed=1

Package contents
- data/original: complete source CSV files and MATLAB MAT files
- data/processed: post-step CSV files with time reset to 0 s; policy_target omitted
- data/summary_metrics.csv: comparable result metrics and source checksums
- figures/individual: original paper-format figures for each load
- figures/comparison: unified 2x2 comparison in PNG and PDF
- scripts: plotting scripts used for the individual and comparison figures

Comparison figure
- (a) Link trajectory tracking for all loads, with an inset covering x=0.5-2.0 s and y=25-35 deg
- (b) Motor-side angle
- (c) Policy network output in raw units
- (d) Tracking error, defined as target angle minus link angle
