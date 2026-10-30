# LocalGaze 眼动数据流程

以下内容来自当前 LocalGaze 源码；切换项目版本或代码副本后应重新核对。这里的阈值是项目候选设置，不是经普遍验证的标准。

## Processing path

1. Preserve original input read-only. Readers support `.xlsx`, `.xlsm`, `.csv`, `.tsv`, `.txt`. Excel prefers sheets `眼动数据`, `EyeData`, `GazeData`, `gaze_data`, then tries the rest. Column aliases cover time, X/Y, blink, left/right pupil. Resolve ambiguous columns explicitly.
2. Preprocessing supports numeric time in seconds, milliseconds, microseconds. It scales and rebases numeric time to the first row; parsed datetime is also rebased. Preserve the original absolute time and timezone separately if needed. Backward time is an error: split trials/segments. Analysis `time_unit=auto` still needs a recognized unit in a numeric-time header or an explicit override.
3. Desktop mapping preprocess job calls `preprocess_mapping_table(..., preset="analysis")`; single-source route calls `preprocess_analysis_paths`, also `analysis`. This disables XY interpolation. Public path API and CLI default `legacy`, which may interpolate XY if a matched model provides calibration error. These routes are not equivalent.
4. Preprocessing preserves rows/source format, appends derived fields, and writes QC report/manifest, per-file QC sidecar, and QC tables. Inspect these artifacts, not only a dialog message.
5. `read_analysis_gaze` uses raw gaze fields; it deliberately will not use display-smoothed `Pre_X`/`Pre_Y`. It requires explicit accepted/valid status and excludes interpolated coordinates and bad segments. It preserves input columns/order and appends canonical `analysis_x_raw`, `analysis_y_raw`, `analysis_time_s`, `analysis_gaze_valid`, `analysis_break_before`, `analysis_segment_id`, `analysis_trial_id`, `analysis_stimulus_id`.
6. `analysis_pipeline.analyze_file` reads normalized data, computes descriptive metrics/event candidates/ROI metrics, and exports tables. It is not a general hypothesis-testing engine.

## PreprocessConfig parameter inventory

Defaults are engineering defaults, not universally validated thresholds. The desktop preset serializes only seven controls; do not add the hidden fields to strict desktop JSON.

| Parameter | Current default | Meaning / unit | In desktop preset? |
|---|---:|---|---|
| `time_unit` | seconds | Numeric timestamp unit | Yes |
| `coordinate_bounds` | (0,2559,0,1439) | Pixel-center rectangle; never treat as monitor fact | No |
| `blink_padding_s` | 0.050 | Blink expansion, seconds | Yes |
| `max_time_gap_s` | None | Gap threshold, seconds | Yes |
| `max_interpolation_gap_s` | 0.050 | Max XY interpolation gap, seconds | No |
| `max_interpolation_distance` | None | XY distance, pixels; legacy may set from calibration error | No |
| `smooth_window_s` | 0.060 | Median smoothing window, seconds | Yes |
| `smooth_max_points` | 5 | Max samples in smoothing | Yes |
| `velocity_mad_multiplier` | 6.0 | Robust spike rule multiplier | No |
| `quality_window_s` | 0.500 | Quality window, seconds | Yes |
| `quality_step_s` | 0.100 | Quality step, seconds | Yes |
| `missing_ratio_threshold` | 0.50 | Missing/out-of-bounds fraction, 0–1 | Yes |
| `round_trip_ratio_threshold` | 0.30 | Round-trip spike fraction | No |
| `noise_mad_multiplier` | 6.0 | Robust noise multiplier | No |
| `min_bad_segment_s` | 0.200 | Minimum suspect segment, seconds | No |
| `bad_segment_merge_gap_s` | 0.100 | Mask merge gap, seconds | No |
| `pupil_max_interpolation_gap_s` | 0.075 | Pupil gap max, seconds | No |
| `pupil_artifact_padding_s` | 0.050 | Pupil artifact padding, seconds | No |
| `pupil_dilation_mad_multiplier` | 6.0 | Pupil change threshold multiplier | No |
| `pupil_filter_cutoff_hz` | 4.0 | Time-aware Gaussian filter cutoff, Hz | No |

Desktop exposes `blink_padding_s`, `max_time_gap_s`, `smooth_window_s`, `smooth_max_points`, `quality_window_s`, `quality_step_s`, `missing_ratio_threshold`. Every setting and chosen execution path/preset belongs in the parameter ledger. For hidden values either use an inspected/version-checked Python API or record the actual default; never claim desktop JSON applied them.

The detector uses timestamp continuity, segments, explicit validity, coordinate bounds, blink and pupil values, robust MAD rules, and rolling quality windows. It computes missing/out-of-bounds ratio, RMS successive displacement, local MAD and round-trip spike ratio. Missing ratio above the threshold marks bad candidates; high variation jointly with high round-trip ratio marks suspect-noise candidates. Robust thresholds derive from internal windows, not an external gold standard. Inspect window/interval tables and report counts/durations. These are candidate QC rules, not validated universal cutoffs.

XY interpolation is limited to an internal unprotected gap within one continuous segment, bounded by finite neighbors, gap duration and distance. `analysis` disables it. Pupil processing is separate: nonpositive/missing values, blink/bad spans, rapid pupil changes and padding are masked; only short in-segment pupil gaps may be interpolated, then a time-aware Gaussian filter may run when sample rate is over twice its cutoff. Validity, artifact and interpolation masks are output. Never treat pupil interpolation as measured data. Never interpolate across trial/event boundaries or long clock gaps.

## Segments and event features

Analysis pairs are valid only when adjacent rows share segment/trial/stimulus, both samples are explicitly valid, times are finite/increasing, and the gap fits the configured/derived threshold. Runs split on invalid samples and gaps. Exposure is actual elapsed `dt` over valid pairs, not row count divided by nominal FPS. Do not bridge missing intervals for analysis.

`analysis.metrics` accepts these exact IDs:

- `trajectory`: coordinate distributions, path length, speed, segment displacement. Units px and px/s until separately transformed.
- `fixation`: dispersion/minimum-sample/minimum-duration candidate; requires known screen diagonal; not ground truth.
- `fast_movement`: screen-diagonal-scaled speed candidate.
- `ivt_saccade`: I-VT speed-threshold candidate; requires verified screen reference.
- `blink`: EAR closure candidate, not confirmed blink classification.
- `roi`: ROI hits, sampled durations/visits and censoring.
- `iris_proxy`: image-derived iris ratio, not pupil diameter or truth.
- `entropy`: occupancy over a grid whose dimensions/bounds are method settings.
- `calibration`: training residual, not independent validation.

Outputs are descriptive/candidate records with status, reason, method and sometimes boundary censoring. Empty/not-computable is not zero. Enabling a group does not convert candidate status into confirmatory inference.

## ROI chain

`roi_contract.py` validates unique IDs/names, shape/geometry, nonnegative priority, boundary semantics, source, coordinate reference, optional mapping, and schema version. Shapes: rectangle, circle, ellipse. New ROIs use inclusive boundaries; strict is reserved for migrated legacy rectangles. Overlaps may yield multiple hits; priority selects a primary hit and all hits are retained. Visit duration is sampled left-hold, not exact crossing time; run-edge visits are censored and excluded from complete-dwell estimates.

ROI is not-computable when gaze reference is unverified, screen dimensions invalid/missing, an ROI reference is unknown, or coordinate frames conflict. Never rescale an ROI because a preview looks plausible. Record monitor/origin/DPI/mapping and verify that gaze and ROI share the same frame. Loading an ROI workbook does not prove its reference is correct.

**ROI 时长的边界：**当前 ROI 输出中的 `observed_duration_s`、`complete_dwell_mean_s` 是 ROI 占用/访问时长，不等于 ROI 内注视时长。若研究主结局是 ROI fixation duration，必须生成并验证一个独立“注视候选 × ROI 命中”派生脚本，预先规定重叠 ROI 计数、跨 ROI 边界的事件拆分/归属、被截断注视、无效 gaze、采样间隔及分母，并保留 event 与 ROI source IDs。当前内置描述表不能直接作为 ROI 注视时长。如无可验证的定义和脚本，应阻断该主结局。
