# Data conversion checklist

Every transformation needs a ledger row: ID; input/output fields; exact operation/order; source; before/after units; parameters; reason/source; row/sample counts; missing/collision policy; expected invariant; observed check; status. Preserve original inputs and write copies to new paths. Maintain source lineage; never relabel derived values as measured raw fields.

## File parsing

- Supported gaze formats: XLSX/XLSM, CSV, TSV, TXT. Text reader tries UTF-8 BOM then GB18030 and sniffs comma/tab/semicolon/vertical-bar (suffix fallback); Excel reader searches preferred sheets then remaining sheets. Mapping table allows CSV/TSV/XLSX/XLSM with required `data_path` and optional `model_path`, `session_id`; relative paths are relative to the mapping file.
- Confirm encoding, delimiter, decimal mark, sheet, header, field names and duplicated/ambiguous columns. Delimiter sniffing does not ensure decimal comma works. If locale uses decimal comma, convert a copy with a documented locale-aware rule; verify `1,25` is parsed as 1.25, not split or scaled. Keep source text/hash.
- Record timezone and absolute clock origin. LocalGaze converts timestamps to elapsed values from the first row; preserve the original absolute timestamp/timezone separately. Do not assume DST or device/video/trigger synchronization.
- Preserve nanosecond integers as text to avoid float rounding above 2^53. Check monotonicity, duplicates, cadence/gaps and segment breaks. Convert time using seconds ×1, milliseconds ×1e-3, microseconds ×1e-6. Never infer units from magnitude.

## Coordinates and eyes

- Record coordinate space, origin, axis direction, units, pixel-center/edge convention, monitor, resolution, OS scaling, physical display size, viewing distance and reference frame. CLI width/height defaults 2560×1440 are engineering defaults, not device facts.
- Pixel-to-degree conversion requires measured geometry. For flat screen horizontal gaze, with pixel count `Npx`, physical screen width `Wcm`, viewing distance `Dcm`, and center `cpx`, angle is `atan(((xpx-cpx)*Wcm/Npx)/Dcm) * 180/pi`; calculate vertical separately using height. This assumes a perpendicular viewing position. Keep original px, geometry and formula. If geometry/head position is uncertain, retain px and mark degrees `missing`/`proposed`.
- Keep left/right eye gaze and pupil streams separate until validity, calibration, units and clocks align. Average only under a stated rule where both eyes qualify; retain both source values, validity and eye-source flags. Never average an invalid zero or mismatched eye. Iris-ratio features are proxies, not pupil diameter.

## Event alignment and reshaping

- Record trigger, device/behavior/video clock, onset definition, expected and observed latency, offset/drift model, residual error and uncertainty. Prefer shared pulses/markers. Keep original clocks plus aligned clock and transformation parameters.
- Join only on documented pseudonymous participant/session/trial/stimulus IDs. Find duplicate/unmatched keys; compare expected trial counts, event order, onset/offset and row counts before/after. Never align by row number when sample rates/clocks can vary.
- Wide-to-long/long-to-wide needs explicit schema, stable key, value field, condition field and duplicate policy. Test round-trip values/keys and preserve nesting/independent units. Do not aggregate away the unit of inference.

## Transforming outcomes

Select the method before seeing condition significance. Record operation/order, domain, unit, zero/negative handling, missing rule, fitted constants and inverse interpretation.

- `log(y)`: positive values only; justify/preregister any `c` in `log(y+c)`. Report transformed/geometric-scale estimand or defensible back-transform and interval.
- `sqrt(y)`: nonnegative count-like values; does not by itself fix overdispersion/dependence.
- `logit(p)`: proportion in [0,1]. If endpoints occur, choose an explicit denominator-appropriate correction (e.g. `(success+0.5)/(n+1)` only for justified binomial counts), preserve denominator, and back-transform. Not for arbitrary ratios.
- `z`: record reference mean/SD and population. Avoid leakage between training/validation. Say whether pooled, within-person or baseline referenced; report original units where useful.
- Baseline: specify baseline window/statistic, missing baseline rule, subtraction/division/log-ratio, zero denominator policy. Never pick baseline because it improves p.

After each stage check ranges, units, missingness, sample/row counts, ID uniqueness, monotonic time, and known conservation/identity invariants. On failure, stop, preserve inputs, log the reason and do not proceed as if conversion succeeded.
