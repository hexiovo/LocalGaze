# Statistical plan and study-specific execution

The current `EyeEvaluate` pipeline supplies descriptive metrics, event candidates, ROI outputs, and CSV/TXT exports. It has no general inferential testing engine. A statistical plan file is not an execution engine. Generate an independent study-specific script using a real installed library/API, or mark execution blocked until design/data/environment are resolved. Keep all statistics out of native desktop JSON.

## Required design specification

Define: (1) scientific question, primary outcome/window, contrast(s), direction, estimand and units; (2) experimental, observation, assignment and analysis units; (3) design: between, paired/within, repeated, crossover, clustered, mixed/crossed, nested or factorial; randomization/counterbalance/order/carryover; (4) factors, outcomes, reference levels, planned contrasts, pre-treatment covariates and rationale; (5) population, exclusions, missing reasons, valid-event/sample rules, aggregation level, and whether exclusions are fixed before outcomes are inspected; (6) test/model family, likelihood/link/variance structure, assumptions/diagnostics, effect and CI, multiplicity family/correction and sensitivity analyses.

If independence/pairing/unit, outcome meaning, primary contrast or measurement meaning is missing, ask the smallest blocker question(s). Draft other items as proposed/missing, but do not provide launch steps or claim executable. Recommend the method from the design; do not ask novice operators to choose a test unaided.

## Match method to design

Choose the simplest method compatible with assignment, outcome, and estimand. Do not treat gaze samples as independent when conditions are assigned by person or trial; reduce to the inference unit or model within-unit dependence.

- Two independent groups, continuous person-level outcome: Welch two-sample test may fit a mean difference; otherwise justify robust/permutation/model alternative.
- Two conditions on the same people/units: paired difference analysis or paired permutation; verify one-to-one pairing and report unmatched IDs.
- More repeated conditions, repeated trials, subject/item effects: repeated-measures or mixed model with justified grouping/slopes/contrasts. Do not row-wise t-test nested trials.
- Binary/count/proportion/positive-skew duration or rate: choose a likelihood/link matching numerator/denominator and exposure. For rates, include exposure offset where appropriate and model repeated IDs. Gaussian event-row test is not a default.
- Permutation must preserve independent/paired/cluster assignment via the correct exchangeability blocks. Never freely permute sample rows when assignment occurs at participant level.

Examples guide judgment; they are not an auto-method menu. When two methods are plausible, state a recommended primary and one design-relevant sensitivity analysis. `statsmodels.MixedLM` has specific supported structures/constraints and does not fit every binary/count outcome.

## Generated artifacts and verification

Produce an independent project-defined `analysis_plan.json` with outcome, estimand, unit, design, predictors/contrasts/coding, exclusions/missing rules, model/test, diagnostics, CI, multiplicity family, sensitivity/transformation, versions, files and status. It is not a LocalGaze schema.

Produce a study-specific `run_statistics.py` (or declared-language script) that reads a documented table, checks schema/IDs once at entry, fits only the declared model, and writes tidy estimates, CI, exact inferential p where applicable, counts/units, diagnostics, warnings/errors, and provenance to a new output. No silent fallback. On failure stop and save a precise failed/NOT_RUN record. Include environment lock/version list and synthetic design-shaped fixture with known code-path expectations. Synthetic output proves only that example path, never a real scientific result.

Call it executable only after running the script against a synthetic fixture and real pilot/blinded input in the target environment, with logs and output reviewed. Distinguish those gates. Actual study results require the approved design, collected data, hardware/context and runtime.

## Reporting

Report estimate, units/scale, 95% CI (or predeclared alternative), exact p where inferential, analysis-unit counts, exclusions/missingness and reasons, model/test, multiplicity and warnings. Diagnose residual/fit/influence and variance assumptions; mixed models need convergence, singularity and random-effect checks. Do not try tests until significant; use a prespecified fallback or stop. Define multiplicity families scientifically and separate confirmatory/exploratory outcomes. Transform before inspecting inferential results; report transformed estimand and interpretable back-transform when valid. Sensitivity analysis varies defensible assumptions, not p-value outcomes.

Official API references (verify against the installed version): [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [SciPy independent t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html), [SciPy permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) (permutation structure matters), [statsmodels MixedLM](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.from_formula.html) (groups, random effects and variance components).
