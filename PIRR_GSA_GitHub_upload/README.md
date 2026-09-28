# PIRR-GSA reproducibility package

This package accompanies **Physics-Interface Residual Reduction for Time-Resolved Global Sensitivity Analysis of Temporal Responses**. It contains the executable minimum needed to audit the three controlled cases, regenerate their time-resolved Sobol figures, and verify the numerical values reported in the manuscript.

## Evidence map

| Manuscript evidence | Packaged result | Main role |
| --- | --- | --- |
| Case 1: Duffing oscillator | `results/case1_duffing/` | Residual-positive phase-drift mechanism and four-budget audit |
| Case 2: dual-domain transport | `results/case3_transport/` | Residual-positive shoulder/tail mechanism, four budgets, and interface leave-one-out audit |
| Case 3: controlled battery | `results/case2_battery/` | Smooth low-rank boundary where PCA-PCE is preferable |

The `case2_battery` and `case3_transport` directory and command names are stable internal identifiers from the experiment code. The manuscript case numbers follow the evidence order above.

`Direct-Time-PCE` is a non-reduced pointwise check. In Case 2 the formal ranking is restricted to methods fitting at most five reduced coordinates; the 192-output Direct-Time route is reported but not ranked with reduced-space methods.

## Environment

The formal runs used Python 3.11.1 with the versions listed in `requirements.txt`. A CPU-only PyTorch installation is sufficient.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux or macOS, use `.venv/bin/python` in place of `.venv/Scripts/python`.

## Fast audit

Run from this directory:

```bash
python run.py verify
python run.py plot --case all
```

`verify` checks the published means, rankings, paired Wilcoxon results, sample-budget trends, and Direct-MC convergence discrepancies. `plot` writes regenerated figures to `generated/figures/` without modifying the packaged results.

The principal values checked automatically are:

- Case 1: 0.1515 for Direct-Time, 0.1029 for lift-only, and 0.0669 for PIRR-PCA at `N=200`; lift-only versus PIRR is significant at all four budgets.
- Case 2: 0.0358 for unranked Direct-Time, 0.1369 for five-coordinate PCA, 0.1272 for lift-only, and 0.0841 for PIRR-PCA at `N=100`; PIRR wins the equal-time reduced-space comparison at all four budgets.
- Case 3: 0.0080 for PCA, 0.0081 for Direct-Time, 0.0153 for PIRR-PCA, and 0.0144 for PIRR-TCN; PCA minus PIRR-TCN is -0.0063.

## Executable checks

A small end-to-end run for every controlled case is available through:

```bash
python run.py smoke --case all
```

The smoke configuration reduces the sample size, Sobol base sample, seeds, and neural epochs. It tests executability and finite outputs; it is not intended to reproduce formal numerical values.

The focused unit tests can be run with:

```bash
python -m pytest tests -q
```

## Formal reruns

The commands below use the bundled converged Direct-MC references and write new outputs under `generated/`:

```bash
python run.py full --case case1_duffing
python run.py full --case case2_battery
python run.py full --case case3_transport
```

Add `--rebuild-reference` to reconstruct the high-cost reference before fitting the method routes:

```bash
python run.py full --case case3_transport --rebuild-reference
```

The Case 2 reference uses a base sample of 65,536 and is intentionally much more expensive than `verify` or `plot`. The reference evaluations are not part of the method training budgets.

## Fixed estimands and comparison rules

- The primary estimand is the complete time-wise first- and total-order Sobol field.
- `RMSE_func` gives equal weight to reported time points and inputs.
- Variance-weighted and active-time errors are robustness metrics.
- Descriptor-oriented sensitivity is secondary and never replaces complete-field recovery.
- Each physical interface is fixed before method comparison.
- PIRR-PCA and PIRR-TCN share the same interface and coordinate budget; only the residual backbone changes.
- Lift-only checks whether residual coordinates add information beyond the declared interface.

## Repository scope

This package contains only the three controlled manuscript cases. It excludes exploratory observational battery data and outputs, which do not provide a reference Sobol field. Once the repository is public, cite its actual URL and a fixed release or commit in the manuscript data-and-code availability statement; do not cite an uncreated record. GitHub does not itself provide a research-data DOI.

## Integrity

`MANIFEST.sha256` records the SHA256 digest of every packaged file except the manifest itself and the mutable `generated/` directory. Formal result JSON files are read-only evidence inputs; all commands write regenerated outputs to `generated/`.
