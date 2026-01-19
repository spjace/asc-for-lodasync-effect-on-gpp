## Overview
This repository contains the analysis code for the manuscript:

**“Increasing leaf-out asynchrony is linked to declining ecosystem productivity.”**

The workflow builds leaf-out asynchrony metrics from remote sensing and ground observations, applies quality/purity constraints and seasonal aggregation, and evaluates trends, associations, drivers, and mechanisms using multiple complementary statistical and causal frameworks.

---

## Python environment
A single conda environment specification (environment.yml) is provided for reproducibility. 
---

## Scripts (part01–part12)

### part01_compute_leafout_asynchrony_modis.py
**Purpose:** Aggregate yearly LOD GeoTIFFs into multiple lat/lon grid resolutions (0.10–1.00°) and compute grid-cell statistics.

**Key outputs (NetCDF):** `lod_mean`, `lod_std` (used as grid-scale LOD<sub>async</sub>), `lod_area` (valid-pixel fraction), written per year and per resolution.

---

### part02_compute_leafout_asynchrony_pep725.py
**Purpose:** Build climate-matched station neighborhoods (z-scored climate space) and compute yearly LOD dispersion within each neighborhood.

**Key outputs (CSV):** per-site per-year summary including `lod_mean`, `lod_std`, `species`, `counts`.

---

### part03_compute_spatiotemporal_purity.py
**Purpose:** Aggregate tiled 30 m categorical maps (e.g., dominant species/type) to a ~500 m grid within each 2°×2° tile and export mode diagnostics.

**Key outputs (NetCDF):** `mode_val` (dominant class), `mode_freq` (dominant count), `count` (valid count).

---

### part04_compute_preseason_length.py
**Purpose:** Estimate the **optimal pre-season length (OPL)** per grid cell and per monthly climate driver by maximizing the absolute partial correlation between the annual target and pre-season means.

**Key outputs (NetCDF in example):** OPL indices for each driver (e.g., `opl_t2m`, `opl_tp`, `opl_ssrd`).

---

### part05_extract_growing_season_means.py
**Purpose:** Compute growing-season means for variables (e.g., GPP, climate) using phenology-based windows, including cross-year growing seasons when needed.

**Key outputs (NetCDF):** annual growing-season means (time = year).

---

### part06_mk_trend_analysis.py
**Purpose:** Pixel-wise Mann–Kendall trend analysis (with Sen’s slope) across multiple spatial resolutions.

**Key outputs (NetCDF):** `p`, `z`, `tau`, `s`, `var_s`, `slope`, `intercept` for the target variable (e.g., `lod_async`).

---

### part07_partial_correlation_and_regression.py
**Purpose:** Grid-cell association analysis between X and Y using either partial correlation (controlling covariates) or OLS.

**Key outputs (NetCDF):** `r` and `p-val` on `(lat, lon)` (for OLS, `r` stores the standardized slope/beta).

---

### part08_ols_and_permutation_importance.py
**Purpose:** OLS-based driver attribution using permutation importance.

**Key outputs (NetCDF in example):** spatial maps of normalized permutation importance for each predictor.

---

### part09_xgboost_shap_pdp.py
**Purpose:** Train XGBoost models, tune hyperparameters, compute SHAP values, and support PDP-style summaries.

**Key outputs:** a cached joblib artifact containing model/metrics/SHAP-related objects used for downstream plotting and interpretation.

---

### part10_dowhy_causal_inference.py
**Purpose:** Estimate grid-cell causal effects using DoWhy under user-specified causal graphs (DAGs) and a backdoor linear regression estimator.

**Key outputs (NetCDF):** `r` (estimated causal effect, not a correlation) and `p-val`, written for each causal-graph configuration.

---

### part11_convergent_cross_mapping.py
**Purpose:** Run **Convergent Cross Mapping (CCM)** (pyEDM) on many samples to provide complementary evidence on directional coupling in nonlinear dynamics.

**Key outputs:** chunked pickle files (`.pkl`) containing CCM results and selected embedding parameters.

---

### part12_structural_equation_modeling.py
**Purpose:** Fit per-sample **Structural Equation Models (SEM)** (semopy), returning standardized path coefficients and fit statistics for mechanism evaluation.

**Key outputs:** a pickle file (`.pkl`) containing SEM results, fit statistics, and metadata for mapping back to spatial coordinates.

---

## Citation and license
We will add the DOI and the recommended citation format once the paper is published. A software license (e.g., MIT or BSD-3-Clause) will be included based on journal and team requirements.

---

## Contact
For questions, please contact Pengju Shen (shenpj@igsnrr.ac.cn).
Questions and collaboration are welcome.
