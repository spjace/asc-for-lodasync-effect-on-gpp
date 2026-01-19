import numpy as np
from sklearn.linear_model import LinearRegression

# Small caches to speed up large-scale grid-cell computation:
# - _IMPORT_CACHE: cache variable-name -> column-index mapping to avoid rebuilding dictionaries ~O(10^5-10^6) times.
# - _PERM_CACHE: cache permutation indices for a given number of valid observations (n_obs) to avoid regenerating them repeatedly.
# Notes:
#   key=(id(var_list), id(x_names), y_name_norm) is used for indexing cache entries.
_IMPORT_CACHE = {}   # key=(id(var_list), id(x_names), y_name_norm) -> (y_idx, x_idx)
_PERM_CACHE = {}     # key=(n_obs, n_perm) -> list(permutation arrays)


def _norm_name(s):
    """
    Normalize variable names to a plain Python string.

    This helper improves robustness when variable names come in different types
    (e.g., bytes, numpy.str_), or when bytes are stringified into forms like "b'xxx'".

    Parameters
    ----------
    s : Any
        A variable name candidate.

    Returns
    -------
    str
        Normalized variable name as a standard Python string.
    """
    if isinstance(s, bytes):
        s = s.decode()
    s = str(s)
    if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
        s = s[2:-1]
    return s


def _get_indices(var_list, y_name, x_names):
    """
    Map variable names to column indices in `data_i` (with caching).

    Given a column-name list `var_list` that matches the column order of `data_i`,
    this function returns:
      - y_idx: the column index of the target variable y
      - x_idx: a list of column indices for predictor variables X

    A small cache is used to avoid rebuilding the name->index dictionary for every grid cell.

    Parameters
    ----------
    var_list : list
        Variable names aligned with the second dimension of `data_i`.
    y_name : str
        Name of the target variable (y).
    x_names : list
        Names of predictor variables (X), in the desired output order.

    Returns
    -------
    tuple
        (y_idx, x_idx) where y_idx is an int and x_idx is a list[int].
    """
    # Cache key: based on the identity of var_list and x_names objects + normalized y_name
    key = (id(var_list), id(x_names), _norm_name(y_name))
    if key in _IMPORT_CACHE:
        return _IMPORT_CACHE[key]

    # Build lookup: normalized name -> column index
    name2idx = {_norm_name(v): i for i, v in enumerate(var_list)}
    y_idx = name2idx[_norm_name(y_name)]
    x_idx = [name2idx[_norm_name(x)] for x in x_names]

    _IMPORT_CACHE[key] = (y_idx, x_idx)
    return y_idx, x_idx


def _get_perms(n_obs, n_perm=3, seed=12345):
    """
    Pre-generate permutation indices for a fixed number of observations (with caching).

    All grid cells with the same `n_obs` share the same set of permutations, which:
      - speeds up computation
      - ensures reproducibility across grid cells

    Parameters
    ----------
    n_obs : int
        Number of valid observations (rows) after masking NaNs/Infs.
    n_perm : int, default 3
        Number of permutation replicates.
    seed : int, default 12345
        Base seed. The effective seed becomes (seed + n_obs).

    Returns
    -------
    list[np.ndarray]
        A list of length `n_perm`, each an integer permutation array of shape (n_obs,).
    """
    key = (n_obs, n_perm)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]

    # Seed depends on n_obs, so different sample sizes get different permutations
    rng = np.random.default_rng(seed + n_obs)
    perms = [rng.permutation(n_obs) for _ in range(n_perm)]
    _PERM_CACHE[key] = perms
    return perms





def get_importance_single(data_i, var_list, y_name, x_names, n_perm=1, eps=1e-12):
    """
    Compute grid-cell permutation importance (ΔMSE) and return normalized relative contributions (sum to 1).

    This function fits an OLS-style linear regression (via sklearn LinearRegression with intercept)
    for a single grid cell / site using time series data:

        y ~ X

    Importance is quantified by permutation importance:
      - For each predictor X_j, permute its values across time (keeping other predictors unchanged),
        and compute the increase in mean squared error (ΔMSE).
      - Only positive ΔMSE values are accumulated (i.e., only when permuting worsens model performance).
      - The final importance scores are normalized so that they sum to 1 across predictors.

    Parameters
    ----------
    data_i : np.ndarray
        Array of shape (T, n_vars) for one grid cell (T time steps).
    var_list : list
        Variable names aligned with columns of `data_i`.
    y_name : str
        Target variable name.
    x_names : list
        Predictor variable names (output order follows this list).
    n_perm : int, default 1
        Number of permutations per predictor.
    eps : float, default 1e-12
        Small constant for numerical stability (e.g., avoid division by zero).

    Returns
    -------
    np.ndarray
        Float32 array of shape (p,) where p=len(x_names).
        Values represent normalized relative contributions (sum ≈ 1).
        Returns all-NaN if insufficient valid observations or if fitting/normalization fails.
    """
    # Map variable names to column indices in `data_i` (cached)
    y_idx, x_idx = _get_indices(var_list, y_name, x_names)

    # Extract y and X time series for this grid cell
    y = data_i[:, y_idx].astype(np.float64)      # (T,)
    X = data_i[:, x_idx].astype(np.float64)      # (T, p)
    p = X.shape[1]

    # Keep only rows where y and all predictors are finite
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    n_obs = int(mask.sum())
    if n_obs < p + 2:
        return np.full(p, np.nan, dtype=np.float32)

    yv = y[mask]
    Xv = X[mask, :]

    # ===== OLS fit (LinearRegression with intercept) =====
    try:
        ols = LinearRegression(fit_intercept=True)
        ols.fit(Xv, yv)
    except Exception:
        return np.full(p, np.nan, dtype=np.float32)

    intercept = float(ols.intercept_)
    beta = np.asarray(ols.coef_, dtype=np.float64)

    # Baseline prediction and MSE
    pred = intercept + Xv @ beta
    resid = yv - pred
    mse0 = float(np.mean(resid * resid))
    if (not np.isfinite(mse0)) or mse0 <= eps:
        return np.full(p, np.nan, dtype=np.float32)

    # Permutation importance: ΔMSE per predictor (using cached permutations by n_obs)
    perms = _get_perms(n_obs, n_perm=n_perm)
    delta = np.zeros(p, dtype=np.float64)

    for j in range(p):
        bj = float(beta[j])
        # If coefficient is ~0 or invalid, permuting this predictor has negligible effect
        if (not np.isfinite(bj)) or abs(bj) <= eps:
            continue

        xj = Xv[:, j]

        # Efficient update: avoid recomputing Xv @ beta for each permutation
        # pred_perm = pred + (xj_perm - xj) * beta_j
        for perm in perms:
            xj_perm = xj[perm]
            predp = pred + (xj_perm - xj) * bj
            rp = yv - predp
            msep = float(np.mean(rp * rp))
            d = msep - mse0
            if d > 0:
                delta[j] += d

    # Average ΔMSE over permutations
    delta /= float(n_perm)

    # Normalize to relative contributions (sum to 1)
    s = float(delta.sum())
    if (not np.isfinite(s)) or s <= eps:
        return np.full(p, np.nan, dtype=np.float32)

    return (delta / (s + eps)).astype(np.float32)




if __name__ == '__main__':


    '''
    Minimal usage example (commented out).

    This block shows how to compute permutation-based OLS importance (relative contributions; sum to 1)
    for multiple grid cells / sites, and map the vectorized outputs back to a 2D lat-lon grid before
    saving as NetCDF.

    Input NPZ is assumed to be preprocessed:
      - each time series contains at least 80% valid values
      - all variables are z-score standardized
    '''

    '''

    import os
    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from phenology.utils import split_array


    # -------------------------------
    # Load preprocessed input
    # -------------------------------
    # The NPZ file has been preprocessed such that:
    # - each time series contains at least 80% valid values
    # - all variables have been z-score standardized
    npz = np.load('data/sample.npz')

    # data: (N, T, V)
    # N = number of spatial locations (points)
    # T = number of time steps (e.g., years)
    # V = number of variables aligned with `var_list`
    data = npz['data']  # (584863, 23, 8) float32

    # Variable names aligned with the last dimension of `data`
    var_list = [str(item) for item in npz['var_list']]  # ['gs_csif' ... 'gs_nhx']

    # Predictors (X) and target (y)
    x_names = [str(item) for item in npz['x_names']]  # ['gs_t2m', ..., 'gs_nhx']
    y_name = str(npz['y_name'])  # 'gs_csif'

    # Spatial coordinates and mask used to map results back to a 2D grid
    lat_arr = npz['lat']
    lon_arr = npz['lon']
    mask = npz['mask']

    # split the N locations into chunks (each chunk processed as one job).
    data_list = split_array(data, chunk_size=200) # 每组分200个

    # Permutation importance per point
    def get_importance(data, var_list, y_name, x_names):
        return np.array([get_importance_single(data_i, var_list, y_name, x_names) for data_i in data])

    # Run chunk-level jobs in parallel
    result = Parallel(n_jobs=200)(delayed(get_importance)(data, var_list, y_name, x_names) for data in tqdm(data_list))
    result = np.concatenate(result, axis=0).astype(np.float32)

     # Map back to 2D grid and save
    rows, cols = mask.shape
    result_mask = np.full((rows, cols, result.shape[-1]), np.nan)
    result_mask[mask] = result

    ds = xr.Dataset(
        data_vars={x: (("lat", "lon"), result_mask[..., idx]) for idx, x in enumerate(x_names)},
        coords={"lat": lat_arr, "lon": lon_arr}
    )

    output_file = "result_offset/perm_ols_importance_fig.2d.full.nc"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds.to_netcdf(output_file)

    '''
    pass