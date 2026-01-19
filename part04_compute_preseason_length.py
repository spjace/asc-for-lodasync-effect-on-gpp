import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from scipy.signal import detrend


def par_corr_npy(data, threshold=0.5):
    """
    Compute partial correlations between the first column (target) and all other columns.

    The input is a 2D array of shape (T, P), where column 0 is the target variable (y)
    and columns 1..P-1 are predictors (x). Rows containing any NaNs are removed. If the
    remaining sample size is below `threshold * T`, NaNs are returned. All columns are
    linearly detrended before computing partial correlations via the inverse correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (T, P): [y, x1, x2, ...].
    threshold : float, default 0.5
        Minimum fraction of valid (non-NaN) rows required to compute correlations.

    Returns
    -------
    np.ndarray
        Partial correlations r(y, xi | other x) for i=1..P-1, shape (P-1,).
    """
    rows, cols = data.shape
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[0] < rows*threshold:
        return np.full(cols-1, np.nan)
    
    data = detrend(data, axis=0)
    # data = zscore(data, axis=0)
    corr_matrix = np.corrcoef(data, rowvar=False)
    inv_matrix = np.linalg.inv(corr_matrix)
    r = np.array([-inv_matrix[0, col] / np.sqrt(inv_matrix[0, 0] * inv_matrix[col, col]) for col in range(1, cols)])
    return r


def partial_corr(x, y, threshold=0.5, n_jobs=-1):
    """
    Compute partial correlations per sample (e.g., per pixel/site) in parallel.

    This function stacks the target y and predictors x into a single array and applies
    `par_corr_npy` independently for each sample along the first dimension.

    Parameters
    ----------
    x : np.ndarray
        Predictor array of shape (N, T, K), where N is the number of samples, T is time,
        and K is the number of predictors.
    y : np.ndarray
        Target array of shape (N, T).
    threshold : float, default 0.5
        Minimum fraction of valid time steps required per sample.
    n_jobs : int, default -1
        Number of parallel workers for joblib.

    Returns
    -------
    np.ndarray
        Partial correlations per sample, shape (N, K).
    """
    data = np.concatenate([y[..., np.newaxis], x], axis=-1) # (660000, 21, 4)
    result = np.array(Parallel(n_jobs=n_jobs)(delayed(par_corr_npy)(data[i], threshold) for i in range(data.shape[0]))) # (660000, 3)
    return result



def get_opl(y:np.ndarray, x:np.ndarray, **kwargs):
    """
    Estimate the optimal pre-season length (OPL) for each predictor based on partial correlation strength.

    For each sample (pixel/site), this function tests pre-season windows of length 1..max_month
    ending at the mean phenological month. For each candidate window, it computes the absolute
    partial correlation between y and each predictor (controlling for other predictors), and
    selects the window length that maximizes |r|.

    Parameters
    ----------
    y : np.ndarray
        Annual target series, typically shape (rows, cols, years) (or compatible lower-rank forms).
    x : np.ndarray
        Monthly predictor series, shape (rows, cols, months, xvars), with months = 12 * (years + 1).
    kwargs :
        n_jobs : int, default -1
            Parallel workers for partial correlation computation.
        max_month : int, default 6
            Maximum pre-season length (months) to test.
        threshold : float, default 0.5
            Minimum fraction of valid values required in y and x.

    Returns
    -------
    np.ndarray
        OPL index (0..max_month-1) for each predictor, shape (rows, cols, xvars).
    """
    n_jobs = kwargs.get('n_jobs', -1)
    max_month = kwargs.get('max_month', 6)
    threshold = kwargs.get('threshold', 0.5)
    
    # Shape adaptation
    y = y.squeeze()
    x = x.squeeze()
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, :, :]  # (1, 1, months, xvars)
        y = y[np.newaxis, np.newaxis, :]  # (1, 1, years)

    elif x.ndim == 3:
        x = x[np.newaxis, :, :, :]  # (1, cols, months, xvars)
        y = y[np.newaxis, :, :]  # (1, cols, years)
        
    
    rows, cols, months, xvars = x.shape
    _, _, years = y.shape

    mask_y = np.sum(~np.isnan(y), axis=2) >= years*threshold 
    mask_x = np.sum(~np.isnan(x), axis=2) >= months*threshold
    mask_x = np.sum(mask_x, axis=2) == xvars
    mask = mask_y & mask_x
    y = y[mask]
    x = x[mask]

    # Get the month of the multi-year average phenological period
    y_mean = np.nanmean(y, axis=-1)
    y_mean = np.array([(datetime(2023, 1, 1) + timedelta(days=int(round(day)))).month for day in y_mean])
    y_mean = np.expand_dims(y_mean, axis=-1).astype(int)

    indices1 = np.arange(years) * 12 + y_mean - 1
    indices0 = np.arange(x.shape[0])[:, np.newaxis]

    x_list = [x[:, 12-i:-i if i!= 0 else None, :] for i in range(max_month)]
    x_list = [np.mean(x_list[:i+1], axis=0)[indices0, indices1, :] for i in range(max_month)]

    result = [partial_corr(x, y, threshold, n_jobs) for x in tqdm(x_list)]
    result = np.abs(np.array(result))
    result = np.argmax(result, axis=0)
    
    result_mask = np.full((rows, cols, xvars), np.nan)
    result_mask[mask] = result
    result_mask = result_mask.squeeze()
    
    return result_mask

    
def single_mean(x, opl, pheno):
    """
    Compute year-wise pre-season means for one 1D monthly series.

    Given a monthly time series x, this function extracts, for each year, the mean
    over a pre-season window of length `opl` months ending at month `pheno`.

    Parameters
    ----------
    x : np.ndarray
        Monthly series of shape (months,).
    opl : int or float
        Pre-season length in months (window size).
    pheno : int or float
        Phenology month (1..12) used as the window endpoint.

    Returns
    -------
    np.ndarray
        Year-wise pre-season means, shape (months//12 - 1,).
    """
    k_indices = np.arange(12, len(x), 12) # (months//12-1, )
    start_indices = (k_indices + pheno - opl - 1).astype(int) # (months//12-1, )
    end_indices = (k_indices + pheno).astype(int) # (months//12-1, )
    mask = (np.arange(len(x)) >= start_indices[:, np.newaxis]) & (np.arange(len(x)) < end_indices[:, np.newaxis])
    x_masked = np.where(mask, x, np.nan) 
    resarr = np.nanmean(x_masked, axis=1) # shape (months//12-1, )
    
    return resarr


def seasonal_mean(x_arr: np.ndarray, opl_arr: np.ndarray, pheno_arr: np.ndarray):
    """
    Compute pre-season means for a gridded monthly variable using per-pixel OPL.

    This function applies `single_mean` to each valid pixel to convert a monthly climate
    cube (months, rows, cols) into a year-wise pre-season mean cube (years, rows, cols),
    using pixel-specific pre-season length (opl_arr) and phenology timing (pheno_arr).

    Parameters
    ----------
    x_arr : np.ndarray
        Monthly climate array of shape (months, rows, cols), where months is a multiple of 12.
    opl_arr : np.ndarray
        Optimal pre-season length per pixel, shape (rows, cols).
    pheno_arr : np.ndarray
        Phenology timing per pixel, shape (rows, cols), in DOY (1..365) or month (1..12).

    Returns
    -------
    np.ndarray
        Pre-season means, shape (months//12 - 1, rows, cols).
    """

    if np.nanmax(pheno_arr) > 12: # if the value of pheno_arr is the "day of year" (1-365), then convert it to the month
        pheno_arr[~np.isnan(pheno_arr)] = np.array([(datetime(2023, 1, 1) + timedelta(days=eos - 1)).month for eos in pheno_arr[~np.isnan(pheno_arr)]])

    months, rows, cols = x_arr.shape
 
    nn_indices = np.where(~np.isnan(opl_arr) & ~np.isnan(pheno_arr)) # nn_indices, means the indices of non-NaN elements
    nn_opl = opl_arr[nn_indices] 
    nn_pheno = pheno_arr[nn_indices] 
    nn_x = x_arr[:, nn_indices[0], nn_indices[1]]

    # slice the data
    args_list = [(nn_x[:, i], nn_opl[i], nn_pheno[i]) for i in range(nn_opl.shape[0])]
    res = np.array(Parallel(n_jobs=-1)(delayed(single_mean)(x, opl, pheno) for x, opl, pheno in tqdm(args_list)))
    result = np.full((months//12-1, rows, cols), np.nan)
    result[:, nn_indices[0], nn_indices[1]] = res.swapaxes(0, 1)

    return result




if __name__ == "__main__":
    """
    Example: estimating optimal pre-season length (OPL) for multiple climate drivers.

    This script demonstrates how to:
      1) Load monthly climate variables (e.g., temperature, precipitation, radiation) from NetCDF.
      2) Load an annual phenology target series (e.g., yearly mean LOD).
      3) Reshape data into the numpy layouts expected by `get_opl`:
           - x_arr: (rows, cols, months, xvars)
           - y_arr: (rows, cols, years)
      4) Estimate per-pixel OPL for each climate variable and export results to NetCDF.

    Notes
    -----
    - The climate series is monthly and is clipped to 2000-01-01 ... 2023-12-31.
    - The phenology target is annual (one value per year) clipped to 2001 ... 2023.
    - `get_opl` assumes the monthly predictor cube covers one more year than the
      annual target (i.e., months = 12 * (years + 1)), so that pre-season windows
      can be defined relative to each phenological year.
    - `get_opl` is provided as a published public API in the `phenology` Python package.
    """
    from phenology.opl import get_opl
    import xarray as xr

    # Monthly climate drivers (ERA5): temperature (t2m), precipitation (tp), radiation (ssrd)
    files = [
        "data/data_0p10/data_common/era5/t2m.2000_2024.0p10.nc",
        "data/data_0p10/data_common/era5/tp.2000_2024.0p10.nc",
        "data/data_0p10/data_common/era5/ssrd.2000_2024.0p10.nc",
    ]

    # Load monthly climate data and restrict to the analysis window
    x_list = [xr.open_dataset(f).sel(time=slice('2000-01-01', '2023-12-31')) for f in files]
    x = xr.merge(x_list)

    # Load annual phenology target (e.g., yearly mean leaf-out date) and restrict years
    y = xr.open_dataset('data/data_0p10/data_common/phenology/pheno.0p10.nc')[['lod_mean']].sel(time=slice('2001-01-01', '2023-12-31'))

    # Convert to numpy arrays with the shapes required by get_opl:
    #   x_arr: (rows, cols, months, xvars)
    #   y_arr: (rows, cols, years)
    x_arr = x.to_array().values.transpose(2, 3, 1, 0) # (600, 3600, 288, 3) 
    y_arr = y.to_array().values.transpose(2, 3, 1, 0).squeeze() # (600, 3600, 23)

    # Estimate per-pixel OPL (index 0..max_month-1) for each climate variable
    opl = get_opl(y_arr, x_arr) # (600, 3600, 3)

    # Package OPL outputs into an xarray.Dataset and export
    vars = ['opl_t2m', 'opl_tp', 'opl_ssrd']
    ds_opl = xr.Dataset(data_vars={var: (('lat', 'lon'), opl[:, :, i]) for i, var in enumerate(vars)}, coords={'lat': x.lat, 'lon': x.lon}).astype('float32')
    ds_opl.to_netcdf(f'data/data_0p10/data_common/phenology/opl.lod_mean.t2m.tp.ssrd.0p10.nc')

