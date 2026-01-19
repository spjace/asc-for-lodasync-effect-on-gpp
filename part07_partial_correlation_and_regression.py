import os
import numpy as np
import pandas as pd
import xarray as xr
import pingouin as pg
import statsmodels.api as sm
from tqdm import tqdm
from scipy.stats import zscore
from scipy.signal import detrend
from joblib import Parallel, delayed



def fit_single_pixel(data, var_list, x_name, y_name, covar_list, method='pcorr', is_detrend=True):
    """
    Estimate the X–Y association for one grid cell (one time-series sample).

    The input `data` is a (T, P) matrix for a single grid cell (T time steps, P variables).
    After dropping NaNs, variables are z-scored and optionally detrended. The association
    between `x_name` and `y_name` is then estimated either by:
      - partial correlation controlling covariates (`method='pcorr'`), or
      - OLS slope of X controlling covariates (`method='ols'`).

    Returns [NaN, NaN] if <10 valid observations remain or required columns are missing.

    Returns
    -------
    np.ndarray of shape (2,): [effect (r or slope), p_value]
    """
    # Build a DataFrame for convenient filtering/column selection
    df = pd.DataFrame(data=data, columns=var_list)
    df = df.dropna()
    if len(df) < 10:
        return np.array([np.nan, np.nan])
    
    # Standardize to comparable scale (and to make OLS slope interpretable as standardized beta)
    df = df.apply(zscore)
    df = df.dropna(axis=1, how='all')
    df = df.dropna()

    # Optionally remove linear trends to reduce co-trending artifacts
    if is_detrend:
        df = df.apply(detrend)
        df = df.dropna()

    # Required variables must survive cleaning
    new_columns = df.columns.tolist()
    if x_name not in new_columns:
        return np.array([np.nan, np.nan])
    elif y_name not in new_columns:
        return np.array([np.nan, np.nan])
    
    # Keep only covariates that are available after cleaning
    new_covar_list = [item for item in covar_list if item in df.columns]

    if len(df) < 10:
        return np.array([np.nan, np.nan])

    # Association estimation
    if method == 'ols':
        x = df[[x_name, *new_covar_list]].values.squeeze()
        y = df[y_name].values.squeeze()
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        return np.array([model.params[1], model.pvalues[1]])
    
    else:
        return pg.partial_corr(df, x=x_name, y=y_name, covar=new_covar_list)[['r', 'p-val']].values.squeeze()


def run_pixelwise_assoc(ds:xr.Dataset, y_name, x_name, covar_list, method='pcorr', n_jobs=-1, is_detrend=True):
    """
    Run grid-cell-wise association analysis between X and Y over a spatial domain.

    The function extracts (Y, X, covariates) from `ds`, applies a coverage screen using
    mean(sos_area) > 0.1, drops grid cells with excessive missingness, and then computes
    an effect size and p-value for each remaining grid cell in parallel.

    Returns
    -------
    xarray.Dataset with variables on (lat, lon): 'r' and 'p-val'.
    """
    var_list = [y_name, x_name, *covar_list] 
    ds_area = ds['sos_area'].mean(dim='time', skipna=True).squeeze()
    ds = ds[var_list].where(ds_area>0.1)
    data = ds.to_array().values.squeeze().transpose(2, 3, 1, 0) # (2160, 4320, 22, 5)
    data[np.isnan(data).any(axis=-1)] = np.nan # 五个变量中有一个变量有nan，则整个变量都nan
    
    # Keep grid cells with <10% missing entries across (time × variables)
    mask = np.isnan(data).sum(axis=(-2, -1)) < 0.1*data.shape[-2]*data.shape[-1]
    valid_data = data[mask] 

    # Parallel per-grid-cell estimation
    res_list = Parallel(n_jobs=n_jobs)(delayed(fit_single_pixel)(data_i, var_list, x_name, y_name, covar_list, method, is_detrend) for data_i in tqdm(valid_data))
    res_arr = np.array(res_list) 

    # Map results back to full grid
    result = np.full((mask.shape[0], mask.shape[1], res_arr.shape[-1]), np.nan)
    result[mask] = res_arr

    result_ds = xr.Dataset(data_vars={
        'r':(('lat', 'lon'), result[:, :, 0]),
        'p-val':(('lat', 'lon'), result[:, :, 1]),
    }, coords={'lat': ds['lat'], 'lon': ds['lon']}
    )
    
    return result_ds



if __name__ == '__main__':
    """
    Example: association between LOD asynchrony (X) and CSIF-based productivity (Y),
    controlling for growing-season covariates (gs_*).
    """

    y_name = 'gs_csif'
    x_name = 'lod_async'
    covar_list = sorted(['gs_t2m', 'gs_tp', 'gs_ssrd', 'lod_mean', 'gs_co2', 'gs_ndep'])

    ds = xr.open_dataset('data/data_0p10/data_part1.nc').sel(lat=slice(90, 30)).sel(time=slice('2001-01-01', '2022-12-31'))
    output_file = f'result/result_0p10/part2_pcorr/pcorr_{y_name}_{x_name}_{"_".join(covar_list)}.nc'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    overwrite = False
    if os.path.exists(output_file) and overwrite==False:
        result_ds = xr.open_dataset(output_file)
    else:
        result_ds = run_pixelwise_assoc(ds, y_name, x_name, covar_list)
        result_ds.to_netcdf(output_file)