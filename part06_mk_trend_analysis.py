import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import pymannkendall as mk
from IPython.display import display
from joblib import Parallel, delayed



def mk_test(x):
    """
    Perform Mann-Kendall trend test for a 1D time series.

    Parameters
    ----------
    x : array-like
        1D time series (e.g., yearly values for one grid cell).

    Returns
    -------
    tuple
        (p, z, tau, s, var_s, slope, intercept)
        - p: p-value of the Mann-Kendall test
        - z: standardized test statistic
        - tau: Kendall's Tau
        - s: Mann-Kendall S statistic
        - var_s: variance of S
        - slope: Sen's slope estimator
        - intercept: intercept of the fitted trend line

    Notes
    -----
    - Uses `pymannkendall.original_test`.
    - Any exception (e.g., all-NaN series) will return 7 NaNs to keep shapes consistent.
    """
    try:
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(x)
        return p, z, Tau, s, var_s, slope, intercept
    except:
        return [np.nan]*7


def mk_main(ds, x_name):
    """
    Pixel-wise Mann-Kendall analysis for ds[x_name] within 90–30N and 2001–2023.

    Notes
    -----
    - Applies a coverage mask using mean(sos_area) > 0.1 (exclude low-coverage grid cells).
    - Assumes ds[x_name] is ordered as (time, lat, lon) when converting to numpy.
    
    Returns
    -------
    xarray.Dataset with variables on (lat, lon):
        p, z, tau, s, var_s, slope, intercept
    """
    ds = ds.sel(lat=slice(90, 30), time=slice('2001-01-01', '2023-01-01')) # subset region and time
    mask = ds['sos_area'].mean(dim='time').values > 0.1  # coverage mask (exclude low-coverage grid cells)
    rows, cols = mask.shape
    data = ds[x_name].values.transpose(1,2,0)[mask]
    res_mk = Parallel(n_jobs=-1)(delayed(mk_test)(x) for x in tqdm(data))
    res_arr = np.array(res_mk)

    # map back to full grids, NaN outside mask
    res_mask = np.full((rows, cols, res_arr.shape[-1]), np.nan)
    res_mask[mask] = res_arr
    ds_mk = xr.Dataset({var: (['lat', 'lon'], res_mask[:,:,i]) for i, var in enumerate(['p', 'z', 'tau', 's', 'var_s', 'slope', 'intercept'])}, 
                    coords={'lat': ds.lat, 'lon': ds.lon})
    return ds_mk



if __name__ == "__main__":
    """
    Example: multi-resolution Mann–Kendall trend analysis for leaf-out asynchrony.

    This script loops over multiple spatial resolutions (0.10°–1.00°), loads the
    corresponding NetCDF file, computes pixel-wise Mann–Kendall trends for `x_name`
    over 90–30°N during 2001–2023, and saves results to disk. A brief summary of
    trend direction (tau) and significance (p) is printed for each resolution.
    """
    overwrite = False
    x_name = 'lod_async' # leaf-out asynchrony
    for resolution in np.arange(0.10, 1.01, 0.05).round(4):
        key = f'{resolution:.2f}'.replace('.','p')  # key like "0p10" to match folder naming convention
        file = f'data/data_{key}/data_part1.nc'     # Input NetCDF file for this resolution
        dst_dir = f'result/result_{key}/part1_mk'
        os.makedirs(dst_dir, exist_ok=True)
        output_file = os.path.join(dst_dir, f'mk.{x_name}.2001_2023.{key}.nc')

        # run or load cached results
        if not os.path.exists(output_file) or overwrite:
            ds = xr.open_dataset(file)
            ds_mk = mk_main(ds, x_name)
            ds_mk.to_netcdf(output_file)
            print(output_file, 'mk analysis done!')
        else:
            print(output_file, 'already exists!')
            ds_mk = xr.open_dataset(output_file)