import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta


class GSCalculator:
    """
    Growing Season Calculator

    This class calculates the growing season mean of a variable (e.g., GPP)
    based on phenology data (start of season (sos) and end of season (eos)). The 
    computation is split into two parts:
    
      1. Same-Year Regions: where phenology data has sos ≥ 1 and eos ≤ 365 (v1 method)
      2. Cross-Year Regions: where sos < 1 or eos > 365 (v2 method)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable data.
    pheno : xarray.Dataset
        Dataset containing phenology data (e.g., sos, eos).
    sos_name : str, optional
        Variable name for the start of season (default is "sos").
    eos_name : str, optional
        Variable name for the end of season (default is "eos").
    years : list, optional
        List of years for which to perform the calculation. If not provided,
        the years are extracted from ds.time.
    need_extend : bool, optional
        Flag indicating whether to extend the time dimension for cross-year data.
    """

    def __init__(self, ds: xr.Dataset, pheno: xr.Dataset, sos_name='sos', eos_name='eos', need_extend=False):
        self.ds = ds
        self.pheno = pheno
        self.sos_name = sos_name
        self.eos_name = eos_name
        self.need_extend = need_extend


    def extend_time(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Extend the time dimension by appending data from the first and last year,
        shifted by -1 and +1 year respectively. This helps cover cross-year 
        growing season windows. Note: Uses ±365 days without leap year adjustment.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset with a 'time' coordinate.
        
        Returns
        -------
        xarray.Dataset
            Dataset with an extended time dimension.
        """
        first_year = int(ds['time'].dt.year.min().values)
        last_year = int(ds['time'].dt.year.max().values)
    
        ds_first = ds.sel(time=slice(f'{first_year}-01-01', f'{first_year}-12-31')).copy(deep=True)
        ds_last  = ds.sel(time=slice(f'{last_year}-01-01', f'{last_year}-12-31')).copy(deep=True)
    
        ds_first = ds_first.assign_coords(time=ds_first['time'] - pd.Timedelta(days=365))
        ds_last  = ds_last.assign_coords(time=ds_last['time'] + pd.Timedelta(days=365))
    
        ext_ds = xr.concat([ds_first, ds, ds_last], dim='time').sortby('time')
        return ext_ds


    @staticmethod
    def _gs_v2_single(ds_ts, sos, eos, t_coord, years):
        """
        Compute the growing season mean for a single grid cell in cross-year regions using the v2 method.
        
        Parameters
        ----------
        ds_ts : DataArray
            Time series of the variable for a single grid point.
        sos : scalar
            Start-of-season day.
        eos : scalar
            End-of-season day.
        t_coord : numpy.ndarray
            Global time coordinate (array of datetime64 values).
        years : list
            List of years for which to compute the mean.
        
        Returns
        -------
        numpy.ndarray
            An array of growing season mean values, one for each year.
        """
        if np.isnan(sos) or np.isnan(eos):
            return np.full((len(years),), np.nan)
    
        means = []
        for year in years:
            start = np.datetime64(datetime(year, 1, 1) + timedelta(days=sos - 1))
            end   = np.datetime64(datetime(year, 1, 1) + timedelta(days=eos - 1))
            mask = (t_coord >= start) & (t_coord <= end)
            means.append(np.nanmean(ds_ts[mask]) if np.any(mask) else np.nan)
        return np.array(means)


    def gs_v2(self, ds: xr.Dataset, sos, eos) -> xr.Dataset:
        """
        Compute the growing season mean for cross-year regions using the vectorized v2 method.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Dataset containing the variable data for cross-year regions.
        sos, eos : DataArray or scalar
            Start and end of season.
        
        Returns
        -------
        xarray.DataArray
            Growing season mean with a new 'time' coordinate representing each year.
        """
        years = sorted(np.unique(ds['time'].dt.year.values))
        if self.need_extend:
            ds = self.extend_time(ds)
    
        res = xr.apply_ufunc(
            GSCalculator._gs_v2_single,
            ds, sos, eos,
            input_core_dims=[['time'], [], []],
            kwargs={'t_coord': ds['time'].values, 'years': years},
            vectorize=True,
            dask='parallelized',
            output_core_dims=[['year']],
            output_dtypes=[float],
        )
    
        res = res.assign_coords(year=years).rename({'year': 'time'})
        res['time'] = pd.to_datetime(res['time'], format='%Y')
        return res


    @staticmethod
    def _gs_v1_single(ds_year, sos, eos):
        """
        Compute the growing season mean for a single year's data (same-year regions, v1 method).
        
        Parameters
        ----------
        ds_year : xarray.Dataset
            Dataset for one year, which must include a 'doy' coordinate.
        sos, eos : scalar
            Thresholds for start and end of season.
        
        Returns
        -------
        xarray.Dataset
            The computed growing season mean for the year.
        """
        mask = (ds_year['doy'] > sos) & (ds_year['doy'] < eos)
        return ds_year.where(mask).mean(dim='time', skipna=True)
    

    def gs_v1(self, sos, eos) -> xr.Dataset:
        """
        Compute growing season means for same-year regions.

        This method groups the dataset by year (using the day-of-year coordinate)
        and calculates the mean within the growing season window defined by `sos` and `eos`.
        It is applicable when the growing season occurs within a single calendar year.

        Parameters
        ----------
        sos, eos : scalar or xarray.DataArray
            Thresholds for start and end of season.

        Returns
        -------
        xarray.Dataset
            Dataset with growing season means, where the time coordinate corresponds to each year.
        """
        ds_group = self.ds.assign_coords(doy=self.ds['time'].dt.dayofyear).groupby('time.year')
        res = ds_group.map(GSCalculator._gs_v1_single, sos=sos, eos=eos).rename({'year': 'time'})
        res['time'] = pd.to_datetime(res['time'], format='%Y')
        return res


    def calc(self) -> xr.Dataset:
        """
        Main entry point.

        Splits the dataset into same-year and cross-year regions based on the phenology data,
        computes the growing season mean using the v1 (same-year) and v2 (cross-year) methods,
        and then combines the results.

        Returns
        -------
        xarray.Dataset
            The combined growing season mean dataset.
        """
        
        same_mask  = (self.pheno[self.sos_name] >= 1) & (self.pheno[self.eos_name] <= 365) # Same-year  region: sos >= 1 and eos <= 365.
        cross_mask = (self.pheno[self.sos_name] <  1) | (self.pheno[self.eos_name] >  365) # Cross-year region: sos <  1 or  eos >  365.
    
        res_same = self.gs_v1(
            self.pheno[self.sos_name].where(same_mask),
            self.pheno[self.eos_name].where(same_mask)
        )
    
        # If there is no cross-year data, return the same-year result.
        if cross_mask.sum().values < 1:
            return res_same
    
        res_cross = self.gs_v2(
            self.ds.where(cross_mask, drop=True), 
            self.pheno[self.sos_name].where(cross_mask, drop=True), 
            self.pheno[self.eos_name].where(cross_mask, drop=True)
            )
        return res_same.combine_first(res_cross)
    


if __name__ == "__main__":
    """
    Notes
    -----
    Start of season (SOS) corresponds to the leaf-out date (LOD),
    and end of season (EOS) corresponds to the leaf senescence date.
    """

    # GSCalculator is open-sourced and released in the `phenology` package.
    # A minimal, reproducible usage example is provided below (uncomment to run).
    #

    # import os
    # import xarray as xr
    # from tqdm import tqdm
    # from glob import glob
    # from datetime import datetime, timedelta
    # from joblib import Parallel, delayed
    # from phenology import GSCalculator


    # def main(input_file: str, pheno_file: str, output_file: str):
    #     ds = xr.open_dataset(input_file).load()
    #     pheno = xr.open_dataset(pheno_file)[['sos_mean', 'eos_mean']].load()
    #     res = GSCalculator(ds, pheno, sos_name='sos_mean', eos_name='eos_mean', need_extend=True).calc().astype('float32').sel(time=slice('2001-01-01', '2023-12-31'))
    #     old_names = list(res.data_vars)
    #     new_names = [f'gs_{var_name}' for var_name in old_names]
    #     res = res.rename(dict(zip(old_names, new_names)))
    #     res.to_netcdf(output_file)


    # input_file = f'data/data_0p10/data_common/gpp/gpp_modis_8day.nc'
    # pheno_file = f'data/data_0p10/data_common/phenology/pheno.0p10.mean.nc'
    # output_file = f'data/data_0p10/data_part1/gs_{os.path.basename(input_file)}'
    # main(input_file, pheno_file, output_file)

    pass