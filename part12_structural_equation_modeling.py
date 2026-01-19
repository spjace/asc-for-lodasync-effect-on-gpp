import pandas as pd
from tqdm import tqdm
from itertools import chain
from scipy.stats import zscore
from scipy.signal import detrend
from semopy import Model, calc_stats
from joblib import Parallel, delayed
from phenology.utils import split_array



def fit_sem_single(data, var_list, formula_sem):
    """
    Fit one SEM (semopy) for a single sample (e.g., one grid cell).

    Steps:
      1) Build a DataFrame from the input array and drop rows with any NaNs.
      2) Standardize each variable (z-score) and detrend along time.
      3) Fit the SEM specified by `formula_sem` using semopy's default optimizer.
      4) Return standardized path estimates and model fit statistics.

    Parameters
    ----------
    data : array-like, shape (T, n_vars)
        Time series for one sample, with columns aligned to `var_list`.
    var_list : list[str]
        Variable names (must match SEM formula variable names).
    formula_sem : str
        SEM specification in semopy/lavaan-like syntax.

    Returns
    -------
    (res, stats)
        res   : DataFrame of parameter estimates (standardized), or None if failed.
        stats : DataFrame of model fit statistics, or None if failed.
    """
    try:
        # Build DataFrame and remove incomplete time steps
        df = pd.DataFrame(data, columns=var_list).dropna()
        df = df.apply(zscore).dropna()
        df = df.apply(detrend).dropna()
        if len(df) < 10: 
            return None, None

        # Fit SEM
        model = Model(formula_sem)
        model.load_dataset(df)  # load data into the SEM model
        model.fit()             # fit parameters (default ML-based fitting in semopy)
        res = model.inspect(std_est=True)   # standardized estimates (comparable scales)
        stats = calc_stats(model)           # fit indices / diagnostics
        return res, stats
    except Exception as e:
        return None, None



def fit_sem_chunk(data_i, var_list, formula):
    """
    Fit SEM for a chunk (list/array) of samples.

    This is a thin wrapper to reduce joblib overhead by letting each worker
    process many samples per task instead of one sample per task.
    """
    return [fit_sem_single(data_j, var_list, formula) for data_j in data_i]



def sem_analysis(data_dict, formula):
    """
    Run SEM fitting for all samples in `data_dict` using parallel chunking.

    Parameters
    ----------
    data_dict : dict
        Expected keys:
          - 'data'    : array-like, shape (n_samples, T, n_features)
          - 'var_list': list[str], length = n_features
          - 'coords'  : spatial coordinates (e.g., lat/lon arrays)
          - 'mask'    : boolean mask mapping samples back to the spatial grid
    formula : str
        SEM specification string.

    Returns
    -------
    res_dict : dict
        Contains formula, per-sample SEM results, and metadata needed to map
        results back to space.
    """
    data_arr = data_dict['data']
    var_list = data_dict['var_list']
    coords = data_dict['coords']
    mask = data_dict['mask']

    # Split the samples into many chunks to balance parallelism and overhead
    chunk_list = split_array(data_arr, num_chunks=1200)

    # Parallel: each job fits SEM for one chunk
    res_stats_list = Parallel(n_jobs=200)(delayed(fit_sem_chunk)(data_i, var_list, formula) for data_i in tqdm(chunk_list))

    # Unzip pairs into two aligned lists
    res_stats_list = list(chain.from_iterable(res_stats_list))
    res_list, stats_list = map(list, zip(*res_stats_list))

    res_dict = {
        'formula': formula,
        'res_list': res_list,
        'stats_list': stats_list,
        'var_list': var_list,
        'coords': coords,
        'mask': mask,
        }
    
    return res_dict




if __name__ == '__main__':
    """
    Example usage: fit SEM across all samples and save the results to disk.

    Notes
    -----
    SEM syntax (semopy / lavaan style):
      "~"  : regression / directed path (A ~ B + C means B->A and C->A)
      "~~" : residual covariance (correlated residuals; no causal direction implied)
    """
    import os
    import pickle

    # Variable glossary (used in the SEM formula):
    # lod_async  — Leaf-out asynchrony (LODasync)
    # lod_mean   — leaf-out mean (LOD mean)

    # gs_csif    — Growing-season mean CSIF

    # gs_t2m     — Growing-season mean 2m air temperature
    # gs_tp      — Growing-season mean total precipitation
    # gs_ssrd    — Growing-season mean surface shortwave downwelling radiation

    # gs_lai     — Growing-season mean leaf area index
    # gs_fvc     — Growing-season mean fractional vegetation cover
    # gs_fapar   — Growing-season mean fraction of absorbed photosynthetically active radiation
    # gs_albedo  — Growing-season mean surface albedo
    # gs_ec      — Growing-season mean canopy evapotranspiration
    # gs_vod     — Growing-season mean vegetation optical depth

    # gs_co2     — Growing-season mean atmospheric CO2 concentration
    # gs_ndep    — Growing-season mean nitrogen deposition

    # SEM syntax (semopy / lavaan style):
    # "~"  : regression / directed path (A ~ B + C means B->A and C->A)
    # "~~" : residual covariance (correlated errors; no causal direction implied)

    # SEM specification

    formula = """ 
        gs_lai     ~ lod_async + gs_t2m + gs_tp + gs_ssrd + lod_mean
        gs_fvc     ~ lod_async + gs_t2m + gs_tp + gs_ssrd + lod_mean

        gs_albedo  ~ lod_async + gs_t2m + gs_tp + gs_ssrd + lod_mean 
        gs_fapar   ~ lod_async + gs_t2m + gs_tp + gs_ssrd + lod_mean + gs_fvc
        gs_ec      ~ lod_async + gs_t2m + gs_tp + gs_ssrd + lod_mean
        gs_vod     ~ lod_async + gs_t2m + gs_tp + gs_ssrd + lod_mean + gs_lai

        gs_lai    ~~ gs_fvc
        gs_albedo ~~ gs_lai
        gs_fapar  ~~ gs_albedo
        gs_fapar  ~~ gs_lai
        gs_fapar  ~~ gs_fvc


        gs_csif ~ lod_async + gs_lai + gs_fvc + gs_albedo + gs_fapar + gs_ec + gs_vod + gs_co2 + gs_ndep
        """

    # Load preprocessed SEM input:
    # data_dict = {
    #   'data'    : data,     # (n_samples, time, n_features) float32
    #   'mask'    : mask,     # (rows, cols) boolean mask of valid grid cells
    #   'coords'  : coords,   # latitude/longitude arrays
    #   'var_list': var_list  # length = n_features, must include all variables in `formula`
    # }
    data_dict = pickle.load(open('data/data_0p10/data_sem.pkl', 'rb'))

    # Run SEM for all grid cells (parallel inside sem_analysis)
    result_dict = sem_analysis(data_dict, formula)

    # Save outputs
    output_file = 'result/result_0p10/result_sem.pkl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pickle.dump(result_dict, open(output_file, 'wb'))