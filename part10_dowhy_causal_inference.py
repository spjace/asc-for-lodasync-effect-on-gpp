import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from graphviz import Digraph
from dowhy import CausalModel
from scipy.stats import zscore
from scipy.signal import detrend
from joblib import Parallel, delayed
from phenology.utils import sequence_split



def build_causal_graph(treatment='lod_async', outcome='gs_csif', common_causes=[],  instruments=[], mediator=[], covars=[], **kwargs):
    """
    Build a causal DAG (Graphviz Digraph) for DoWhy causal-effect estimation.

    The constructed graph encodes an assumed causal structure among:
      - treatment (T): the exposure variable whose causal effect is to be estimated
      - outcome (Y): the response variable
      - common_causes (C): confounders that affect both T and Y (C -> T and C -> Y)
      - instruments (Z): variables that affect T (Z -> T) in this graph
      - mediator (M): mediators on the pathway from T to Y (T -> M -> Y)
      - covars (W): outcome-only covariates that affect Y (W -> Y)

    Node styling:
      - treatment/outcome: box
      - confounders/covariates/mediators: ellipse
      - instruments: diamond

    Parameters
    ----------
    treatment : str
        Treatment variable name (T).
    outcome : str
        Outcome variable name (Y).
    common_causes : list[str] or None
        Confounders affecting both treatment and outcome.
    instruments : list[str] or None
        Instrumental variables affecting treatment (as encoded in this DAG).
    mediator : list[str] or None
        Mediators on the causal pathway from treatment to outcome.
    covars : list[str] or None
        Outcome-only covariates affecting outcome.
    **kwargs
        Extra keyword arguments are ignored (kept for flexible calling).

    Returns
    -------
    graphviz.Digraph
        Graphviz Digraph object representing the causal graph (use `.source` for DOT text).
    """
    dot = Digraph()

    # Add treatment and outcome nodes (boxes)
    for node in [treatment, outcome]:
        dot.node(node, node, shape='box')

    # Direct causal edge: T -> Y (always included)
    dot.edge(treatment, outcome)

    # Confounders: C -> T and C -> Y
    for node in common_causes:
        dot.node(node, node, shape='ellipse')
        dot.edge(node, treatment)
        dot.edge(node, outcome)

    # Instruments: Z -> T
    for node in instruments:
        dot.node(node, node, shape='diamond')
        dot.edge(node, treatment)

    # Outcome-only covariates: W -> Y
    for node in covars:
        dot.node(node, node, shape='ellipse')
        dot.edge(node, outcome)

    # Mediators: T -> M -> Y
    for node in mediator:
        dot.node(node, node, shape='ellipse')
        dot.edge(treatment, node)
        dot.edge(node, outcome)

    return dot


def estimate_dowhy_single(data_i, var_list, outcome, treatment, dot, is_detrend):
    """
    Estimate the causal effect for a single grid cell (or site) using DoWhy.

    Workflow (one location):
      1) Convert local time-series array to a DataFrame with named columns.
      2) Drop missing rows, z-score standardize each column, optionally detrend.
      3) Build a DoWhy CausalModel using the provided causal graph (DOT).
      4) Identify the estimand and estimate effect using backdoor linear regression.
      5) Return the estimated causal effect and its p-value.

    Parameters
    ----------
    data_i : np.ndarray
        Local array with shape (T, n_vars), where T is the number of time steps.
    var_list : list
        Variable names aligned with columns of `data_i`.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    dot : graphviz.Digraph
        Causal DAG; DoWhy uses its DOT source via `dot.source`.
    is_detrend : bool
        Whether to detrend each variable time series before estimation.

    Returns
    -------
    tuple or np.ndarray
        On success: (causal_effect, p_value).
        If sample size is insufficient: array([nan, nan]).
        If DoWhy fails: (nan, nan).
        (Note: return type differs by branch; kept unchanged to avoid altering tested logic.)
    """
    # Build a tidy table and drop rows with any missing values
    df = pd.DataFrame(data=data_i, columns=var_list).dropna()

    # Z-score standardization (stabilize regression / comparable scales)
    df = df.apply(zscore).dropna()
    if is_detrend:
        df = df.apply(detrend)

    # Minimum sample-size guard
    if len(df) < 10:
        return np.array([np.nan, np.nan])
    
    try:
        # Identify the estimand (backdoor set etc.)
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=dot.source
        )

        # Estimate the causal effect (backdoor linear regression)
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression") 

        # Extract effect size and significance
        causal_effect = estimate.value
        p_value = estimate.test_stat_significance()['p_value'].squeeze()
        return causal_effect, p_value
    
    except Exception as e:
        return np.nan, np.nan


def dowhy_single(data_i_list, var_list, outcome, treatment, dot, is_detrend):
    """
    Chunk-level wrapper for parallel computation.

    Each worker receives a list of locations (grid cells / sites) and applies
    `estimate_dowhy_single` to each item.

    Parameters
    ----------
    data_i_list : list[np.ndarray]
        List of local arrays, each shape (T, n_vars).
    var_list : list
        Variable names aligned with columns of each local array.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    dot : graphviz.Digraph
        Causal DAG.
    is_detrend : bool
        Whether to detrend each time series before estimation.

    Returns
    -------
    list
        List of (causal_effect, p_value) pairs, one per location.
    """
    res_list_i = [estimate_dowhy_single(data_i, var_list, outcome, treatment, dot, is_detrend) for data_i in data_i_list]
    return res_list_i


def dowhy_main(ds:xr.Dataset, outcome, treatment, var_list, dot, n_jobs=-1, is_detrend=False):
    """
    Run grid-cell DoWhy causal-effect estimation for a spatiotemporal xarray Dataset.

    This function:
      1) Applies a spatial coverage filter using `lod_area` (mean over time).
      2) Reorders data into a numpy array with shape (lat, lon, time, var).
      3) Enforces "complete-case by time": if any variable is NaN at a given (lat, lon, time),
         all variables are set to NaN at that (lat, lon, time).
      4) Selects grid cells with sufficient valid data coverage.
      5) Runs DoWhy estimation in parallel across valid grid cells.
      6) Scatters results back to the full (lat, lon) grid and returns an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing at least `var_list` and `lod_area`.
        Expected dims include 'time', 'lat', 'lon'.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    var_list : list[str]
        Variables used for modeling (must include outcome, treatment, and other graph variables).
    dot : graphviz.Digraph
        Causal DAG used by DoWhy.
    n_jobs : int, default -1
        Number of joblib workers. Use -1 to use all cores.
    is_detrend : bool, default False
        Whether to detrend each variable time series within each grid cell.

    Returns
    -------
    xarray.Dataset
        Dataset with:
          - 'r'     : estimated causal effect (DoWhy estimate.value)
          - 'p-val' : p-value from DoWhy significance test
    """
    # Spatial coverage filter: keep only grid cells with enough valid LOD coverage
    ds_area = ds['lod_area'].mean(dim='time', skipna=True).squeeze()
    ds = ds[var_list].where(ds_area>0.2)
    data = ds.to_array().values.squeeze().transpose(2, 3, 1, 0)

    # If any variable is NaN at a given (lat, lon, time), mark all variables as NaN at that position
    data[np.isnan(data).any(axis=-1)] = np.nan

    # Select grid cells with sufficient valid data coverage (require NaN fraction < 10% across (time, var) for each (lat, lon))
    mask = np.isnan(data).sum(axis=(-2, -1)) < 0.1*data.shape[-2]*data.shape[-1]
    valid_data = data[mask] # shape: (n_valid_cells, time, var)

    # Split valid grid cells into chunks for parallel processing
    valid_data_list = sequence_split(valid_data, 600)

    # Parallel DoWhy estimation (each worker handles a chunk/list)
    res_list = Parallel(n_jobs=n_jobs)(delayed(dowhy_single)(data_i_list, var_list, outcome, treatment, dot, is_detrend) for data_i_list in tqdm(valid_data_list))
    res_list = [item for res_list_i in res_list for item in res_list_i]
    res_arr = np.array(res_list) # (1635698, 2)

    # Scatter results back to (lat, lon)
    result = np.full((mask.shape[0], mask.shape[1], res_arr.shape[-1]), np.nan)
    result[mask] = res_arr

    result_ds = xr.Dataset(data_vars={
        'r':(('lat', 'lon'), result[:, :, 0]),      # causal effect estimate
        'p-val':(('lat', 'lon'), result[:, :, 1]),  # significance p-value
        }, coords={'lat': ds['lat'], 'lon': ds['lon']}
    )
    
    return result_ds



if __name__ == '__main__':
    """
    Example usage: run DoWhy causal-effect estimation under multiple graph specifications.

    Notes
    -----
    - Prefix conventions depend on your dataset:
        gs_* : growing-season mean variables
        sm_* : spring-mean variables
        ma_* : variables computed over another (preseason / moving-average) window (as defined in your pipeline)
    - `ds` must include `lod_area` for the coverage filter.
    - Output variable 'r' stores the DoWhy causal effect estimate (NOT correlation r).
    """
    # Example usage
    # Note: sm = spring mean; gs = growing-season mean

    # Covariate sets used in different specifications
    covars1 = ['gs_t2m', 'gs_tp', 'gs_ssrd', 'gs_co2', 'gs_nhx', 'lod_mean']
    covars3 = ['gs_co2', 'gs_nhx', 'lod_mean']
    args_list = [
        dict(key="f1", treatment="lod_async", outcome="gs_csif", covars=covars1, common_causes=[], instruments=[], mediator=[]),
        dict(key="f2", treatment="lod_async", outcome="gs_csif", covars=covars1, common_causes=[], instruments=[], mediator=["gs_lai", "gs_fapar", "gs_fvc"]),
        dict(key="f3", treatment="lod_async", outcome="gs_csif", covars=covars3, common_causes=["ma_t2m", "ma_tp", "ma_ssrd"], instruments=[], mediator=[]),
        dict(key="f4", treatment="lod_async", outcome="gs_csif", covars=covars1, common_causes=[], instruments=["sm_t2m", "sm_tp", "sm_ssrd"], mediator=[]),
    ]

    for args in args_list:
        # Build causal graph (DAG) for DoWhy; pass dict as keyword arguments.
        args['dot'] = build_causal_graph(**args)
        # Build a consistent variable list: outcome + treatment + all graph-related variables.
        args['var_list'] = [args['outcome'], args['treatment']] + sorted(args['covars'] + args['common_causes'] + args['mediator'] + args['instruments'])


    # Run each specification and save output
    for args in args_list:
        outcome, treatment, var_list, dot, key = args['outcome'], args['treatment'], args['var_list'], args['dot'], args['key']
        output_file = f"result/result_0p10/part6_dowhy/dowhy_{key}.{'.'.join(var_list)}.nc"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file): 
            print(output_file, 'Already exists! Loading...')
            result_ds = xr.open_dataset(output_file)
        else:
            
            ds = xr.open_dataset('data/data_0p10/data_part1.nc')
            result_ds = dowhy_main(ds, outcome, treatment, var_list, dot, n_jobs=200, is_detrend=True)
            result_ds.to_netcdf(output_file)
            print(output_file, 'Finished!')